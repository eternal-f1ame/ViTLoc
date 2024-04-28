import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
import time
import math
from tqdm import tqdm, trange
from models.ray_utils import *
from models.nerf import img2mse, mse2psnr,to8b
from einops import rearrange, reduce, repeat
import pdb

''' 
Render Procedure:
render()->batchify_rays()->render_rays()->raw2outputs()->sample_pdf()
'''

DEBUG = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    
    dists = z_vals[...,1:] - z_vals[...,:-1]

    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples] the last delta is infinity (1e10)

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1) # why raw2outputs_NeRFW doesn't have this step?

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def raw2outputs_NeRFW(raw, z_vals, rays_d, raw_noise_std=0, output_transient=False, beta_min=0.1, white_bkgd=False, test_time=False, static_only=True, typ="coarse"):
    ''' Convert NeRFW fine network output to rendered colors
    This version is implemented in nerf_pl https://github.com/kwea123/nerf_pl/tree/nerfw
    Inputs:
        raw: torch.Tensor() [N_rays, N_samples, 9]

    '''

    if typ=="coarse" and test_time:
        static_sigmas = raw[..., 0]
        transient_sigmas = None
    else:
        if output_transient==False:
            N_rays, N_samples, ch = raw.size()
            ch_rgbs = ch - 1
        else:
            N_rays, N_samples, ch = raw.size()
            ch_rgbs = (ch - 3) // 2

        static_rgbs = raw[..., :ch_rgbs]
        static_sigmas = raw[..., ch_rgbs]
        if output_transient:
            transient_rgbs = raw[..., ch_rgbs + 1: 2 * ch_rgbs + 1]
            transient_sigmas = raw[..., 2 * ch_rgbs + 1]
            transient_betas = raw[..., 2 * ch_rgbs + 2]
        else:
            transient_sigmas = None

    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 1e2 * torch.ones_like(deltas[:, :1])
    
    deltas = torch.cat([deltas, delta_inf], -1)

    if output_transient:
        static_alphas = 1-torch.exp(-deltas*static_sigmas)
        transient_alphas = 1-torch.exp(-deltas*transient_sigmas)
        alphas = 1-torch.exp(-deltas*(static_sigmas+transient_sigmas))
    else:
        noise = torch.randn_like(static_sigmas) * raw_noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(static_sigmas+noise))

    alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1)
    transmittance = torch.cumprod(alphas_shifted[:, :-1], -1)

    if output_transient:
        static_weights = static_alphas * transmittance
        transient_weights = transient_alphas * transmittance

    weights = alphas * transmittance
    weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')
    acc_map = weights_sum

    if typ=="coarse" and test_time:
        rgb_map=None
        disp_map=None
        depth_map=None
        beta=None
        return rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta

    if output_transient:
        static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                                'n1 n2 c -> n1 c', 'sum')
        if white_bkgd:
            static_rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
        
        transient_rgb_map = \
            reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                   'n1 n2 c -> n1 c', 'sum')
        beta = reduce(transient_weights*transient_betas, 'n1 n2 -> n1', 'sum')

        beta += beta_min

        rgb_fine = static_rgb_map + transient_rgb_map
        rgb_map = rgb_fine

        if test_time and static_only:
            static_alphas_shifted = \
                torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)
            static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
            static_weights_ = static_alphas * static_transmittance
            static_rgb_map_ = \
                reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1')*static_rgbs,
                       'n1 n2 c -> n1 c', 'sum')
            if white_bkgd:
                static_rgb_map_ += 1-rearrange(weights_sum, 'n -> n 1')
            rgb_fine = static_rgb_map_
            depth_map = reduce(static_weights_*z_vals, 'n1 n2 -> n1', 'sum')
            disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
            return rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta

    else:
        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                             'n1 n2 c -> n1 c', 'sum')
        if white_bkgd:
            rgb_map += 1-rearrange(weights_sum, 'n -> n 1')

        beta=torch.Tensor([0]).repeat(weights_sum.shape[0])

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    return rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                i_epoch=-1,
                embedding_a=None,
                embedding_t=None,
                test_time=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]
    viewdirs = ray_batch[:,8:11] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1]
    img_idxs = ray_batch[...,11:]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    if i_epoch>=0:
        raw = network_query_fn(pts, viewdirs, None, network_fn, 'coarse', None, None, False, test_time=test_time, epoch=i_epoch)
    else:
        raw = network_query_fn(pts, viewdirs, None, network_fn, 'coarse', None, None, False, test_time=test_time)


    rgb_map, disp_map, acc_map, weights, depth_map, _, _ = raw2outputs_NeRFW(raw, z_vals, rays_d, raw_noise_std, white_bkgd, test_time=test_time, typ="coarse")
    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        output_transient = True

        if i_epoch>=0:
            raw = network_query_fn(pts, viewdirs, img_idxs, network_fine, 'fine', embedding_a, embedding_t, output_transient, test_time=test_time, epoch=i_epoch)
        else:
            raw = network_query_fn(pts, viewdirs, img_idxs, network_fine, 'fine', embedding_a, embedding_t, output_transient, test_time=test_time)


        rgb_map, disp_map, acc_map, weights, depth_map, transient_sigmas, beta = raw2outputs_NeRFW(raw, z_vals, rays_d, raw_noise_std, output_transient, network_fine.beta_min, white_bkgd, test_time, typ="fine")

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw

    if N_importance > 0 and test_time:
        pass

    elif N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)
        ret['transient_sigmas'] = transient_sigmas
        ret['beta'] = beta

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, img_idx=torch.Tensor(0),
                  **kwargs):
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) 

    if img_idx.shape[0] != rays.shape[0]:
        img_idx = img_idx.repeat(rays.shape[0],1)
    rays = torch.cat([rays, img_idx], 1) 

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(args, render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, single_gt_img=False, img_ids=torch.Tensor(0)):

    H, W, focal = hwf

    if render_factor!=0:
        H = int(H//render_factor)
        W = int(W//render_factor)
        focal = focal/render_factor

    rgbs = []
    disps = []
    rgb0s = []
    psnr = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        t = time.time()
        rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], img_idx=img_ids[i], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None:
            if single_gt_img:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs)))
            else:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            psnr.append(p)

        if savedir is not None:

            rgb8_f = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8_f)

            rgb_gt = to8b(gt_imgs[i])
            filename = os.path.join(savedir, '{:03d}_GT.png'.format(i))
            imageio.imwrite(filename, rgb_gt)

            rgb_disp = to8b(disps[-1] / np.max(disps[-1]))
            filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename, rgb_disp)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    psnr = np.mean(psnr,0)
    print("Mean PSNR of this run is:", psnr)
    return rgbs, disps

def render_test(args, train_dl, val_dl, hwf, start, render_kwargs_test, decoder_coarse=None, decoder_fine=None):

    trainsavedir = os.path.join(args.basedir, args.expname, 'evaluate_train_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(trainsavedir, exist_ok=True)
    images_train = []
    poses_train = []
    index_train = []
    for img, pose, img_idx in train_dl:
        img_val = img.permute(0,2,3,1)
        pose_val = torch.zeros(1,4,4)
        pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4]
        pose_val[0,3,3] = 1.
        images_train.append(img_val)
        poses_train.append(pose_val)
        index_train.append(img_idx)

    images_train = torch.cat(images_train, dim=0).numpy()
    poses_train = torch.cat(poses_train, dim=0).to(device)
    index_train = torch.cat(index_train, dim=0).to(device)
    print('train poses shape', poses_train.shape)

    with torch.no_grad():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        rgbs, disps = render_path(args, poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train)
        torch.set_default_tensor_type('torch.FloatTensor')
    print('Saved train set')
    if args.render_video_train:
        print('Saving trainset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.expname, '{}_trainset_{:06d}_'.format(args.expname, start))
        imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'train_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    del images_train
    del poses_train
    torch.cuda.empty_cache()

    testsavedir = os.path.join(args.basedir, args.expname, 'evaluate_val_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)
    images_val = []
    poses_val = []
    index_val = []
    for img, pose, img_idx in val_dl: 
        img_val = img.permute(0,2,3,1)
        pose_val = torch.zeros(1,4,4)
        pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4]
        pose_val[0,3,3] = 1.
        images_val.append(img_val)
        poses_val.append(pose_val)
        index_val.append(img_idx)

    images_val = torch.cat(images_val, dim=0).numpy()
    poses_val = torch.cat(poses_val, dim=0).to(device)
    index_val = torch.cat(index_val, dim=0).to(device)
    print('test poses shape', poses_val.shape)
    with torch.no_grad():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        rgbs, disps = render_path(args, poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val)
        torch.set_default_tensor_type('torch.FloatTensor')
    print('Saved test set')
    if args.render_video_test:
        print('Saving testset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.expname, '{}_test_{:06d}_'.format(args.expname, start))
        imageio.mimwrite(moviebase + 'test_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'test_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    del images_val
    del poses_val
    return
