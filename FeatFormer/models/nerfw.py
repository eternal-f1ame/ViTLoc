import os
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from models.nerf import run_network, NeRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def run_network_NeRFW(inputs, viewdirs, ts, fn, embed_fn, embeddirs_fn, 
                    typ, embedding_a, embedding_t, output_transient, 
                    netchunk=1024*64, test_time=False):

    out_chunks = []
    N_rays, N_samples = inputs.shape[0], inputs.shape[1]

    if typ == 'coarse' and test_time:
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        for i in range(0, inputs_flat.shape[0], netchunk):
            embedded_inputs = [embed_fn(inputs_flat[i: i+netchunk])]
            out_chunks += [fn(torch.cat(embedded_inputs, 1), sigma_only=True)]
        out = torch.cat(out_chunks, 0)
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]])
        return out
    if typ == 'coarse':
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        for i in range(0, inputs_flat.shape[0], netchunk):
            embedded_inputs = [embed_fn(inputs_flat[i: i+netchunk]), embeddirs_fn(input_dirs_flat[i:i+netchunk])]
            out_chunks += [fn(torch.cat(embedded_inputs, 1), output_transient=output_transient)]

        out = torch.cat(out_chunks, 0)
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]])
        return out

    elif typ == 'fine':
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        a_embedded = embedding_a(ts.long())

        t_embedded = embedding_t(ts.long())

        if len(a_embedded.size()) == 3:
            a_embedded = a_embedded.reshape(N_rays, -1)
        if len(t_embedded.size()) == 3:
            t_embedded = t_embedded.reshape(N_rays, -1)

        a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)
        t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)

        for i in range(0, inputs_flat.shape[0], netchunk):
            embedded_inputs = [embed_fn(inputs_flat[i: i+netchunk]), embeddirs_fn(input_dirs_flat[i:i+netchunk])]

            embedded_inputs += [a_embedded_[i:i+netchunk]]
            embedded_inputs += [t_embedded_[i:i+netchunk]]

            out_chunks += [fn(torch.cat(embedded_inputs, 1), output_transient=output_transient)]
        out = torch.cat(out_chunks, 0)
        out = torch.reshape(out, list(inputs.shape[:-1]) + [out.shape[-1]])
        return out

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.N_freqs = 0
        self.N = -1
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        self.N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=self.N_freqs) 

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        if self.kwargs['max_freq_log2'] != 0:
            ret = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        else:
            ret = inputs
        return ret

    def get_embed_weight(self, epoch, num_freqs, N):
        ''' Nerfie Paper Eq.(8) '''
        alpha = num_freqs * epoch / N
        W_j = []
        for i in range(num_freqs):
            tmp = torch.clamp(torch.Tensor([alpha - i]), 0, 1)
            tmp2 = (1 - torch.cos(torch.Tensor([np.pi]) * tmp)) / 2
            W_j.append(tmp2)
        return W_j

    def embed_DNeRF(self, inputs, epoch):
        ''' Nerfie paper section 3.5 Coarse-to-Fine Deformation Regularization '''
        W_j = self.get_embed_weight(epoch, self.N_freqs, self.N)
        
        
        out = []
        for fn in self.embed_fns:
            out.append(fn(inputs))

        for i in range(len(W_j)):
            out[2*i+1] = W_j[i] * out[2*i+1]
            out[2*i+2] = W_j[i] * out[2*i+2]
        ret = torch.cat(out, -1)
        return ret

    def update_N(self, N):
        self.N=N


def get_embedder(multires, i=0, reduce_mode=-1, epochToMaxFreq=-1):
    if i == -1:
        return nn.Identity(), 3
    
    if reduce_mode == 0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : (multires-1)//2,
                    'num_freqs' : multires//2,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
    elif reduce_mode == 1:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : 0,
                    'num_freqs' : 0,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
    elif reduce_mode == 2:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
    else:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }

    embedder_obj = Embedder(**embed_kwargs)
    if reduce_mode == 2:
        embedder_obj.update_N(epochToMaxFreq)
        embed = lambda x, epoch, eo=embedder_obj: eo.embed_DNeRF(x, epoch)
    else: 
        embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim, embedder_obj

class NeRFW(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.1, out_ch_size=3):

        super().__init__()
        torch.manual_seed(0)
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        
        if out_ch_size == 3:
            self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        else:
            self.static_rgb = nn.Sequential(nn.Linear(W//2, out_ch_size))

        if self.encode_transient:
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(W//2, W//2), nn.ReLU(True))
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            if out_ch_size == 3:
                self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
                self.transient_rgb = nn.Sequential(nn.Linear(W//2, out_ch_size))
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):

        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding)
        static = torch.cat([static_rgb, static_sigma], 1)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding)
        transient_rgb = self.transient_rgb(transient_encoding)
        transient_beta = self.transient_beta(transient_encoding)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1)

        return torch.cat([static, transient], 1)

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """

    if args.reduce_embedding==2:
        embed_fn, input_ch, embedder_obj = get_embedder(args.multires, args.i_embed, args.reduce_embedding, args.epochToMaxFreq)
    else:
        embed_fn, input_ch, _ = get_embedder(args.multires, args.i_embed, args.reduce_embedding)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        if args.reduce_embedding==2:
            if args.no_DNeRF_viewdir:
                raise NotImplementedError
                embeddirs_fn, input_ch_views, _ = get_embedder(args.multires_views, args.i_embed)
            else:
                embeddirs_fn, input_ch_views, embedddirs_obj = get_embedder(args.multires_views, args.i_embed, args.reduce_embedding, args.epochToMaxFreq)
        else:
            embeddirs_fn, input_ch_views, _ = get_embedder(args.multires_views, args.i_embed, args.reduce_embedding)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    device = torch.device("cuda")

    encode_a = True
    encode_t = True
    if encode_a:
        if args.encode_hist:
            embedding_a = torch.nn.Embedding(args.N_vocab, 5)
        embedding_a = embedding_a.to(device)
    if encode_t:
        if args.encode_hist:
            embedding_t = torch.nn.Embedding(args.N_vocab, 2)
        embedding_t = embedding_t.to(device)

    if args.NeRFH:
        model = NeRFW('coarse', D=args.netdepth, W=args.netwidth, skips=skips, in_channels_xyz=input_ch, in_channels_dir=input_ch_views)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.NeRFH:
            model_fine = NeRFW('fine', D=args.netdepth, W=args.netwidth, skips=skips, 
                in_channels_xyz=input_ch, in_channels_dir=input_ch_views,
                encode_appearance=True, encode_transient=True,
                in_channels_a=args.in_channels_a, in_channels_t=args.in_channels_t)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        if args.multi_gpu:
            model_fine = torch.nn.DataParallel(model_fine).to(device)
        else:
            model_fine = model_fine.to(device)
        grad_vars += list(model_fine.parameters())
        grad_vars += list(embedding_a.parameters())
        grad_vars += list(embedding_t.parameters())

    if args.NeRFH:
        network_query_fn = lambda inputs, viewdirs, ts, network_fn, \
                typ, embedding_a, embedding_t, output_transient, test_time : run_network_NeRFW(inputs, viewdirs, ts, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                typ=typ, 
                                                                embedding_a=embedding_a, 
                                                                embedding_t=embedding_t, 
                                                                output_transient=output_transient,
                                                                netchunk=args.netchunk,
                                                                test_time=test_time)
    else:
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    if args.no_grad_update:
        grad_vars = None
        optimizer = None
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            embedding_a.load_state_dict(ckpt['embedding_a_state_dict'])
            embedding_t.load_state_dict(ckpt['embedding_t_state_dict'])


    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'embedding_a' : embedding_a,
        'embedding_t' : embedding_t,
        'test_time' : False
    }

    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['test_time'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
