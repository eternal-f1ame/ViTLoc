import argparse
import torch
import numpy as np
import json
import logging
from utils import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
import wandb

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model_name", help="name of model to create (e.g. posenet, transposenet)")
    arg_parser.add_argument("mode", help="train or evaluate")
    arg_parser.add_argument("dataset_path", help="path to the dataset location")
    arg_parser.add_argument("labels_file", help="file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained model (should match the model indicated in model_name)")
    arg_parser.add_argument("--experiment", help="a short string describing the experiment/commit")

    args = arg_parser.parse_args()
    utils.init_logger()

    logging.info(f"Starting {args.mode} for {args.model_name}")
    if args.experiment:
        logging.info(f"Experiment details: {args.experiment}")
    logging.info(f"Dataset: {args.dataset_path}")
    logging.info(f"Labels file: {args.labels_file}")

    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Configuration:\n{}".format('\n'.join([f"\t{k}: {v}" for k, v in config.items()])))

    wandb.init(project="ViTLoc", name="ThreeFormer - Final - Full Cambridge Five", config=config)

    use_cuda = torch.cuda.is_available()
    device = 'cpu' if not use_cuda else config.get('device_id')
    torch.manual_seed(0)
    if use_cuda:
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
    np.random.seed(2)

    model = get_model(args.model_name, config).to(device)

    if args.checkpoint_path:
       model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
       logging.info(f"Initialized from checkpoint: {args.checkpoint_path}")

    if args.mode == 'train':
        model.train()

        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, param in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                    param.requires_grad_(False)

        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss = torch.nn.NLLLoss()

        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=config.get('lr'), eps=config.get('eps'), weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=config.get('lr_scheduler_step_size'), gamma=config.get('lr_scheduler_gamma'))

        no_augment = config.get("no_augment")
        transform = utils.test_transforms.get('baseline') if no_augment else utils.train_transforms.get('baseline')

        equalize_scenes = config.get("equalize_scenes")
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, equalize_scenes)
        loader_params = {'batch_size': config.get('batch_size'), 'shuffle': True, 'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        batch_size = config.get("batch_size")

        checkpoint_prefix = join(utils.create_output_dir('checkpoints'), utils.get_stamp_from_log())
        n_total_samples = 0
        scene_error = 0
        loss_vals = []
        sample_count = []
        running_loss = 0.0

        for epoch in range(n_epochs):

            n_samples = 0
            
            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                
                n_total_samples += batch_size
                n_samples += batch_size

                if freeze:
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()

                optim.zero_grad()

                if freeze:
                    res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('pose')
                est_scene_log_distr = res.get('scene_log_distr')
                if est_scene_log_distr is not None:
                    pose_error, s_x, s_q = pose_loss(est_pose, gt_pose)
                    scene_error = nll_loss(est_scene_log_distr, gt_scene)
                else:
                    pose_error, s_x, s_q = pose_loss(est_pose, gt_pose)

                criterion = pose_error + scene_error
                running_loss = criterion.item()
                loss_vals.append(criterion.item())

                # Failsafe for Adverserial Prediction
                if running_loss/batch_size > 500:
                    logging.info(f"UNEXPECTED PREDICTION at Batch-{batch_idx+1} EPOCH-{epoch}")
                    continue
                
                sample_count.append(n_total_samples)

                criterion.backward()
                optim.step()

                if batch_idx % n_freq_print == 0:
                    position_error, orientation_error = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info(f"[Batch-{batch_idx+1}/Epoch-{epoch+1}] Running camera pose loss: {running_loss/batch_size:.3f}, "
                                    f"Position error: {position_error.mean().item():.2f}[m], Orientation error: {orientation_error.mean().item():.2f}[deg]")
                    wandb.log({ 'epoch': epoch+1,
                                'n_samples': n_samples,
                                'pose_loss': pose_error,
                                'scene_loss': scene_error,
                                'net_loss': running_loss/n_total_samples,
                                'position_error': position_error.mean().item(),
                                'orientation_error': orientation_error.mean().item(),
                                's_x': s_x,
                                's_q': s_q
                                })
            
                running_loss = 0

            n_samples = 0



            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), f"{checkpoint_prefix}_checkpoint-{epoch}.pth")

            scheduler.step()

        logging.info('Training complete')
        torch.save(model.state_dict(), f"{checkpoint_prefix}_final.pth")

    else:
        model.eval()

        transform = utils.test_transforms.get('baseline')
        dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_scene = minibatch.get('scene')
                minibatch['scene'] = None

                gt_pose = minibatch.get('pose').to(dtype=torch.float32)

                tic = time.time()
                est_pose = model(minibatch).get('pose')
                toc = time.time()

                posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
