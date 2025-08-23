import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch.cuda
import torch.nn as nn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.distributed as dist
import yaml
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from carhynet.models import HyNetnetFeature2D
from models.gmatcher import GMatcher
from models.agc import *
from utils.common import increment_path, init_seeds, reduce_tensor, download_base_files, time_synchronized, test_model, ModelEMA, sift_forward
from utils.preprocess_utils import torch_find_matches
from utils.dataset import COCO_loader, COCO_valloader, collate_batch

def change_lr(epoch, config, optimizer):
    changed_lr = config['optimizer_params']['lr']
    if epoch >= config['optimizer_params']['step_epoch']:
        changed_lr *=  (config['optimizer_params']['step_value'] ** (epoch - config['optimizer_params']['step_epoch']))
    print(f"Chang learning rate to {changed_lr}")
    for g in optimizer.param_groups: g['lr'] = changed_lr

def train(config, args, rank, device, is_distributed=False):
    if is_distributed: print(f"[{rank}] GPU: {torch.cuda.get_device_name(device)}")
    save_dir = Path(config['train_params']['save_dir'])
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    results_file = None
    if rank in [0, -1]: results_file = open(save_dir / "results.txt", 'a')
    with open(save_dir / 'config.yaml', 'w') as file: yaml.dump(config, file, sort_keys=False)
    init_seeds(config['train_params']['init_seed'])
    config['train_params']['transformer_layers'] = ['self', 'cross'] * config['train_params']['tf_layers']
    gmodel = GMatcher(config['train_params']).to(device)
    carhynet = HyNetnetFeature2D()
    start_epoch = config['train_params']['start_epoch'] if config['train_params']['start_epoch'] > -1 else 0
    if is_distributed and config['train_params']['sync_bn']: gmodel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gmodel).to(device)
    pg0, pg1, pg2 = [], [], []
    for k, v in gmodel.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if hasattr(v, 'bin_score'):
            pg0.append(v.bin_score)
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.BatchNorm1d) or isinstance(v, nn.SyncBatchNorm):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    if config['optimizer_params']['opt_type'].lower() == "adam":
        optimizer = optim.Adam(pg0, lr=config['optimizer_params']['lr'], betas=(0.9, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=config['optimizer_params']['lr'], momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': config['optimizer_params']['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    ema = None
    if config['train_params']['use_ema']:
        ema = ModelEMA(gmodel) if rank in [-1, 0] else None
        print("Keeping track of weights in ema..")
    if is_distributed: gmodel = DDP(gmodel, device_ids=[rank], output_device=rank)
    train_dataset = COCO_loader(config['dataset_params'], typ="train", limit=args.limit, color=True)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train_params']['batch_size'],
                                                    num_workers=config['train_params']['num_workers'],
                                                    shuffle = False if is_distributed else True,
                                                    sampler=sampler,
                                                    collate_fn=collate_batch,
                                                    pin_memory=True)
    num_batches = len(train_dataloader)
    if rank in [-1, 0]:
        val_dataset = COCO_valloader(config['dataset_params'], color=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=1,
                                                     num_workers=0,
                                                     sampler=None,
                                                     collate_fn=collate_batch,
                                                     pin_memory=True)
    num_epochs = config['train_params']['num_epochs']
    best_val_score = 1e-10
    if rank in [-1, 0]:
        print("Started training for {} epochs".format(num_epochs))
        print("Number of batches: {}".format(num_batches))
    warmup_iters = config['optimizer_params']['warmup_epochs'] * num_batches
    change_lr(start_epoch, config, optimizer)
    best_min_loss = 1e9
    for epoch in range(start_epoch, num_epochs):
        print("Started epoch: {} in rank {}".format(epoch + 1, rank))
        gmodel.train()
        if rank != -1: train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(train_dataloader)
        if rank in [-1, 0]: pbar = tqdm(pbar, total=num_batches)
        optimizer.zero_grad()
        mloss = torch.zeros(6, device=device)
        if rank in [-1, 0]: print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'Iteration','PosLoss', 'NegLoss', 'TotLoss', 'Dtime', 'Ptime', 'Mtime'))
        t5 = time_synchronized()
        for i, (orig_warped, homographies) in pbar:
            ni = i + num_batches * epoch
            if ni < warmup_iters:
                xi = [0, warmup_iters]
                for _, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [0.0, config['optimizer_params']['lr']])
            t1 = time_synchronized()
            homographies = homographies.to(device, non_blocking=True)
            midpoint = len(orig_warped) // 2
            with torch.no_grad():
                all_match_index_0, all_match_index_1, all_match_index_2 = torch.empty(0, dtype=torch.int64,device=homographies.device), torch.empty(0, dtype=torch.int64,device=homographies.device), torch.empty(0, dtype=torch.int64,device=homographies.device)
                t2 = time_synchronized()
                superpoint_results = sift_forward({'homography': homographies, 'image': orig_warped, 'max_keypoints': 2048, 'carhynet': carhynet, 'is_train': True}, device=device)
                keypoints = torch.stack(superpoint_results['keypoints'], 0).to(device)
                descriptors = torch.stack(superpoint_results['descriptors'], 0).to(device)
                scores = torch.stack(superpoint_results['scores'], 0).to(device)
                keypoints0, keypoints1 = keypoints[:midpoint, :, :], keypoints[midpoint:, :, :]
                descriptors0, descriptors1 = descriptors[:midpoint, :, :], descriptors[midpoint:, :, :]
                scores0, scores1 = scores[:midpoint, :], scores[midpoint:, :]
                images0, images1 = orig_warped[:midpoint, :, :, :], orig_warped[midpoint:, :, :, :]
                for k in range(midpoint):
                    ma_0, ma_1, miss_0, miss_1 = torch_find_matches(keypoints0[k], keypoints1[k], homographies[k], dist_thresh=3, n_iters=1)
                    all_match_index_0 = torch.cat([all_match_index_0, torch.empty(len(ma_0) + len(miss_0) + len(miss_1), dtype=torch.long, device=ma_0.device).fill_(k)])
                    all_match_index_1 = torch.cat([all_match_index_1, ma_0, miss_0, torch.empty(len(miss_1), dtype=torch.long, device=miss_1.device).fill_(-1)])
                    all_match_index_2 = torch.cat([all_match_index_2, ma_1, torch.empty(len(miss_0), dtype=torch.long, device=miss_0.device).fill_(-1), miss_1])
                match_indexes = torch.stack([all_match_index_0, all_match_index_1, all_match_index_2], -1)
                gt_vector = torch.ones(len(match_indexes), dtype=torch.float32, device=match_indexes.device)
            t3 = time_synchronized()
            gmodel_input = {
                'keypoints0': keypoints0, 'keypoints1': keypoints1,
                'descriptors0': descriptors0, 'descriptors1': descriptors1,
                'image0': images0, 'image1': images1,
                'scores0': scores0, 'scores1': scores1,
                'matches': match_indexes, 'gt_vec': gt_vector,
                'device': device
            }
            total_loss, pos_loss, neg_loss = gmodel(gmodel_input, **{'mode': 'train'})
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t4 = time_synchronized()
            if ema: ema.update(gmodel)
            data_time, preprocess_time, model_time = torch.tensor(t1 - t5, device=device), torch.tensor(t3-t2, device=device), torch.tensor(t4-t3, device=device)
            loss_items = torch.stack((pos_loss, neg_loss, total_loss, data_time, preprocess_time, model_time)).detach()
            if is_distributed: loss_items = reduce_tensor(loss_items)
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 7) % (str(epoch),mem, i, *mloss)
                pbar.set_description(s)
                if ((i+1) % config['train_params']['log_interval']) == 0:
                    write_str = "Epoch: {} Iter: {}, Loss: {}\n".format(epoch, i, mloss[0].item())
                    results_file.write(write_str)
                if config['train_params']['use_wandb']:
                    wandb.log({'total_loss': total_loss, 'pos_loss': pos_loss, 'neg_loss': neg_loss, 'mloss': mloss[2].item(), 'data_time': data_time.item(), 'preprocess_time': preprocess_time.item(), 'model_time': model_time.item()})
                ckpt = {'epoch': epoch,
                        'iter': i,
                        'ema': ema.ema.state_dict() if ema else None,
                        'ema_updates': ema.updates if ema else 0,
                        'model': gmodel.module.state_dict() if is_distributed else gmodel.state_dict(),
                        'optimizer': optimizer.state_dict()}
                if ((i+1) % 2000) == 0:
                    print("save lastiter {} with loss {}".format(epoch, best_min_loss))
                    torch.save(ckpt, weight_dir / 'lastiter.pt')
                if ((i+1) % 200)==0 and mloss[2].item() < best_min_loss:
                    best_min_loss = mloss[2].item()
                    print("save minloss {} with loss {}".format(epoch, best_min_loss))
                    torch.save(ckpt, weight_dir / 'minloss.pt')
                t5 = time_synchronized()
        if rank in [-1, 0]:
            print("\nDoing evaluation..")
            with torch.no_grad():
                eval_gmodel = ema.ema if ema else (gmodel.module if is_distributed else gmodel)
                results = test_model(val_dataloader, eval_gmodel, config['train_params']['val_images_count'], device, carhynet=carhynet)
            ckpt = {'epoch': epoch,
                    'iter': -1,
                    'ema': ema.ema.state_dict() if ema else None,
                    'ema_updates': ema.updates if ema else 0,
                    'model': gmodel.module.state_dict() if is_distributed else gmodel.state_dict(),
                    'optimizer': optimizer.state_dict(), 'metrics': results}
            torch.save(ckpt, weight_dir / 'last.pt')
            if results['weight_score'] > best_val_score:
                best_val_score = results['weight_score']
                print("Saving best model at epoch {} with score {}".format(epoch, best_val_score))
                torch.save(ckpt, weight_dir / 'best.pt')
        change_lr(epoch, config, optimizer)
    if rank > 0: dist.destroy_process_group()


def process_wrapper(rank, config, args, func):
    if args.nprocs > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(args.nprocs)
        os.environ['LOCAL_RANK'] = str(rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    if torch.cuda.device_count() >= 1:
        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(rank)
        print(f"GPU {rank}: {device_name}")
    else:
        device = torch.device('cpu')
        print(f"use CPU")

    func(config, args, rank if args.nprocs>1 else -1, device, args.nprocs>1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/coco_config.yaml")
    parser.add_argument("--backend", type=str, default='nccl')
    parser.add_argument("--gpus", type=str, default='2')
    parser.add_argument("--name", type=str, default='gims')
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    for i in range(torch.cuda.device_count()): print(i, "=>", torch.cuda.get_device_name(i))
    args.nprocs = len(args.gpus.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    with open(args.config_path, 'r', encoding='utf-8') as file: config = yaml.full_load(file)
    config["train_params"]['experiment_name'] = args.name
    config["train_params"]['experiment_tag'] = args.name
    config["train_params"]['save_dir'] = increment_path(Path(config['train_params']['output_dir']) / config['train_params']['experiment_name'])
    for i, k in config.items(): print(f">> {i}: {k}")
    if config['train_params']['use_wandb']: wandb.init(name=args.name, config=config, notes="train", project="gims")

    # train(config, args, -1, torch.device('cuda', int(args.gpus), False)
    torch.multiprocessing.spawn(process_wrapper, args=(config, args, train), nprocs=args.nprocs)

