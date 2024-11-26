import os
import sys
import time
import wandb
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassJaccardIndex)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_pretreatment.dataloader import GPCDataloader
from models.backbone import GPCNetSemanticSegmentor
from utils.misc import seed_torch, init_logger, print_num_params


def train_process():
    model.train()
    train_losses = []
    train_bar = tqdm(train_loader)
    for data_dict, label_dict in train_bar:
        p_seg_label = label_dict["point_seg_labels"]

        opt.zero_grad(set_to_none=True)

        p_seg_pred = model(data_dict)

        loss_seg = seg_loss(p_seg_pred, p_seg_label)
        loss = wandb.config['seg_a'] * loss_seg
        train_losses.append(loss.item())

        lr = opt.param_groups[0]["lr"]
        info = "Epoch:%d LR:%f Seg:%f Total:%f" % (epoch, lr, loss_seg, loss)
        train_bar.set_description(info)

        loss.backward()
        opt.step()

        # Update the moving average with the new parameters from the last optimizer step
        ema.update()

        train_seg_acc.update(p_seg_pred, p_seg_label)
        train_seg_iou.update(p_seg_pred, p_seg_label)

    scheduler.step()
    # batch end
    mean_train_loss = np.mean(train_losses).item()
    mean_train_seg_acc = train_seg_acc.compute().item()
    mean_train_seg_iou = train_seg_iou.compute().item()

    logger.info(f'train_loss : {mean_train_loss}, \
                  train_seg_acc: {mean_train_seg_acc}, \
                  train_seg_iou: {mean_train_seg_iou}')
    wandb.log({'epoch': epoch,
               'train_loss': mean_train_loss,
               'train_seg_acc': mean_train_seg_acc,
               'train_seg_iou': mean_train_seg_iou
               })

    train_seg_acc.reset()
    train_seg_iou.reset()


def eval_process():
    with torch.no_grad():
        with ema.average_parameters():
            model.eval()
            val_losses = []
            for data_dict, label_dict in tqdm(val_loader):
                seg_label = label_dict["segmentation_labels"]

                seg_pred = model(data_dict)

                loss_seg = seg_loss(seg_pred, seg_label)
                loss = wandb.config['seg_a'] * loss_seg
                val_losses.append(loss.item())

                val_seg_acc.update(seg_pred, seg_label)
                val_seg_iou.update(seg_pred, seg_label)
            # val end
            mean_val_loss = np.mean(val_losses).item()
            mean_val_seg_acc = val_seg_acc.compute().item()
            mean_val_seg_iou = val_seg_iou.compute().item()

            logger.info(f'val_loss : {mean_val_loss}, \
                          val_seg_acc: {mean_val_seg_acc}, \
                          val_seg_iou: {mean_val_seg_iou}')
            wandb.log({'epoch': epoch,
                       'val_loss': mean_val_loss,
                       'val_seg_acc': mean_val_seg_acc,
                       'val_seg_iou': mean_val_seg_iou
                       })

            val_seg_acc.reset()
            val_seg_iou.reset()

            cur_acc = mean_val_seg_iou

    return cur_acc


def test_process():
    with torch.no_grad():
        logger.info(f'------------- Now start testing ------------- ')
        model.eval()
        test_losses = []
        for data_dict, label_dict in tqdm(test_loader):
            seg_label = label_dict["segmentation_labels"]

            # Forward pass
            seg_pred = model(data_dict)

            loss_seg = seg_loss(seg_pred, seg_label)
            loss = wandb.config['seg_a'] * loss_seg
            test_losses.append(loss.item())

            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)

        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()

        logger.info(f'test_loss : {mean_test_loss}, \
                      test_seg_acc: {mean_test_seg_acc}, \
                      test_seg_iou: {mean_test_seg_iou}')
        wandb.log({'test_loss': mean_test_loss,
                   'test_seg_acc': mean_test_seg_acc,
                   'test_seg_iou': mean_test_seg_iou
                   })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--batch_path", type=str, required=True, help="Path to load the batched data from")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    os.environ["WANDB_API_KEY"] = '##################'
    os.environ["WANDB_MODE"] = "offline"

    # Start a new wandb run to track this script
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    wandb.init(
        # Set the wandb project where this run will be logged
        project="g&pcnet" + args.dataset_name,

        # track hyperparameters and run metadata
        config={
            "node_attr_dim": 10,
            "node_grid_dim": 7,
            "edge_attr_dim": 12,
            "edge_grid_dim": 0,
            "graph_out_channels": 64,
            "graph_num_layers": 9,
            "point_attr_dim": 3,
            "point_out_channels": 128,
            "num_classes": 16 if args.dataset_name == 'MFCAD' else 25,

            "seed": 42,
            "device": 'cuda',
            "architecture": "G&PCNet",
            "dataset": args.batch_path,

            "epochs": 350 if args.dataset_name == 'MFCAD' else 100,
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "ema_decay_per_epoch": 1. / 2.,
            "seg_a": 1.,
        }
    )

    print(wandb.config)
    seed_torch(wandb.config['seed'])
    device = wandb.config['device']
    dataset = wandb.config['dataset']
    n_classes = wandb.config['num_classes']

    model = GPCNetSemanticSegmentor(
        node_attr_dim=wandb.config['node_attr_dim'],
        node_grid_dim=wandb.config['node_grid_dim'],
        edge_attr_dim=wandb.config['edge_attr_dim'],
        edge_grid_dim=wandb.config['edge_grid_dim'],
        graph_out_channels=wandb.config['graph_out_channels'],
        graph_num_layers=wandb.config['graph_num_layers'],
        point_attr_dim=wandb.config['point_attr_dim'],
        point_out_channels=wandb.config['point_out_channels'],
        num_classes=wandb.config['num_classes']
    )
    model = model.to(device)

    total_params = print_num_params(model)
    wandb.config['total_params'] = total_params

    train_dataset = GPCDataloader(root_dir=dataset, dataset_type='train')

    val_dataset = GPCDataloader(root_dir=dataset, dataset_type='val')

    seg_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'], weight_decay=wandb.config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=wandb.config['epochs'], eta_min=0)

    train_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)

    train_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    val_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)

    val_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    iters = train_dataset.num_batches()
    ema_decay = wandb.config['ema_decay_per_epoch'] ** (1 / iters)
    print(f'EMA decay: {ema_decay}')
    ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    best_acc = 0.
    best_epoch = 0
    save_path = args.dataset_name + '_output'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, time_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = init_logger(os.path.join(save_path, 'log.txt'))
    for epoch in range(wandb.config['epochs']):
        logger.info(f'------------- Now start epoch {epoch}------------- ')
        train_loader = train_dataset.gpc_dataloader_semantic()
        val_loader = val_dataset.gpc_dataloader_semantic()
        train_process()
        current_acc = eval_process()

        if current_acc > best_acc:
            best_acc = current_acc
            best_epoch = epoch
            logger.info(f'best metric: {current_acc}, model saved')
            torch.save(model.state_dict(), os.path.join(save_path, "weight_%d-epoch.pth" % epoch))
        # epoch end

    # Load the best model saved
    model.load_state_dict(torch.load(os.path.join(save_path, "weight_%d-epoch.pth" % best_epoch)))
    model.eval()

    # training end test
    test_dataset = GPCDataloader(root_dir=dataset, dataset_type='test')

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)

    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)

    test_loader = test_dataset.gpc_dataloader_semantic()
    test_process()
