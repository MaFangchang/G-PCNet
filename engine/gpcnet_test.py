import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    MulticlassJaccardIndex)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_pretreatment.dataloader import GPCDataloader
from models.backbone import GPCNetSegmentor
from utils.misc import seed_torch, print_num_params


def test_process():
    with torch.no_grad():
        print(f'------------- Now start testing ------------- ')
        model.eval()
        test_losses = []
        for data_dict, label_dict in tqdm(test_loader):
            seg_label = label_dict["segmentation_labels"]
            inst_label = label_dict["instance_labels"]
            bottom_label = label_dict["bottom_labels"]

            # Forward pass
            seg_pred, inst_pred, bottom_pred = model(data_dict)

            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = instance_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)
            loss = (config['seg_a'] * loss_seg +
                    config['inst_a'] * loss_inst +
                    config['bottom_a'] * loss_bottom)
            test_losses.append(loss.item())

            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            test_inst_acc.update(inst_pred, inst_label)
            test_inst_f1.update(inst_pred, inst_label)
            test_bottom_acc.update(bottom_pred, bottom_label)
            test_bottom_iou.update(bottom_pred, bottom_label)

        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        mean_test_inst_acc = test_inst_acc.compute().item()
        mean_test_inst_f1 = test_inst_f1.compute().item()
        mean_test_bottom_acc = test_bottom_acc.compute().item()
        mean_test_bottom_iou = test_bottom_iou.compute().item()

        print(f'test_loss : {mean_test_loss}, \
                      test_seg_acc: {mean_test_seg_acc}, \
                      test_seg_iou: {mean_test_seg_iou}, \
                      test_inst_acc: {mean_test_inst_acc}, \
                      test_inst_f1: {mean_test_inst_f1}, \
                      test_bottom_acc: {mean_test_bottom_acc}, \
                      test_bottom_iou: {mean_test_bottom_iou}')


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
    config = {
        "node_attr_dim": 10,
        "node_grid_dim": 7,
        "edge_attr_dim": 12,
        "edge_grid_dim": 0,
        "graph_out_channels": 64,
        "graph_num_layers": 9,
        "point_attr_dim": 3,
        "point_out_channels": 128,
        "num_classes": 25,

        "seed": 42,
        "device": 'cuda',
        "architecture": "G&PCNet",
        "dataset": args.batch_path,

        "epochs": 100,
        "seg_a": 1.,
        "inst_a": 1.,
        "bottom_a": 1.,
    }

    print(config)
    seed_torch(config['seed'])
    device = config['device']
    dataset = config['dataset']
    n_classes = config['num_classes']

    model = GPCNetSegmentor(
        node_attr_dim=config['node_attr_dim'],
        node_grid_dim=config['node_grid_dim'],
        edge_attr_dim=config['edge_attr_dim'],
        edge_grid_dim=config['edge_grid_dim'],
        graph_out_channels=config['graph_out_channels'],
        graph_num_layers=config['graph_num_layers'],
        point_attr_dim=config['point_attr_dim'],
        point_out_channels=config['point_out_channels'],
        num_classes=config['num_classes']
    )
    model = model.to(device)

    model_param = torch.load(f"./{args.dataset_name}_output/weight_best_on_{args.dataset_name}.pth", map_location=device)
    model.load_state_dict(model_param)

    total_params = print_num_params(model)
    config['total_params'] = total_params
    test_dataset = GPCDataloader(root_dir=dataset, dataset_type='test')

    seg_loss = nn.CrossEntropyLoss()
    instance_loss = nn.BCEWithLogitsLoss()
    bottom_loss = nn.BCEWithLogitsLoss()

    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_inst_acc = BinaryAccuracy().to(device)
    test_bottom_acc = BinaryAccuracy().to(device)

    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    test_inst_f1 = BinaryF1Score().to(device)
    test_bottom_iou = BinaryJaccardIndex().to(device)

    test_loader = test_dataset.gpc_dataloader()
    test_process()
