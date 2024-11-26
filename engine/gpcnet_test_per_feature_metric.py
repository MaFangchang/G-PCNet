import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data_pretreatment.dataloader import GPCDataloader
from models.backbone import GPCNetSegmentor
from utils.misc import seed_torch

EPS = 1e-6
INST_THRES = 0.5
BOTTOM_THRES = 0.5

feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
              'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
              'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
              'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
              'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock']


def print_class_metric(metric):
    string = ''
    for i in range(len(metric)):
        string += feat_names[i] + ': ' + str(metric[i]) + ', '
    print(string)


class FeatureInstance:
    def __init__(self, name: int = None,
                 faces: np.array = None,
                 bottoms: list = None):
        self.name = name
        self.faces = faces
        self.bottoms = bottoms


def parser_label(inst_label, seg_label, bottom_label, nodes_per_graph):
    label_list = []
    # parse instance label
    for idx, inst in enumerate(inst_label):
        seg = seg_label[nodes_per_graph == idx]
        inst = np.array(inst, dtype=np.uint8)
        used_faces = []
        for row_idx, row in enumerate(inst):
            if np.sum(row) == 0:
                # stock face, no linked face, so the sum of the column is 0
                continue
            # when I_ij = 1 mean face_i is linked with face_j
            # so can get the indices of linked faces in an instance
            linked_face_idx = np.where(row == 1)[0]
            # used
            if len(set(linked_face_idx).intersection(set(used_faces))) > 0:
                # the face has been assigned to an instance
                continue
            # create new feature
            new_feat = FeatureInstance()
            new_feat.faces = linked_face_idx
            used_faces.extend(linked_face_idx)
            # all the linked faces in an instance
            # have the same segmentation label
            # so get the name of the instance
            a_face_id = new_feat.faces[0]
            seg_id = seg[int(a_face_id)]
            new_feat.name = seg_id
            # get the bottom face segmentation label
            # new_feat.bottoms = np.where(bottom_label==1)[0]
            # # add new feature into list and used face counter
            label_list.append(new_feat)

    return label_list


def post_process(out, nodes_per_graph, inst_thres, bottom_thres):
    seg_out, inst_out, bottom_out = out
    # post-processing for semantic segmentation 
    # face_logits = torch.argmax(seg_out, dim=1)
    face_logits = seg_out.cpu().numpy()

    features_list = []
    for idx, inst in enumerate(inst_out):
        face_logit = face_logits[nodes_per_graph == idx]
        # post-processing for instance segmentation
        inst = inst.sigmoid()
        adj = inst > inst_thres
        adj = adj.cpu().numpy().astype('int32')

        # post-processing for bottom classification
        # bottom_out = bottom_out.sigmoid()
        # bottom_logits = bottom_out > bottom_thres
        # bottom_logits = bottom_logits.cpu().numpy()

        # Identify individual proposals of each feature
        proposals = set()  # use to delete repeat proposals
        # record whether the face belongs to an instance
        used_flags = np.zeros(adj.shape[0], dtype=np.bool_)
        for row_idx, row in enumerate(adj):
            if used_flags[row_idx]:
                # the face has been assigned to an instance
                continue
            if np.sum(row) <= EPS:
                # stock face, no linked face, so the sum of the column is 0
                continue
            # non-stock face
            proposal = set()  # use to delete repeat faces
            for col_idx, item in enumerate(row):
                if used_flags[col_idx]:
                    # the face has been assigned to a proposal
                    continue
                if item:  # have connections with current face
                    proposal.add(col_idx)
                    used_flags[col_idx] = True
            if len(proposal) > 0:
                proposals.add(frozenset(proposal))  # frozenset is a hashable set
        # TODO: better post-process

        # save new results
        for instance in proposals:
            instance = list(instance)
            # sum voting for the class of the instance
            sum_inst_logit = 0
            for face in instance:
                sum_inst_logit += face_logit[face]
            # the index of max score is the class of the instance
            inst_logit = np.argmax(sum_inst_logit)
            if inst_logit == 24:
                # is stock, ignore
                continue
            # get instance label name from face_logits

            inst_name = inst_logit
            # get the bottom faces
            # bottom_faces = []
            # for face_idx in instance:
            #     if bottom_logits[face_idx]:
            #         bottom_faces.append(face_idx)
            features_list.append(
                FeatureInstance(name=inst_name, faces=np.array(instance)))

    return features_list


def cal_recognition_performance(feature_list, label_list):
    # one hot encoding
    pre = np.zeros(24, dtype=int)
    g_t = np.zeros(24, dtype=int)
    for feature in feature_list:
        pre[feature.name] += 1
    for label in label_list:
        g_t[label.name] += 1
    t_p = np.minimum(g_t, pre)

    return pre, g_t, t_p


def cal_localization_performance(feature_list, label_list):
    # one hot encoding
    pre = np.zeros(24, dtype=int)
    g_t = np.zeros(24, dtype=int)
    for feature in feature_list:
        pre[feature.name] += 1
    for label in label_list:
        g_t[label.name] += 1

    # sort the feature_list and label_list by name
    feature_list.sort(key=lambda x: x.name)
    label_list.sort(key=lambda x: x.name)
    t_p = np.zeros(24, dtype=int)

    found_lbl = np.zeros(len(label_list))
    # for each detection
    for pred_i in range(len(feature_list)):
        pred_name = feature_list[pred_i].name

        # among the ground-truths, choose one that belongs to the same class and has the highest IoU with the detection
        for lbl_i in range(len(label_list)):
            lbl_name = label_list[lbl_i].name

            if pred_name != lbl_name or found_lbl[lbl_i] == 1:
                continue

            # compute IoU
            pred_faces = feature_list[pred_i].faces
            lbl_faces = label_list[lbl_i].faces
            intersection = np.intersect1d(pred_faces, lbl_faces)
            union = np.union1d(pred_faces, lbl_faces)
            iou = len(intersection) / len(union)

            # when IOU == 1, the detection is correct
            # else the detection is wrong
            if iou >= 1.0 - EPS:
                found_lbl[lbl_i] = 1
                t_p[pred_name] += 1
                break

    # when tp gt not equal, print the detail
    # if not np.all(tp == gt):
    #     for feature in feature_list:
    #         feature.faces.sort()
    #         print('feature', feature.name, feature.faces)
    #     for label in label_list:
    #         label.faces.sort()
    #         print('label', label.name, label.faces)

    #     print('tp', tp)
    #     print('pd', pred)
    #     print('gt', gt)

    return pre, g_t, t_p


def eval_metric(pre, t_l, t_p):
    precision = t_p / (pre + EPS)
    recall = t_p / (t_l + EPS)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    # if the gt[i] == 0, mean class i is not in the ground truth
    # so the precision and recall of class i is not defined
    # so set the precision and recall of class i to 1
    precision[t_l == 0] = 1
    recall[t_l == 0] = 1

    return precision, recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--batch_path", type=str, required=True, help="Path to load the batched data from")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    os.environ["WANDB_API_KEY"] = '##################'
    os.environ["WANDB_MODE"] = "offline"

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

    model_param = torch.load(f"./{args.dataset_name}_output/weight_best_on_{args.dataset_name}.pth",
                             map_location=device)
    model.load_state_dict(model_param)

    test_dataset = GPCDataloader(root_dir=dataset, dataset_type='test')
    test_loader = test_dataset.gpc_dataloader()

    rec_predictions = np.zeros(24, dtype=int)
    rec_true_labels = np.zeros(24, dtype=int)
    rec_true_positives = np.zeros(24, dtype=int)

    loc_predictions = np.zeros(24, dtype=int)
    loc_true_labels = np.zeros(24, dtype=int)
    loc_true_positives = np.zeros(24, dtype=int)

    time_accumulator = 0
    with torch.no_grad():
        print(f'------------- Now start testing ------------- ')
        model.eval()
        test_losses = []
        for data_dict, label_dict in tqdm(test_loader):
            seg_labels = label_dict["segmentation_labels"].cpu().numpy()
            inst_labels = label_dict["instance_labels"].cpu().numpy()
            bottom_labels = label_dict["bottom_labels"].cpu().numpy()
            graph_node_map = data_dict["graph_node_map"].cpu().numpy()

            # Forward pass
            start_time = time.time()
            outs = model(data_dict)
            features = post_process(outs, graph_node_map, inst_thres=INST_THRES, bottom_thres=BOTTOM_THRES)
            time_accumulator += time.time() - start_time

            # calculate recognition performance
            labels = parser_label(inst_labels, seg_labels, bottom_labels, graph_node_map)
            pred, gt, tp = cal_recognition_performance(features, labels)
            rec_predictions += pred
            rec_true_labels += gt
            rec_true_positives += tp

            # calculate localization performance
            pred, gt, tp = cal_localization_performance(features, labels)
            loc_predictions += pred
            loc_true_labels += gt
            loc_true_positives += tp

        print('------------- recognition performance------------- ')
        print('rec_pred', rec_predictions)
        print('rec_true', rec_true_labels)
        print('rec_trpo', rec_true_positives)
        precision, recall = eval_metric(rec_predictions, rec_true_labels, rec_true_positives)
        print('recognition Precision scores')
        # print precision for each class
        print_class_metric(precision)
        precision = precision.mean()
        print('AVG recognition Precision:', precision)
        print('recognition Recall scores')
        # print recall for each class
        print_class_metric(recall)
        recall = recall.mean()
        print('AVG recognition Precision:', recall)
        print('recognition F scores')
        rec_F = (2 * recall * precision) / (recall + precision)
        print(rec_F)

        print('------------- localization performance------------- ')
        print('loc_pred', loc_predictions)
        print('loc_true', loc_true_labels)
        print('loc_trpo', loc_true_positives)
        precision, recall = eval_metric(loc_predictions, loc_true_labels, loc_true_positives)
        print('localization Precision scores')
        # print precision for each class
        print_class_metric(precision)
        precision = precision.mean()
        print('AVG localization Precision:', precision)
        print('localization Recall scores')
        # print recall for each class
        print_class_metric(recall)
        recall = recall.mean()
        print('AVG localization Precision:', recall)
        print('localization F scores')
        loc_F = (2 * recall * precision) / (recall + precision)
        print(loc_F)

        print('------------- average time cost per STEP------------- ')
        iters = test_dataset.num_batches()
        print(time_accumulator / iters)
        print('------------- Final ------------- ')
        print('rec F scores(%):', rec_F * 100)
        print('loc F scores(%):', loc_F * 100)
