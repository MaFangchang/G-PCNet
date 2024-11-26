import gc
import os
import os.path as osp
import argparse
import h5py
import random
from pathlib import Path
from tqdm import tqdm

from src.batch_hander import *


class GenerateBatch:
    def __init__(self, sub_data, file_path, batch_path, graphs_per_batch):
        self.sub_data = sub_data
        self.file_path = file_path
        self.batch_path = batch_path
        self.graphs_per_batch = graphs_per_batch
        self.normalize = True

    def process(self):
        with h5py.File(self.file_path, 'r') as hf:
            batch_counter = 0
            batch_num = 0
            graph_counter = 0
            vertex_counter = 0
            point_counter = 0
            offset = []
            instance_labels_list = []

            graph_face_attr = graph_face_grid = graph_edge_attr = graph_edge_grid = None
            edges = points = normals = sequence = None
            names = segmentation_labels = bottom_labels = None

            keys = list(hf.keys())
            random.shuffle(keys)

            # Loop over groups in h5 file
            for key in tqdm(keys):
                group = hf[key]

                # Check if adding the new graph will make the batch graph too. If so add the batch graph to the file.
                if graph_counter == self.graphs_per_batch:
                    face_point_map = point_batches_map(sequence)
                    graph_point_map = point_batches_map(np.array(offset))
                    point_seg_labels = point_labels_map(segmentation_labels, sequence)
                    point_inst_labels, instance_labels = expand_and_pad_matrices(instance_labels_list, sequence, num_nodes)
                    point_bottom_labels = point_labels_map(bottom_labels, sequence)
                    graph_node_map = np.repeat(np.arange(len(num_nodes)), num_nodes, axis=0)
                    batch = {
                        "names": names,
                        "graph_face_attr": graph_face_attr,
                        "graph_face_grid": graph_face_grid,
                        "graph_edge_attr": graph_edge_attr,
                        "graph_edge_grid": graph_edge_grid,
                        "edges": edges,
                        "points": points,
                        "normals": normals,
                        "face_point_map": face_point_map,
                        "graph_point_map": graph_point_map,
                        "segmentation_labels": segmentation_labels,
                        "instance_labels": instance_labels,
                        "bottom_labels": bottom_labels,
                        "point_seg_labels": point_seg_labels,
                        # "point_inst_labels": point_inst_labels,
                        "point_bottom_labels": point_bottom_labels,
                        "graph_node_map": graph_node_map
                    }
                    self.write_batch_to_file(batch_num, batch)

                    # Reset batch
                    batch_counter = 0
                    graph_counter = 0
                    vertex_counter = 0
                    point_counter = 0
                    offset = []
                    instance_labels_list = []
                    batch_num += 1

                    graph_face_attr = graph_face_grid = graph_edge_attr = graph_edge_grid = None
                    edges = points = normals = sequence = None
                    names = segmentation_labels = bottom_labels = None

                # If limit is not reached add new graph to batch.
                else:
                    group_data = self.extract_data_from_h5_group(group)
                    # If there is no graphs in current batch
                    if batch_counter == 0:
                        edges = group_data["edges"]
                        num_nodes = group_data["num_nodes"]
                        graph_face_attr = group_data["graph_face_attr"]
                        graph_face_grid = group_data["graph_face_grid"]
                        graph_edge_attr = group_data["graph_edge_attr"]
                        graph_edge_grid = group_data["graph_edge_grid"]
                        points = group_data["points"]
                        normals = group_data["normals"]
                        sequence = group_data["sequence"]
                        segmentation_labels = group_data["segmentation_labels"]
                        instance_labels = group_data["instance_labels"]
                        bottom_labels = group_data["bottom_labels"]
                        names = np.array([[key]], dtype='S')

                        batch_counter += 1
                        point_counter = points.shape[0]
                        offset.append(point_counter)
                        instance_labels_list.append(instance_labels)

                    # If there are graphs in current batch
                    else:
                        graph_face_attr = np.vstack((graph_face_attr, group_data["graph_face_attr"]))
                        graph_face_grid = np.concatenate((graph_face_grid, group_data["graph_face_grid"]), axis=0)
                        graph_edge_attr = np.vstack((graph_edge_attr, group_data["graph_edge_attr"]))
                        graph_edge_grid = np.concatenate((graph_edge_grid, group_data["graph_edge_grid"]), axis=0)
                        edges = np.hstack((edges, group_data["edges"] + vertex_counter))
                        num_nodes = np.hstack((num_nodes, group_data["num_nodes"]))

                        points = np.vstack((points, group_data["points"]))
                        normals = np.vstack((normals, group_data["normals"]))
                        sequence = np.hstack((sequence, group_data["sequence"] + point_counter))
                        point_counter = points.shape[0]
                        offset.append(point_counter)

                        names = np.vstack((names, np.array([[key]], dtype='S')))
                        segmentation_labels = np.hstack((segmentation_labels, group_data["segmentation_labels"]))
                        instance_labels_list.append(group_data["instance_labels"])
                        bottom_labels = np.vstack((bottom_labels, group_data["bottom_labels"]))

                        batch_counter += 1
                    vertex_counter += group_data["num_nodes"]
                graph_counter += 1

            if names is not None:
                face_point_map = point_batches_map(sequence)
                graph_point_map = point_batches_map(np.array(offset))
                point_seg_labels = point_labels_map(segmentation_labels, sequence)
                point_inst_labels, instance_labels = expand_and_pad_matrices(instance_labels_list, sequence, num_nodes)
                point_bottom_labels = point_labels_map(bottom_labels, sequence)
                graph_node_map = np.repeat(np.arange(len(num_nodes)), num_nodes, axis=0)
                batch = {
                    "names": names,
                    "graph_face_attr": graph_face_attr,
                    "graph_face_grid": graph_face_grid,
                    "graph_edge_attr": graph_edge_attr,
                    "graph_edge_grid": graph_edge_grid,
                    "edges": edges,
                    "points": points,
                    "normals": normals,
                    "face_point_map": face_point_map,
                    "graph_point_map": graph_point_map,
                    "segmentation_labels": segmentation_labels,
                    "instance_labels": instance_labels,
                    "bottom_labels": bottom_labels,
                    "point_seg_labels": point_seg_labels,
                    # "point_inst_labels": point_inst_labels,
                    "point_bottom_labels": point_bottom_labels,
                    "graph_node_map": graph_node_map
                }
                self.write_batch_to_file(batch_num, batch)

    def extract_data_from_h5_group(self, h5_group):
        """Extracting data from a group in a h5 file"""

        group_data = {}
        edges = np.array(h5_group["edges"])
        num_nodes = np.array(h5_group["num_nodes"])
        graph_face_attr = np.array(h5_group["graph_face_attr"])
        if num_nodes.size == 0:
            num_nodes = np.array([graph_face_attr.shape[0]])

        graph_face_grid = np.array(h5_group["graph_face_grid"])
        graph_edge_attr = np.array(h5_group["graph_edge_attr"])
        graph_edge_grid = np.array(h5_group["graph_edge_grid"])
        points = np.array(h5_group["points"])
        normals = np.array(h5_group["normals"])
        sequence = np.array(h5_group["sequence"])
        segmentation_labels = np.array(h5_group["segmentation_labels"])
        instance_labels = np.array(h5_group["instance_labels"])
        bottom_labels = np.array(h5_group["bottom_labels"])
        bottom_labels = bottom_labels[:, np.newaxis]

        if self.normalize:
            graph_face_attr[:, 5] = normalize_data(graph_face_attr[:, 5])
            graph_edge_attr[:, 3] = normalize_data(graph_edge_attr[:, 3])

        group_data["edges"] = edges
        group_data["num_nodes"] = num_nodes
        group_data["graph_face_attr"] = graph_face_attr
        group_data["graph_face_grid"] = graph_face_grid
        group_data["graph_edge_attr"] = graph_edge_attr
        group_data["graph_edge_grid"] = graph_edge_grid
        group_data["points"] = points
        group_data["normals"] = normals
        group_data["sequence"] = sequence
        group_data["segmentation_labels"] = segmentation_labels
        group_data["instance_labels"] = instance_labels
        group_data["bottom_labels"] = bottom_labels

        return group_data

    def write_batch_to_file(self, batch_num, batch):
        """Writes batch graph to h5 file.

        param batch_num: Index of batch.
        param batch: List containing batch graph information.
        param file_path: File path of h5 file.
        return: None
        """

        with h5py.File(self.batch_path, 'a') as hf:
            batch_group = hf.create_group(str(batch_num))

            batch_group.create_dataset("names", data=batch["names"], dtype='|S21', compression="lzf")
            batch_group.create_dataset("graph_face_attr", data=batch["graph_face_attr"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_face_grid", data=batch["graph_face_grid"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_edge_attr", data=batch["graph_edge_attr"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_edge_grid", data=batch["graph_edge_grid"], dtype='float64', compression="lzf")
            batch_group.create_dataset("edges", data=batch["edges"], dtype='int32', compression="lzf")
            batch_group.create_dataset("points", data=batch["points"], dtype='float64', compression="lzf")
            batch_group.create_dataset("normals", data=batch["normals"], dtype='float64', compression="lzf")
            batch_group.create_dataset("face_point_map", data=batch["face_point_map"], dtype='int32', compression="lzf")
            batch_group.create_dataset("graph_point_map", data=batch["graph_point_map"], dtype='int32', compression="lzf")
            batch_group.create_dataset("segmentation_labels", data=batch["segmentation_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("instance_labels", data=batch["instance_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("bottom_labels", data=batch["bottom_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("point_seg_labels", data=batch["point_seg_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("point_bottom_labels", data=batch["point_bottom_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_node_map", data=batch["graph_node_map"], dtype='int32')


class GenerateSemanticBatch:
    def __init__(self, sub_data, file_path, batch_path, graphs_per_batch):
        self.sub_data = sub_data
        self.file_path = file_path
        self.batch_path = batch_path
        self.graphs_per_batch = graphs_per_batch
        self.normalize = True

    def process(self):
        with h5py.File(self.file_path, 'r') as hf:
            batch_counter = 0
            batch_num = 0
            graph_counter = 0
            vertex_counter = 0
            point_counter = 0
            offset = []

            graph_face_attr = graph_face_grid = graph_edge_attr = graph_edge_grid = None
            edges = points = normals = sequence = None
            names = segmentation_labels = None

            keys = list(hf.keys())
            random.shuffle(keys)

            # Loop over groups in h5 file
            for key in tqdm(keys):
                group = hf[key]

                # Check if adding the new graph will make the batch graph too. If so add the batch graph to the file.
                if graph_counter == self.graphs_per_batch:
                    face_point_map = point_batches_map(sequence)
                    graph_point_map = point_batches_map(np.array(offset))
                    point_seg_labels = point_labels_map(segmentation_labels, sequence)
                    graph_node_map = np.repeat(np.arange(len(num_nodes)), num_nodes, axis=0)
                    batch = {
                        "names": names,
                        "graph_face_attr": graph_face_attr,
                        "graph_face_grid": graph_face_grid,
                        "graph_edge_attr": graph_edge_attr,
                        "graph_edge_grid": graph_edge_grid,
                        "edges": edges,
                        "points": points,
                        "normals": normals,
                        "face_point_map": face_point_map,
                        "graph_point_map": graph_point_map,
                        "segmentation_labels": segmentation_labels,
                        "point_seg_labels": point_seg_labels,
                        "graph_node_map": graph_node_map
                    }
                    self.write_batch_to_file(batch_num, batch)

                    # Reset batch
                    batch_counter = 0
                    graph_counter = 0
                    vertex_counter = 0
                    point_counter = 0
                    offset = []
                    batch_num += 1

                    graph_face_attr = graph_face_grid = graph_edge_attr = graph_edge_grid = None
                    edges = points = normals = sequence = None
                    names = segmentation_labels = None

                # If limit is not reached add new graph to batch.
                else:
                    group_data = self.extract_data_from_h5_group(group)
                    # If there is no graphs in current batch
                    if batch_counter == 0:
                        edges = group_data["edges"]
                        num_nodes = group_data["num_nodes"]
                        graph_face_attr = group_data["graph_face_attr"]
                        graph_face_grid = group_data["graph_face_grid"]
                        graph_edge_attr = group_data["graph_edge_attr"]
                        graph_edge_grid = group_data["graph_edge_grid"]
                        points = group_data["points"]
                        normals = group_data["normals"]
                        sequence = group_data["sequence"]
                        segmentation_labels = group_data["segmentation_labels"]
                        names = np.array([[key]], dtype='S')

                        batch_counter += 1
                        point_counter = points.shape[0]
                        offset.append(point_counter)

                    # If there are graphs in current batch
                    else:
                        graph_face_attr = np.vstack((graph_face_attr, group_data["graph_face_attr"]))
                        graph_face_grid = np.concatenate((graph_face_grid, group_data["graph_face_grid"]), axis=0)
                        graph_edge_attr = np.vstack((graph_edge_attr, group_data["graph_edge_attr"]))
                        graph_edge_grid = np.concatenate((graph_edge_grid, group_data["graph_edge_grid"]), axis=0)
                        edges = np.hstack((edges, group_data["edges"] + vertex_counter))
                        num_nodes = np.hstack((num_nodes, group_data["num_nodes"]))

                        points = np.vstack((points, group_data["points"]))
                        normals = np.vstack((normals, group_data["normals"]))
                        sequence = np.hstack((sequence, group_data["sequence"] + point_counter))
                        point_counter = points.shape[0]
                        offset.append(point_counter)

                        names = np.vstack((names, np.array([[key]], dtype='S')))
                        segmentation_labels = np.hstack((segmentation_labels, group_data["segmentation_labels"]))

                        batch_counter += 1
                    vertex_counter += group_data["num_nodes"]
                graph_counter += 1

            if names is not None:
                face_point_map = point_batches_map(sequence)
                graph_point_map = point_batches_map(np.array(offset))
                point_seg_labels = point_labels_map(segmentation_labels, sequence)
                graph_node_map = np.repeat(np.arange(len(num_nodes)), num_nodes, axis=0)
                batch = {
                    "names": names,
                    "graph_face_attr": graph_face_attr,
                    "graph_face_grid": graph_face_grid,
                    "graph_edge_attr": graph_edge_attr,
                    "graph_edge_grid": graph_edge_grid,
                    "edges": edges,
                    "points": points,
                    "normals": normals,
                    "face_point_map": face_point_map,
                    "graph_point_map": graph_point_map,
                    "segmentation_labels": segmentation_labels,
                    "point_seg_labels": point_seg_labels,
                    "graph_node_map": graph_node_map
                }
                self.write_batch_to_file(batch_num, batch)

    def extract_data_from_h5_group(self, h5_group):
        """Extracting data from a group in a h5 file"""

        group_data = {}
        edges = np.array(h5_group["edges"])
        num_nodes = np.array(h5_group["num_nodes"])
        graph_face_attr = np.array(h5_group["graph_face_attr"])
        if num_nodes.size == 0:
            num_nodes = np.array([graph_face_attr.shape[0]])

        graph_face_grid = np.array(h5_group["graph_face_grid"])
        graph_edge_attr = np.array(h5_group["graph_edge_attr"])
        graph_edge_grid = np.array(h5_group["graph_edge_grid"])
        points = np.array(h5_group["points"])
        normals = np.array(h5_group["normals"])
        sequence = np.array(h5_group["sequence"])
        segmentation_labels = np.array(h5_group["segmentation_labels"])

        if self.normalize:
            graph_face_attr[:, 5] = normalize_data(graph_face_attr[:, 5])
            graph_edge_attr[:, 3] = normalize_data(graph_edge_attr[:, 3])

        group_data["edges"] = edges
        group_data["num_nodes"] = num_nodes
        group_data["graph_face_attr"] = graph_face_attr
        group_data["graph_face_grid"] = graph_face_grid
        group_data["graph_edge_attr"] = graph_edge_attr
        group_data["graph_edge_grid"] = graph_edge_grid
        group_data["points"] = points
        group_data["normals"] = normals
        group_data["sequence"] = sequence
        group_data["segmentation_labels"] = segmentation_labels

        return group_data

    def write_batch_to_file(self, batch_num, batch):
        """Writes batch graph to h5 file.

        param batch_num: Index of batch.
        param batch: List containing batch graph information.
        param file_path: File path of h5 file.
        return: None
        """

        with h5py.File(self.batch_path, 'a') as hf:
            batch_group = hf.create_group(str(batch_num))

            batch_group.create_dataset("names", data=batch["names"], dtype='|S21', compression="lzf")
            batch_group.create_dataset("graph_face_attr", data=batch["graph_face_attr"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_face_grid", data=batch["graph_face_grid"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_edge_attr", data=batch["graph_edge_attr"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_edge_grid", data=batch["graph_edge_grid"], dtype='float64', compression="lzf")
            batch_group.create_dataset("edges", data=batch["edges"], dtype='int32', compression="lzf")
            batch_group.create_dataset("points", data=batch["points"], dtype='float64', compression="lzf")
            batch_group.create_dataset("normals", data=batch["normals"], dtype='float64', compression="lzf")
            batch_group.create_dataset("face_point_map", data=batch["face_point_map"], dtype='int32', compression="lzf")
            batch_group.create_dataset("graph_point_map", data=batch["graph_point_map"], dtype='int32', compression="lzf")
            batch_group.create_dataset("segmentation_labels", data=batch["segmentation_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("point_seg_labels", data=batch["point_seg_labels"], dtype='float64', compression="lzf")
            batch_group.create_dataset("graph_node_map", data=batch["graph_node_map"], dtype='int32')


def main(sub_data):
    original_path = Path(args.dataset_path + '/g_pc')
    batch_path = Path(args.dataset_path + '/g_pc_batch')
    graphs_num = args.graphs_num
    if not batch_path.exists():
        batch_path.mkdir()

    if osp.exists(osp.join(batch_path, f'{sub_data}_batches.h5')):
        response = input("The file already exists. Do you want to delete it? (yes/no):")
        if response.lower() == 'yes':
            os.remove(osp.join(batch_path, f'{sub_data}_batches.h5'))
            print("The file has been deleted")
        else:
            print("The file has been retained")

    if args.dataset_name == 'MFCAD' or args.dataset_name == 'MFCAD2':
        generator = GenerateSemanticBatch(sub_data, osp.join(original_path, f'{sub_data}_graphs.h5'),
                                          osp.join(batch_path, f'{sub_data}_batches.h5'), graphs_num)
    else:
        generator = GenerateBatch(sub_data, osp.join(original_path, f'{sub_data}_graphs.h5'),
                                  osp.join(batch_path, f'{sub_data}_batches.h5'), graphs_num)
    generator.process()

    gc.collect()
    print("The batches packaging is complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path of the dataset")
    parser.add_argument("--graphs_num", type=int, default=30,
                        help="The number of graphs contained in a batch")
    parser.add_argument("--normalize", type=bool, default=True,
                        help="Whether to normalize the data. Defaults to True.")
    parser.add_argument("--center_and_scale", type=bool, default=True,
                        help="Whether to center and scale the solid. Defaults to True.")
    args = parser.parse_args()

    for sub_dataset in ['train', 'val', 'test']:
        main(sub_dataset)
