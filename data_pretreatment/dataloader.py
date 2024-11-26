import h5py
import numpy as np
import pathlib
import torch


class GPCDataloader:
    def __init__(self,
                 root_dir,
                 dataset_type,
                 transform=None):
        """
        Load the Dataset from the root directory.

        Args:
            root_dir (str): Root path of the dataset.
            dataset_type (str): Data split to load.
            transform (callable, optional): Transformation to apply to the data.
        """
        self.path = pathlib.Path(root_dir)
        self.dataset_type = dataset_type
        self.batch_path = self.path.joinpath(f"{self.dataset_type}_batches.h5")
        self.transform = transform

        with h5py.File(self.batch_path, 'r') as hf:
            self.total_batches = len(hf.keys())

    def num_batches(self):
        return self.total_batches

    def gpc_dataloader(self):
        """Yield data from the datasets."""
        with h5py.File(self.batch_path, 'r') as hf:
            for key in hf.keys():
                group = hf[key]

                data_dict = {
                    "graph_face_attr": np.array(group["graph_face_attr"]),
                    "graph_face_grid": np.array(group["graph_face_grid"]),
                    "graph_edge_attr": np.array(group["graph_edge_attr"]),
                    "graph_edge_grid": np.array(group["graph_edge_grid"]),
                    "edges": np.array(group["edges"]),
                    "points": np.array(group["points"]),
                    "normals": np.array(group["normals"]),
                    "face_point_map": np.array(group["face_point_map"]),
                    "graph_point_map": np.array(group["graph_point_map"]),
                    "graph_node_map": np.array(group["graph_node_map"]),
                    "segmentation_labels": np.array(group["segmentation_labels"]),
                    "instance_labels": np.array(group["instance_labels"]),
                    "bottom_labels": np.array(group["bottom_labels"]),
                    "point_seg_labels": np.array(group["point_seg_labels"]),
                    "point_bottom_labels": np.array(group["point_bottom_labels"])
                }

                # Covert the numpy arrays to tensors
                tensor_data_dict = {
                    "graph_face_attr":
                        torch.tensor(data_dict["graph_face_attr"], dtype=torch.float).to('cuda', non_blocking=True),
                    "graph_face_grid":
                        torch.tensor(data_dict["graph_face_grid"], dtype=torch.float).to('cuda', non_blocking=True),
                    "graph_edge_attr":
                        torch.tensor(data_dict["graph_edge_attr"], dtype=torch.float).to('cuda', non_blocking=True),
                    "graph_edge_grid":
                        torch.tensor(data_dict["graph_edge_grid"], dtype=torch.float).to('cuda', non_blocking=True),
                    "edges":
                        torch.tensor(data_dict["edges"], dtype=torch.long).to('cuda', non_blocking=True),
                    "points":
                        torch.tensor(data_dict["points"], dtype=torch.float).to('cuda', non_blocking=True),
                    "normals":
                        torch.tensor(data_dict["normals"], dtype=torch.float).to('cuda', non_blocking=True),
                    "face_point_map":
                        torch.tensor(data_dict["face_point_map"], dtype=torch.long).to('cuda', non_blocking=True),
                    "graph_point_map":
                        torch.tensor(data_dict["graph_point_map"], dtype=torch.long).to('cuda', non_blocking=True),
                    "graph_node_map":
                        torch.tensor(data_dict["graph_node_map"], dtype=torch.long).to('cuda', non_blocking=True)
                }

                tensor_label_dict = {
                    "segmentation_labels":
                        torch.tensor(data_dict["segmentation_labels"], dtype=torch.long).to('cuda', non_blocking=True),
                    "instance_labels":
                        torch.tensor(data_dict["instance_labels"], dtype=torch.float).to('cuda', non_blocking=True),
                    "bottom_labels":
                        torch.tensor(data_dict["bottom_labels"], dtype=torch.float).to('cuda', non_blocking=True),
                    "point_seg_labels":
                        torch.tensor(data_dict["point_seg_labels"], dtype=torch.long).to('cuda', non_blocking=True),
                    "point_bottom_labels":
                        torch.tensor(data_dict["point_bottom_labels"], dtype=torch.float).to('cuda', non_blocking=True)
                }

                yield tensor_data_dict, tensor_label_dict

    def gpc_dataloader_semantic(self):
        """Yield data from the datasets."""
        with h5py.File(self.batch_path, 'r') as hf:
            for key in hf.keys():
                group = hf[key]

                data_dict = {
                    "graph_face_attr": np.array(group["graph_face_attr"]),
                    "graph_face_grid": np.array(group["graph_face_grid"]),
                    "graph_edge_attr": np.array(group["graph_edge_attr"]),
                    "graph_edge_grid": np.array(group["graph_edge_grid"]),
                    "edges": np.array(group["edges"]),
                    "points": np.array(group["points"]),
                    "normals": np.array(group["normals"]),
                    "face_point_map": np.array(group["face_point_map"]),
                    "graph_point_map": np.array(group["graph_point_map"]),
                    "graph_node_map": np.array(group["graph_node_map"]),
                    "segmentation_labels": np.array(group["segmentation_labels"]),
                    "point_seg_labels": np.array(group["point_seg_labels"])
                }

                # Covert the numpy arrays to tensors
                tensor_data_dict = {
                    "graph_face_attr":
                        torch.tensor(data_dict["graph_face_attr"], dtype=torch.float).to('cuda', non_blocking=True),
                    "graph_face_grid":
                        torch.tensor(data_dict["graph_face_grid"], dtype=torch.float).to('cuda', non_blocking=True),
                    "graph_edge_attr":
                        torch.tensor(data_dict["graph_edge_attr"], dtype=torch.float).to('cuda', non_blocking=True),
                    "graph_edge_grid":
                        torch.tensor(data_dict["graph_edge_grid"], dtype=torch.float).to('cuda', non_blocking=True),
                    "edges":
                        torch.tensor(data_dict["edges"], dtype=torch.long).to('cuda', non_blocking=True),
                    "points":
                        torch.tensor(data_dict["points"], dtype=torch.float).to('cuda', non_blocking=True),
                    "normals":
                        torch.tensor(data_dict["normals"], dtype=torch.float).to('cuda', non_blocking=True),
                    "face_point_map":
                        torch.tensor(data_dict["face_point_map"], dtype=torch.long).to('cuda', non_blocking=True),
                    "graph_point_map":
                        torch.tensor(data_dict["graph_point_map"], dtype=torch.long).to('cuda', non_blocking=True),
                    "graph_node_map":
                        torch.tensor(data_dict["graph_node_map"], dtype=torch.long).to('cuda', non_blocking=True)
                }

                tensor_label_dict = {
                    "segmentation_labels":
                        torch.tensor(data_dict["segmentation_labels"], dtype=torch.long).to('cuda', non_blocking=True),
                    "point_seg_labels":
                        torch.tensor(data_dict["point_seg_labels"], dtype=torch.long).to('cuda', non_blocking=True)
                }

                yield tensor_data_dict, tensor_label_dict
