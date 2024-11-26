import h5py
import json
import numpy as np
import os
import os.path as osp


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


class WriteH5:
    def __init__(self, h5_path, label_path, attr_path):
        self.h5_path = h5_path
        self.label_path = label_path
        self.attr_path = attr_path

    def extract_labels(self, fn, num_faces):
        """
        Extract labels from a json
        """

        label_file = osp.join(self.label_path, fn + '.json')
        labels_data = load_json(label_file)

        _, labels = labels_data[0]
        seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']
        assert len(seg_label) == len(inst_label) and len(seg_label) == len(bottom_label), \
            'have wrong label: ' + fn
        assert num_faces == len(seg_label), \
            'File {} have wrong number of labels {} with AAG faces {}. '.format(
                fn, len(seg_label), num_faces)
        # Read semantic segmentation label for each face
        face_segmentation_labels = np.zeros(num_faces)
        for idx, face_id in enumerate(range(num_faces)):
            index = seg_label[str(face_id)]
            face_segmentation_labels[idx] = index
        # Read instance segmentation labels for each instance
        # Just a face adjacency
        instance_labels = np.array(inst_label, dtype=np.int32)
        # Read bottom face segmentation label for each face
        bottom_segmentation_labels = np.zeros(num_faces)
        for idx, face_id in enumerate(range(num_faces)):
            index = bottom_label[str(face_id)]
            bottom_segmentation_labels[idx] = index

        return face_segmentation_labels, instance_labels, bottom_segmentation_labels

    def write_h5_file(self, results):
        """
        Save all step files in a hdf5
        """

        if osp.exists(self.h5_path):
            response = input("The file already exists. Do you want to delete it? (yes/no):")
            if response.lower() == 'yes':
                os.remove(self.h5_path)
                print("The file has been deleted")
            else:
                print("The file has been retained")

        with h5py.File(self.h5_path, 'a') as hf:
            for result in results:
                assert isinstance(result[0], str)
                assert isinstance(result[1], dict)
                group = hf.create_group(result[0])
                data_dict = result[1]
                graph = data_dict["graph"]
                graph_face_attr = data_dict["graph_face_attr"]
                graph_face_grid = data_dict["graph_face_grid"]
                graph_edge_attr = data_dict["graph_edge_attr"]
                graph_edge_grid = data_dict["graph_edge_grid"]
                points = data_dict["points"]
                normals = data_dict["normals"]
                sequence = data_dict["sequence"]

                group.create_dataset("edges", data=graph["edges"], compression="lzf")
                group.create_dataset("num_nodes", data=graph["num_nodes"])
                group.create_dataset("graph_face_attr", data=graph_face_attr, compression="lzf")
                group.create_dataset("graph_face_grid", data=graph_face_grid, compression="lzf")
                group.create_dataset("graph_edge_attr", data=graph_edge_attr, compression="lzf")
                group.create_dataset("graph_edge_grid", data=graph_edge_grid, compression="lzf")
                group.create_dataset("points", data=points, compression="lzf")
                group.create_dataset("normals", data=normals, compression="lzf")
                group.create_dataset("sequence", data=sequence, compression="lzf")

                segmentation_labels, instance_labels, bottom_labels = self.extract_labels(
                    result[0], graph["num_nodes"].item())
                group.create_dataset("segmentation_labels", data=segmentation_labels, compression="lzf")
                group.create_dataset("instance_labels", data=instance_labels, compression="lzf")
                group.create_dataset("bottom_labels", data=bottom_labels, compression="lzf")

    def write_attr_data(self, data):
        """Export a data to a json file"""

        if osp.exists(self.attr_path):
            response = input("The attribute data already exists. Do you want to delete it? (yes/no):")
            if response.lower() == 'yes':
                os.remove(self.attr_path)
                print("The attribute data has been deleted")
            else:
                print("The attribute data has been retained")

        with h5py.File(self.attr_path, 'a') as hf:
            group = hf.create_group('attribute statistics')
            mean_face_attr = data["mean_face_attr"]
            std_face_attr = data["std_face_attr"]
            mean_edge_attr = data["mean_edge_attr"]
            std_edge_attr = data["std_edge_attr"]

            group.create_dataset("mean_face_attr", data=mean_face_attr, compression="lzf")
            group.create_dataset("std_face_attr", data=std_face_attr, compression="lzf")
            group.create_dataset("mean_edge_attr", data=mean_edge_attr, compression="lzf")
            group.create_dataset("std_edge_attr", data=std_edge_attr, compression="lzf")


class WriteH5MFCAD:
    def __init__(self, h5_path, label_path, attr_path):
        self.h5_path = h5_path
        self.label_path = label_path
        self.attr_path = attr_path

    def extract_labels(self, fn, num_faces):
        """
        Extract labels from a json
        """

        label_file = osp.join(self.label_path, fn + '_color_ids.json')
        label = load_json(label_file)
        seg_label = []
        for face in label["body"]["faces"]:
            index = face["segment"]["index"]
            seg_label.append(index)

        assert num_faces == len(seg_label), \
            'File {} have wrong number of labels {} with AAG faces {}. '.format(
                fn, len(seg_label), num_faces)
        # Read semantic segmentation label for each face
        face_segmentation_labels = np.array(seg_label, dtype=np.int32)

        return face_segmentation_labels

    def write_h5_file(self, results):
        """
        Save all step files in a hdf5
        """

        if osp.exists(self.h5_path):
            response = input("The file already exists. Do you want to delete it? (yes/no):")
            if response.lower() == 'yes':
                os.remove(self.h5_path)
                print("The file has been deleted")
            else:
                print("The file has been retained")

        with h5py.File(self.h5_path, 'a') as hf:
            for result in results:
                assert isinstance(result[0], str)
                assert isinstance(result[1], dict)
                group = hf.create_group(result[0])
                data_dict = result[1]
                graph = data_dict["graph"]
                graph_face_attr = data_dict["graph_face_attr"]
                graph_face_grid = data_dict["graph_face_grid"]
                graph_edge_attr = data_dict["graph_edge_attr"]
                graph_edge_grid = data_dict["graph_edge_grid"]
                points = data_dict["points"]
                normals = data_dict["normals"]
                sequence = data_dict["sequence"]

                group.create_dataset("edges", data=graph["edges"], compression="lzf")
                group.create_dataset("num_nodes", data=graph["num_nodes"])
                group.create_dataset("graph_face_attr", data=graph_face_attr, compression="lzf")
                group.create_dataset("graph_face_grid", data=graph_face_grid, compression="lzf")
                group.create_dataset("graph_edge_attr", data=graph_edge_attr, compression="lzf")
                group.create_dataset("graph_edge_grid", data=graph_edge_grid, compression="lzf")
                group.create_dataset("points", data=points, compression="lzf")
                group.create_dataset("normals", data=normals, compression="lzf")
                group.create_dataset("sequence", data=sequence, compression="lzf")

                segmentation_labels = self.extract_labels(result[0], graph["num_nodes"].item())
                group.create_dataset("segmentation_labels", data=segmentation_labels, compression="lzf")

    def write_attr_data(self, data):
        """Export a data to a json file"""

        if osp.exists(self.attr_path):
            response = input("The attribute data already exists. Do you want to delete it? (yes/no):")
            if response.lower() == 'yes':
                os.remove(self.attr_path)
                print("The attribute data has been deleted")
            else:
                print("The attribute data has been retained")

        with h5py.File(self.attr_path, 'a') as hf:
            group = hf.create_group('attribute statistics')
            mean_face_attr = data["mean_face_attr"]
            std_face_attr = data["std_face_attr"]
            mean_edge_attr = data["mean_edge_attr"]
            std_edge_attr = data["std_edge_attr"]

            group.create_dataset("mean_face_attr", data=mean_face_attr, compression="lzf")
            group.create_dataset("std_face_attr", data=std_face_attr, compression="lzf")
            group.create_dataset("mean_edge_attr", data=mean_edge_attr, compression="lzf")
            group.create_dataset("std_edge_attr", data=std_edge_attr, compression="lzf")


class WriteH5MFCAD2:
    def __init__(self, h5_path, label_path, attr_path):
        self.h5_path = h5_path
        self.label_path = label_path
        self.attr_path = attr_path

    def extract_labels(self, fn, num_faces):
        """
        Extract labels from a json
        """

        label_file = osp.join(self.label_path, fn + '.json')
        seg_label = load_json(label_file)

        assert num_faces == len(seg_label), \
            'File {} have wrong number of labels {} with AAG faces {}. '.format(
                fn, len(seg_label), num_faces)
        # Read semantic segmentation label for each face
        face_segmentation_labels = np.array(seg_label, dtype=np.int32)

        return face_segmentation_labels

    def write_h5_file(self, results):
        """
        Save all step files in a hdf5
        """

        if osp.exists(self.h5_path):
            response = input("The file already exists. Do you want to delete it? (yes/no):")
            if response.lower() == 'yes':
                os.remove(self.h5_path)
                print("The file has been deleted")
            else:
                print("The file has been retained")

        with h5py.File(self.h5_path, 'a') as hf:
            for result in results:
                assert isinstance(result[0], str)
                assert isinstance(result[1], dict)
                group = hf.create_group(result[0])
                data_dict = result[1]
                graph = data_dict["graph"]
                graph_face_attr = data_dict["graph_face_attr"]
                graph_face_grid = data_dict["graph_face_grid"]
                graph_edge_attr = data_dict["graph_edge_attr"]
                graph_edge_grid = data_dict["graph_edge_grid"]
                points = data_dict["points"]
                normals = data_dict["normals"]
                sequence = data_dict["sequence"]

                group.create_dataset("edges", data=graph["edges"], compression="lzf")
                group.create_dataset("num_nodes", data=graph["num_nodes"])
                group.create_dataset("graph_face_attr", data=graph_face_attr, compression="lzf")
                group.create_dataset("graph_face_grid", data=graph_face_grid, compression="lzf")
                group.create_dataset("graph_edge_attr", data=graph_edge_attr, compression="lzf")
                group.create_dataset("graph_edge_grid", data=graph_edge_grid, compression="lzf")
                group.create_dataset("points", data=points, compression="lzf")
                group.create_dataset("normals", data=normals, compression="lzf")
                group.create_dataset("sequence", data=sequence, compression="lzf")

                segmentation_labels = self.extract_labels(result[0], graph["num_nodes"].item())
                group.create_dataset("segmentation_labels", data=segmentation_labels, compression="lzf")

    def write_attr_data(self, data):
        """Export a data to a json file"""

        if osp.exists(self.attr_path):
            response = input("The attribute data already exists. Do you want to delete it? (yes/no):")
            if response.lower() == 'yes':
                os.remove(self.attr_path)
                print("The attribute data has been deleted")
            else:
                print("The attribute data has been retained")

        with h5py.File(self.attr_path, 'a') as hf:
            group = hf.create_group('attribute statistics')
            mean_face_attr = data["mean_face_attr"]
            std_face_attr = data["std_face_attr"]
            mean_edge_attr = data["mean_edge_attr"]
            std_edge_attr = data["std_edge_attr"]

            group.create_dataset("mean_face_attr", data=mean_face_attr, compression="lzf")
            group.create_dataset("std_face_attr", data=std_face_attr, compression="lzf")
            group.create_dataset("mean_edge_attr", data=mean_edge_attr, compression="lzf")
            group.create_dataset("std_edge_attr", data=std_edge_attr, compression="lzf")
