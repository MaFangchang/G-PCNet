import os
import sys
import random
import torch
import time
import numpy as np
from contextlib import contextmanager
from OCC.Core.STEPControl import STEPControl_Reader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from get_information import main
from models.backbone import GPCNetSegmentor


def seed_torch(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timer(process_name):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"[{process_name}]运行时长：{(end_time - start_time) / 100:.6f}秒")


def read_step_file(file_path: str):
    """
    Read STEP file and return TopoDS_Shape object
    """
    step_reader = STEPControl_Reader()
    if step_reader.ReadFile(file_path) == 1:  # File read successful
        step_reader.TransferRoots()
        return step_reader.OneShape()
    raise Exception("Reading STEP file failed")


class ModelHandler:
    def __init__(self):
        self.config = config
        self.device = config['device']
        seed_torch(config['seed'])
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the model and set its parameters
        """
        model = GPCNetSegmentor(
            node_attr_dim=self.config['node_attr_dim'],
            node_grid_dim=self.config['node_grid_dim'],
            edge_attr_dim=self.config['edge_attr_dim'],
            edge_grid_dim=self.config['edge_grid_dim'],
            graph_out_channels=self.config['graph_out_channels'],
            graph_num_layers=self.config['graph_num_layers'],
            point_attr_dim=self.config['point_attr_dim'],
            point_out_channels=self.config['point_out_channels'],
            num_classes=self.config['num_classes']
        )
        model = model.to(self.device)
        model_param = torch.load(f"weight_best_on_{dataset_name}.pth", map_location=self.device)
        model.load_state_dict(model_param)

        return model

    def predict(self, data):
        """
        Make prediction using the model
        """
        self.model.eval()
        with torch.no_grad():
            # start = time.time()
            seg_pred, inst_pred, bottom_pred = self.model(data)
            # end = time.time()
            # print('前向传播时长: {}'.format(end - start))

            seg_pred = torch.argmax(seg_pred, dim=1)
            inst_pred = (torch.sigmoid(inst_pred) > 0.5).int()
            bottom_pred = (torch.sigmoid(bottom_pred) > 0.5).int()

        return seg_pred.cpu().numpy(), inst_pred.cpu().numpy(), bottom_pred.cpu().numpy()


class FeatureRecognizer:
    def __init__(self, sem_result, inst_result):
        self.semantic_result = sem_result
        self.instance_result = np.squeeze(inst_result)
        self.num_faces = sem_result.shape[0]

    def find_features(self):
        """
        Find all recognized machining features and group faces by their instances
        :return: A dict where keys are feature types and values are dict mapping instance IDs to lists of face indices
        """
        feature_groups = {}
        visited = [False] * self.num_faces  # Track visited faces

        for face_index in range(self.num_faces):
            if visited[face_index]:
                continue  # Skip if this face has already been processed

            feature_index = self.semantic_result[face_index]
            feature_type = feat_names[feature_index]
            if feature_type == '毛坯':
                continue  # Skip stock face

            if feature_type not in feature_groups:
                feature_groups[feature_type] = {}

            # Create a list to store faces belonging to the current instance
            current_instance = [face_index]
            instance_id = len(feature_groups[feature_type])

            # Mark the current face as visited
            visited[face_index] = True

            # Check all other faces for the same instance
            for other_face in range(self.num_faces):
                if other_face != face_index and not visited[other_face]:
                    other_feature_index = self.semantic_result[other_face]
                    other_feature_type = feat_names[other_feature_index]

                    # Check if they are part of the same instance and have the same feature type
                    if (self.instance_result[face_index, other_face] == 1 and
                            self.instance_result[other_face, face_index] == 1 and
                            feature_type == other_feature_type):

                        current_instance.append(other_face)
                        visited[other_face] = True

            # Store the current instance's faces in the feature group
            feature_groups[feature_type][instance_id] = current_instance

        return feature_groups


if __name__ == '__main__':
    dataset_name = 'HeteroMF'
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
        "epochs": 100,
        "seg_a": 1.,
        "inst_a": 1.,
        "bottom_a": 1.,
    }

    feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
                  'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
                  'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring',
                  'blind_hole', 'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
                  'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
                  'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock']

    # Initialize Handler
    model_handler = ModelHandler()

    print("------------------------------Start testing------------------------------")
    # # warm up
    # for i in range(10):
    #     data_dict = main('./example/partA.step')
    #
    #     semantic_result, instance_result, bottom_result = model_handler.predict(data_dict)
    #
    #     recognizer = FeatureRecognizer(semantic_result, instance_result)
    #     features_dict = recognizer.find_features()

    # time test on part A B C D
    with timer("PartA 识别"):
        for i in range(100):
            # start = time.time()
            data_dict = main('./example/partA.step')
            # end = time.time()
            # print('数据转换时长: {}'.format(end - start))

            semantic_result, instance_result, bottom_result = model_handler.predict(data_dict)

            recognizer = FeatureRecognizer(semantic_result, instance_result)
            features_dict = recognizer.find_features()

    with timer("PartB 识别"):
        for i in range(100):
            # start = time.time()
            data_dict = main('./example/partB.step')
            # end = time.time()
            # print('数据转换时长: {}'.format(end - start))

            semantic_result, instance_result, bottom_result = model_handler.predict(data_dict)

            recognizer = FeatureRecognizer(semantic_result, instance_result)
            features_dict = recognizer.find_features()

    with timer("PartC 识别"):
        for i in range(100):
            # start = time.time()
            data_dict = main('./example/partC.step')
            # end = time.time()
            # print('数据转换时长: {}'.format(end - start))

            semantic_result, instance_result, bottom_result = model_handler.predict(data_dict)

            recognizer = FeatureRecognizer(semantic_result, instance_result)
            features_dict = recognizer.find_features()

    with timer("PartD 识别"):
        for i in range(100):
            # start = time.time()
            data_dict = main('./example/partD.step')
            # end = time.time()
            # print('数据转换时长: {}'.format(end - start))

            semantic_result, instance_result, bottom_result = model_handler.predict(data_dict)

            recognizer = FeatureRecognizer(semantic_result, instance_result)
            features_dict = recognizer.find_features()
