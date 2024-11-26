import os
import sys
import random
import torch
import time
import numpy as np
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from contextlib import contextmanager
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Graphic3d import Graphic3d_TOS_SHADING
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Display.OCCViewer import rgb_color
from OCC.Display.SimpleGui import init_display

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
    print(f"[{process_name}]运行时长：{end_time - start_time:.6f}秒")


def read_step_file(file_path: str):
    """
    Read STEP file and return TopoDS_Shape object
    """
    step_reader = STEPControl_Reader()
    if step_reader.ReadFile(file_path) == 1:  # File read successful
        step_reader.TransferRoots()
        return step_reader.OneShape()
    raise Exception("Reading STEP file failed")


class UserInputHandler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        self.selected_feature = None

    @staticmethod
    def select_file() -> str:
        """
        Open a file dialog to select a STEP file and return its path
        """
        file_path = filedialog.askopenfilename(
            title="选择STEP文件",
            filetypes=[("STEP files", "*.step"), ("All files", "*.*")]
        )
        return file_path

    def select_feature(self, features: dict) -> str:
        """
        Display the feature selection dialog box
        :return: the feature selected by user
        """
        features_str = list(features.keys())

        # Create a new window
        selection_window = tk.Toplevel(self.root)
        selection_window.title("选择特征")

        label = tk.Label(selection_window, text="请从已识别出的加工特征中选择你要可视化的特征：")
        label.pack()

        listbox = tk.Listbox(selection_window)
        for feature in features_str:
            listbox.insert(tk.END, feature)
        listbox.pack()

        def on_select():
            if listbox.curselection():
                self.selected_feature = listbox.get(listbox.curselection())
                selection_window.destroy()  # Close the window

        # Create a selection button
        select_button = tk.Button(selection_window, text="确定", command=on_select)
        select_button.pack()

        selection_window.wait_window()  # Wait until the window is closed

        return self.selected_feature


class Visualizer:
    def __init__(self):
        self.display, self.start_display, self.add_menu, self.add_function_to_menu = init_display()

    def fill_color(self, face, rgb_value: tuple, transparency):
        """
        Fill the specified face with a color
        """
        ais_shape = AIS_Shape(face)
        ais_shape.SetColor(rgb_color(*rgb_value))
        ais_shape.SetDisplayMode(Graphic3d_TOS_SHADING)  # Set as fill mode
        self.display.Context.Display(ais_shape, True)  # Display face
        self.display.Context.SetTransparency(ais_shape, transparency, True)

    def highlight_faces_by_instance(self, shape, instance_groups: dict):
        """
        Highlight faces according to different instances with specified colors
        :param shape: TopoDS_Shape
        :param instance_groups: A dict mapping instance IDs to face indices
        """
        color_map = {
            0: (1.0, 1.0, 0.0),  # Yellow
            1: (0.0, 0.0, 1.0),  # Blue
            2: (0.0, 1.0, 0.0),  # Green
            3: (1.0, 0.0, 0.0),  # Red
            # Add more colors as needed
        }

        for instance_id, face_indices in instance_groups.items():
            # Choose a color based on instance ID
            rgb_value = color_map.get(instance_id % len(color_map), (1.0, 1.0, 1.0))  # Default to white if out of range
            self.highlight_face(shape, face_indices, rgb_value)

    def highlight_face(self, shape, face_indices: list, rgb_value: tuple):
        """
        Highlight the face with the specified number as the specified color
        """
        explorer = TopExp_Explorer(shape, TopAbs_FACE)  # Used to traverse all faces
        current_index = 0
        while explorer.More():
            face = explorer.Current()
            if current_index in face_indices:
                self.fill_color(face, rgb_value, 0.3)
            current_index += 1
            explorer.Next()

    def visualize_recognition_result(self, file_path: str, instance_groups: dict):
        """
        Visualize the recognition results with highlighted faces
        """
        step_path = str(Path(file_path).with_suffix('.step'))
        shape = read_step_file(step_path)
        self.display.DisplayShape(shape, update=True)  # Display model
        self.highlight_faces_by_instance(shape, instance_groups)
        self.start_display()


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
        with timer("前向传播"):
            with torch.no_grad():
                seg_pred, inst_pred, bottom_pred = self.model(data)
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

    feat_names = ['倒角', '通孔', '三角通道', '矩形通道', '六边形通道',
                  '三角通槽', '矩形通槽', '圆形通槽',
                  '矩形通台阶', '两侧通台阶', '斜通台阶', 'O型圈', '盲孔',
                  '三角凹槽', '矩形凹槽', '六边形凹槽', '圆端凹槽',
                  '矩形盲槽', '垂直圆端盲槽', '水平圆端盲槽',
                  '三角盲台阶', '圆形盲台阶', '矩形盲台阶', '圆角', '毛坯']

    # Initialize Handler
    model_handler = ModelHandler()
    input_handler = UserInputHandler()

    # Get STEP file name from user
    filename = input_handler.select_file()
    if not filename:
        sys.exit("未选择文件，程序结束")

    # Get input data
    with timer("数据转换"):
        data_dict = main(filename)

    semantic_result, instance_result, bottom_result = model_handler.predict(data_dict)

    with timer("后处理"):
        recognizer = FeatureRecognizer(semantic_result, instance_result)
        features_dict = recognizer.find_features()

    # User selects feature to display in a loop until they cancel
    while True:
        feature_display = input_handler.select_feature(features_dict)
        if feature_display is None:
            sys.exit("未选择有效特征，程序结束")

        instance_display = features_dict[feature_display]

        # Initialize visualization tool
        visualizer = Visualizer()
        visualizer.visualize_recognition_result(file_path=filename, instance_groups=instance_display)

        # Optionally, ask if the user wants to select another feature
        continue_selection = messagebox.askyesno("继续选择", "是否继续选择其他特征？")
        if not continue_selection:
            break
