from torch_geometric.typing import WITH_TORCH_CLUSTER

from models.encoder import *
from models.decoder import *

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class GPCNetSegmentor(nn.Module):
    def __init__(self,
                 node_attr_dim,
                 node_grid_dim,
                 edge_attr_dim,
                 edge_grid_dim,
                 graph_out_channels,
                 graph_num_layers,
                 point_attr_dim,
                 point_out_channels,
                 num_classes,
                 hidden_channels=256,
                 sample_ratio=(0.2, 0.25),
                 knn_num=(18, 12),
                 neighbor_num=(1, 3, 3)
                 ):
        super().__init__()

        self.grid_enc = GridEncoder(
            node_grid_dim=node_grid_dim,
            node_grid_emb=128
        )

        self.node_encoder = Linear(node_attr_dim, 128)
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)

        self.graph_enc = GraphEncoder(
            hidden_channels=hidden_channels,
            graph_out_channels=graph_out_channels,
            num_layers=graph_num_layers,
            dropout=0.2
        )

        self.point_enc = PointEncoder(
            point_attr_dim=point_attr_dim,
            point_out_channels=point_out_channels,
            sample_ratio=sample_ratio,
            knn_num=knn_num,
            neighbor_num=neighbor_num
        )

        self.seg_head = SemanticHead(in_channels=hidden_channels, num_classes=num_classes)

        self.inst_head = InstanceHead(in_channels=hidden_channels, out_channels=256)

        self.bottom_head = BottomHead(in_channels=hidden_channels, out_channels=1)

    def forward(self, data_dict):
        # Load graph data
        face_attr = data_dict["graph_face_attr"]
        face_grid = data_dict["graph_face_grid"]
        edge_attr = data_dict["graph_edge_attr"]
        edge_grid = data_dict["graph_edge_grid"]
        edge_index = data_dict["edges"]

        # Load point cloud data
        points = data_dict["points"]
        normals = data_dict["normals"]
        face_point_map = data_dict["face_point_map"]
        graph_point_map = data_dict["graph_point_map"]
        graph_node_map = data_dict["graph_node_map"]

        # Forward propagation
        face_grid_feat = self.grid_enc(face_grid)
        face_attr_feat = self.node_encoder(face_attr)
        edge_feat = self.edge_encoder(edge_attr)
        face_feat = torch.cat([face_attr_feat, face_grid_feat], dim=1)

        face_embed, graph_embed = self.graph_enc(face_feat, edge_index, edge_feat, graph_node_map)
        graph_embed = graph_embed[graph_point_map]
        face_embed = face_embed[face_point_map]
        point_embed = self.point_enc(normals, points, graph_point_map)

        seg_out = self.seg_head(torch.cat([graph_embed, face_embed, point_embed], dim=1),
                                face_point_map)

        inst_out = self.inst_head(torch.cat([graph_embed, face_embed, point_embed], dim=1),
                                  face_point_map,
                                  graph_node_map)

        bottom_out = self.bottom_head(torch.cat([graph_embed, face_embed, point_embed], dim=1),
                                      face_point_map)

        return seg_out, inst_out, bottom_out


class GPCNetSemanticSegmentor(nn.Module):
    def __init__(self,
                 node_attr_dim,
                 node_grid_dim,
                 edge_attr_dim,
                 edge_grid_dim,
                 graph_out_channels,
                 graph_num_layers,
                 point_attr_dim,
                 point_out_channels,
                 num_classes,
                 hidden_channels=256,
                 sample_ratio=(0.2, 0.25),
                 knn_num=(18, 12),
                 neighbor_num=(1, 3, 3)
                 ):
        super().__init__()

        self.grid_enc = GridEncoder(
            node_grid_dim=node_grid_dim,
            node_grid_emb=128
        )

        self.node_encoder = Linear(node_attr_dim, 128)
        self.edge_encoder = Linear(edge_attr_dim, hidden_channels)

        self.graph_enc = GraphEncoder(
            hidden_channels=hidden_channels,
            graph_out_channels=graph_out_channels,
            num_layers=graph_num_layers,
            dropout=0.2
        )

        self.point_enc = PointEncoder(
            point_attr_dim=point_attr_dim,
            point_out_channels=point_out_channels,
            sample_ratio=sample_ratio,
            knn_num=knn_num,
            neighbor_num=neighbor_num
        )

        self.seg_head = SemanticHead(in_channels=hidden_channels, num_classes=num_classes)

    def forward(self, data_dict):
        # Load graph data
        face_attr = data_dict["graph_face_attr"]
        face_grid = data_dict["graph_face_grid"]
        edge_attr = data_dict["graph_edge_attr"]
        edge_grid = data_dict["graph_edge_grid"]
        edge_index = data_dict["edges"]

        # Load point cloud data
        points = data_dict["points"]
        normals = data_dict["normals"]
        face_point_map = data_dict["face_point_map"]
        graph_point_map = data_dict["graph_point_map"]
        graph_node_map = data_dict["graph_node_map"]

        # Forward propagation
        face_grid_feat = self.grid_enc(face_grid)
        face_attr_feat = self.node_encoder(face_attr)
        edge_feat = self.edge_encoder(edge_attr)
        face_feat = torch.cat([face_attr_feat, face_grid_feat], dim=1)

        face_embed, graph_embed = self.graph_enc(face_feat, edge_index, edge_feat, graph_node_map)
        graph_embed = graph_embed[graph_point_map]
        face_embed = face_embed[face_point_map]
        point_embed = self.point_enc(normals, points, graph_point_map)

        seg_out = self.seg_head(torch.cat([graph_embed, face_embed, point_embed], dim=1),
                                face_point_map)

        return seg_out
