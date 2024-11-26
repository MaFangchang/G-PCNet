# -*- coding: utf-8 -*-
import json
import torch
import os.path as osp
import numpy as np
from pathlib import Path

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle,
                              GeomAbs_Ellipse)
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Dir
from OCC.Core.GProp import GProp_GProps
from OCC.Core.ShapeAnalysis import shapeanalysis_GetFaceUVBounds
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Solid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Extend import TopologyUtils

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from occwl.graph import face_adjacency


#######################################
# Calculate mean & std of attributes
#######################################


def scale_solid_to_unit_box(solid):
    """
    We want to apply a transform so that the solid is centered on the origin and scaled,
    so it just fits into a box [-1, 1]^3
    """

    if isinstance(solid, Solid):
        return solid.scale_to_unit_box(copy=True)

    solid = Solid(solid, allow_compound=True)
    solid = solid.scale_to_unit_box(copy=True)

    return solid.topods_shape()


def extract_point_cloud(face, num_srf_u, num_srf_v):
    """
    Extract point grid from surface
    face: Must be TopoDS_Face
    """

    # Obtain the UV boundaries of the face
    u_min, u_max, v_min, v_max = shapeanalysis_GetFaceUVBounds(face)
    u_length = u_max - u_min
    v_length = v_max - v_min

    # Adjust the step size based on the dimensions of the face
    u_step_size = u_length / (num_srf_u + 1)
    v_step_size = v_length / (num_srf_v + 1)

    points = []
    normals = []
    surface = BRep_Tool.Surface(face)
    props = GeomLProp_SLProps(surface, 1, 1e-6)

    last_normal = None  # Initialize the last successful normal

    # Step through the U and V parameters
    u = u_min + u_step_size
    for i in range(num_srf_u):
        v = v_min + v_step_size
        for j in range(num_srf_v):
            try:
                props.SetParameters(u, v)
                point = props.Value()
                normal = props.Normal()
                if face.Orientation() == TopAbs_REVERSED:
                    normal.Reverse()
                last_normal = normal
            except RuntimeError as e:
                print(f"Warning: Cannot compute normal at u={u}, v={v}: {e}")
                if last_normal is None:
                    # If this is the first point and normal computation fails
                    normal = gp_Dir(1., 0., 0.)
                    point = props.Value()
                else:
                    # Use the last successful normal
                    normal = last_normal
                    point = props.Value()

            points.append(list(point.Coord()))
            normals.append(list(normal.Coord()))
            v += v_step_size
        u += u_step_size

    return np.array(points), np.array(normals)


class TopologyChecker:
    # Modified from BREPNET:
    # https://github.com/AutodeskAILab/BRepNet/blob/master/pipeline/extract_brepnet_data_from_step.py
    def __init__(self):
        pass

    @staticmethod
    def find_edges_from_wires(top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    @staticmethod
    def find_edges_from_top_exp(top_exp):
        edge_set = set(top_exp.edges())
        return edge_set

    def check_closed(self, body):
        """
        In Open Cascade, unlinked (open) edges can be identified as they appear in the
        edges iterator when ignore_orientation=False but are not present in any wire
        """

        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    @staticmethod
    def check_manifold(top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True

    @staticmethod
    def check_unique_coedges(top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                # We want to detect the case where the coedges are not unique
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True

    def __call__(self, body):
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        if top_exp.number_of_faces() == 0:
            print('Empty shape')
            return False
        # OCC.BRepCheck, perform topology and geometrical check
        analyzer = BRepCheck_Analyzer(body)
        if not analyzer.IsValid(body):
            print('BRepCheck_Analyzer found defects')
            return False
        # Other topology check
        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return False
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return False
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return False
        return True


class GPCExtractor:
    def __init__(self, step_file, attribute_schema, topo_checker, scale_body=True):
        self.step_file = step_file
        self.attribute_schema = attribute_schema
        self.scale_body = scale_body
        self.body = None
        self.topo_checker = topo_checker
        self.num_srf_u = self.attribute_schema["UV-grid"]["num_srf_u"]
        self.num_srf_v = self.attribute_schema["UV-grid"]["num_srf_v"]
        self.num_crv_u = self.attribute_schema["UV-grid"]["num_crv_u"]

    def process(self):
        """
        Obtaining a face-edge attributed adjacency graph and face sampling points from the given shape (Solid)
        Args: solid
        Returns: graph and point cloud
        """

        # Load the body from the STEP file
        self.body = self.load_body_from_step()
        assert self.body is not None, \
            "the shape {} is non-manifold or open".format(self.step_file)
        assert self.topo_checker(self.body), \
            "the shape {} has wrong topology".format(self.step_file)
        assert isinstance(self.body, TopoDS_Solid), \
            'file {} is {}, not TopoDS_Solid'.format(self.step_file, type(self.body))

        # Scaling shapes to unit volume
        if self.scale_body:
            self.body = scale_solid_to_unit_box(self.body)

        # Build face adjacency graph with B-rep entities as node and edge features
        try:
            graph = face_adjacency(Solid(self.body))
        except Exception as e:
            print(e)
            assert False, 'Wrong shape {}'.format(self.step_file)

        graph_face_attr = []
        graph_face_grid = []
        points = []
        normals = []
        sequence = []
        serial = 0
        # The FaceCentroidAttribute has xyz coordinate so the length of face attributes should add 2 if containing centroid
        len_of_face_attr = len(self.attribute_schema["face_attributes"]) + \
            2 if "FaceCentroidAttribute" in self.attribute_schema["face_attributes"] else 0
        for face_idx in graph.nodes:
            # Get the B-rep face
            face = graph.nodes[face_idx]["face"]
            # Get the attributes from face
            face_attr = self.extract_attributes_from_face(
                face.topods_shape())  # From occwl.Face to OCC.TopoDS_Face
            assert len_of_face_attr == len(face_attr)
            graph_face_attr.append(face_attr)
            # Get the UV point grid from face
            if self.num_srf_u and self.num_srf_v:
                uv_grid = self.extract_face_point_grid(face)
                face_points, face_normals = extract_point_cloud(face.topods_shape(), self.num_srf_u, self.num_srf_v)
                assert face_points.shape[0] == (self.num_srf_u * self.num_srf_v)
                # assert face_points.shape[0] >= 3
                graph_face_grid.append(uv_grid.tolist())
                serial += len(face_points)
                sequence.append(serial)
                points.append(face_points.tolist())
                normals.append(face_normals.tolist())
        graph_face_attr = np.array(graph_face_attr)
        graph_face_grid = np.array(graph_face_grid)
        points = np.vstack(points)
        normals = np.vstack(normals)
        sequence = np.array(sequence)

        graph_edge_attr = []
        graph_edge_grid = []
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            # Ignore dgenerate edges, e.g. at apex of cone
            if not edge.has_curve():
                continue
            # Get the attributes from edge
            edge = edge.topods_shape()  # From occwl.Edge to OCC.TopoDS_Edge
            edge_attr = self.extract_attributes_from_edge(edge)
            assert len(self.attribute_schema["edge_attributes"]) == len(edge_attr)
            graph_edge_attr.append(edge_attr)
            # get the UV point grid from edge
            if self.num_crv_u:
                u_grid = self.extract_edge_point_grid(edge)
                assert u_grid.shape[0] == 12
                graph_edge_grid.append(u_grid.tolist())
        graph_edge_attr = np.array(graph_edge_attr)
        graph_edge_grid = np.array(graph_edge_grid)

        # Get graph from nx.DiGraph
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        graph = {
            'edges': np.array([src, dst]),
            'num_nodes': np.array(len(graph.nodes))
        }

        return {
            'graph': graph,
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
            'points': points,
            'normals': normals,
            'sequence': sequence
        }

    ########################
    # Step Loader
    ########################

    def load_body_from_step(self):
        """
        Load the body from the step file.
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    ########################
    # Face Attributes Extractor
    ########################

    def extract_attributes_from_face(self, face) -> list:
        def plane_attribute():
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Plane:
                return 1.0
            return 0.0

        def cylinder_attribute():
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cylinder:
                return 1.0
            return 0.0

        def cone_attribute():
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cone:
                return 1.0
            return 0.0

        def sphere_attribute():
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Sphere:
                return 1.0
            return 0.0

        def torus_attribute():
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Torus:
                return 1.0
            return 0.0

        def area_attribute():
            geometry_properties = GProp_GProps()
            brepgprop_SurfaceProperties(face, geometry_properties)
            return geometry_properties.Mass()

        def rational_nurbs_attribute():
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_BSplineSurface:
                bspline = surf.BSpline()
            elif surf.GetType() == GeomAbs_BezierSurface:
                bspline = surf.Bezier()
            else:
                bspline = None

            if bspline is not None:
                if bspline.IsURational() or bspline.IsVRational():
                    return 1.0
            return 0.0

        def centroid_attribute():
            """Get centroid normal vector of B-Rep face"""

            mass_props = GProp_GProps()
            brepgprop_SurfaceProperties(face, mass_props)
            g_pt = mass_props.CentreOfMass()

            return g_pt.Coord()

        face_attributes = []
        for attribute in self.attribute_schema["face_attributes"]:
            if attribute == "Plane":
                face_attributes.append(plane_attribute())
            elif attribute == "Cylinder":
                face_attributes.append(cylinder_attribute())
            elif attribute == "Cone":
                face_attributes.append(cone_attribute())
            elif attribute == "SphereFaceAttribute":
                face_attributes.append(sphere_attribute())
            elif attribute == "TorusFaceAttribute":
                face_attributes.append(torus_attribute())
            elif attribute == "FaceAreaAttribute":
                face_attributes.append(area_attribute())
            elif attribute == "RationalNurbsFaceAttribute":
                face_attributes.append(rational_nurbs_attribute())
            elif attribute == "FaceCentroidAttribute":
                face_attributes.extend(centroid_attribute())
            else:
                assert False, "Unknown face attribute"
        return face_attributes

    ########################
    # Edge Attributes Extractor
    ########################

    def extract_attributes_from_edge(self, edge) -> list:
        def find_edge_convexity(faces):
            edge_data = EdgeDataExtractor(Edge(edge), faces, use_arclength_params=False)
            if not edge_data.good:
                # This is the case where the edge is a pole of a sphere
                return 0.0
            angle_tol_rads = 0.0872664626  # 5 degrees
            convexity = edge_data.edge_convexity(angle_tol_rads)
            return convexity

        def convexity_attribute(convexity, attribute):
            if attribute == "Convex edge":
                return convexity == EdgeConvexity.CONVEX
            if attribute == "Concave edge":
                return convexity == EdgeConvexity.CONCAVE
            if attribute == "Smooth":
                return convexity == EdgeConvexity.SMOOTH
            assert False, "Unknown convexity"
            return 0.0

        def edge_length_attribute():
            geometry_properties = GProp_GProps()
            brepgprop_LinearProperties(edge, geometry_properties)
            return geometry_properties.Mass()

        def circular_edge_attribute():
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curve_type = brep_adaptor_curve.GetType()
            if curve_type == GeomAbs_Circle:
                return 1.0
            return 0.0

        def closed_edge_attribute():
            if BRep_Tool().IsClosed(edge):
                return 1.0
            return 0.0

        def elliptical_edge_attribute():
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curve_type = brep_adaptor_curve.GetType()
            if curve_type == GeomAbs_Ellipse:
                return 1.0
            return 0.0

        def helical_edge_attribute():
            # We don't have this attribute in Open Cascade
            assert False, "Not implemented for the OCC pipeline"
            return 0.0

        def int_curve_edge_attribute():
            # We don't have this attribute in Open Cascade
            assert False, "Not implemented for the OCC pipeline"
            return 0.0

        def straight_edge_attribute():
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curve_type = brep_adaptor_curve.GetType()
            if curve_type == GeomAbs_Line:
                return 1.0
            return 0.0

        def hyperbolic_edge_attribute():
            if Edge(edge).curve_type() == "hyperbola":
                return 1.0
            return 0.0

        def parabolic_edge_attribute():
            if Edge(edge).curve_type() == "parabola":
                return 1.0
            return 0.0

        def bezier_edge_attribute():
            if Edge(edge).curve_type() == "bezier":
                return 1.0
            return 0.0

        def non_rational_bspline_edge_attribute():
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
                return 1.0
            return 0.0

        def rational_bspline_edge_attribute():
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
                return 1.0
            return 0.0

        def offset_edge_attribute():
            if Edge(edge).curve_type() == "offset":
                return 1.0
            return 0.0

        # Get the faces from an edge
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        attribute_list = self.attribute_schema["edge_attributes"]
        if "Concave edge" in attribute_list or \
            "Convex edge" in attribute_list or \
                "Smooth" in attribute_list:
            convexity = find_edge_convexity(faces_of_edge)
        edge_attributes = []
        for attribute in attribute_list:
            if attribute == "Concave edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Convex edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Smooth":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "EdgeLengthAttribute":
                edge_attributes.append(edge_length_attribute())
            elif attribute == "CircularEdgeAttribute":
                edge_attributes.append(circular_edge_attribute())
            elif attribute == "ClosedEdgeAttribute":
                edge_attributes.append(closed_edge_attribute())
            elif attribute == "EllipticalEdgeAttribute":
                edge_attributes.append(elliptical_edge_attribute())
            elif attribute == "HelicalEdgeAttribute":
                edge_attributes.append(helical_edge_attribute())
            elif attribute == "IntcurveEdgeAttribute":
                edge_attributes.append(int_curve_edge_attribute())
            elif attribute == "StraightEdgeAttribute":
                edge_attributes.append(straight_edge_attribute())
            elif attribute == "HyperbolicEdgeAttribute":
                edge_attributes.append(hyperbolic_edge_attribute())
            elif attribute == "ParabolicEdgeAttribute":
                edge_attributes.append(parabolic_edge_attribute())
            elif attribute == "BezierEdgeAttribute":
                edge_attributes.append(bezier_edge_attribute())
            elif attribute == "NonRationalBSplineEdgeAttribute":
                edge_attributes.append(non_rational_bspline_edge_attribute())
            elif attribute == "RationalBSplineEdgeAttribute":
                edge_attributes.append(rational_bspline_edge_attribute())
            elif attribute == "OffsetEdgeAttribute":
                edge_attributes.append(offset_edge_attribute())
            else:
                assert False, "Unknown face attribute"
        return edge_attributes

    ########################
    # Face UV Point Grid Extractor
    ########################

    def extract_face_point_grid(self, face) -> np.array:
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ 7 x num_pts_u x num_pts_v ]

        For each point the values are

            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast

        """
        points = uvgrid(face, self.num_srf_u, self.num_srf_v, method="point")
        normals = uvgrid(face, self.num_srf_u, self.num_srf_v, method="normal")
        mask = uvgrid(face, self.num_srf_u, self.num_srf_v, method="inside")

        # This has shape [ num_pts_u x num_pts_v x 7 ]
        single_grid = np.concatenate([points, normals, mask], axis=2)

        return np.transpose(single_grid, (2, 0, 1))

    ########################
    # Edge UV Point Grid Extractor
    ########################

    def extract_edge_point_grid(self, edge) -> np.array:
        """
        Extract an edge grid (aligned with the coedge direction).

        The edge grids will be of size

            [ 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """

        # get the faces from an edge
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        edge_data = EdgeDataExtractor(Edge(edge), faces_of_edge,
                                      num_samples=self.num_crv_u, use_arclength_params=True)
        if not edge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # In this case we return zeros
            return np.zeros((12, self.num_crv_u))

        single_grid = np.concatenate(
            [
                edge_data.points,
                edge_data.tangents,
                edge_data.left_normals,
                edge_data.right_normals
            ],
            axis=1
        )
        return np.transpose(single_grid, (1, 0))


def initializer():
    """Ignore CTRL+C in the worker process"""

    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def process_one_file(fn, feature_schema):
    topo_checker = TopologyChecker()
    extractor = GPCExtractor(fn, feature_schema, topo_checker)
    out = extractor.process()

    return [str(fn.stem), out]


class WriteBatch:
    def __init__(self, label_path):
        self.label_path = label_path
        self.normalize = True

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

    @staticmethod
    def normalize_data(data, epsilon=1e-10):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        ranges = max_vals - min_vals
        ranges_nonzero = np.where(ranges != 0, ranges, epsilon)

        normalized_data = (data - min_vals) / ranges_nonzero
        normalized_data -= 0.5

        return normalized_data

    @staticmethod
    def point_batches_map(batches):
        repetitions = np.diff(np.insert(batches, 0, 0))

        batches_map = np.repeat(np.arange(len(batches)), repetitions, axis=0)

        return batches_map

    def write_batch(self, result):
        """
        Data preprocessing
        """

        group_data = {}
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)
        data_dict = result[1]
        graph = data_dict["graph"]

        graph_face_attr = data_dict["graph_face_attr"]
        graph_face_grid = data_dict["graph_face_grid"]
        graph_edge_attr = data_dict["graph_edge_attr"]
        graph_edge_grid = data_dict["graph_edge_grid"]
        points = data_dict["points"]
        normals = data_dict["normals"]
        sequence = data_dict["sequence"]

        edges = graph["edges"]
        num_nodes = graph["num_nodes"]
        if num_nodes.size == 0:
            num_nodes = np.array([graph_face_attr.shape[0]])

        if self.normalize:
            graph_face_attr[:, 5] = self.normalize_data(graph_face_attr[:, 5])
            graph_edge_attr[:, 3] = self.normalize_data(graph_edge_attr[:, 3])

        group_data["edges"] = edges
        group_data["num_nodes"] = num_nodes
        group_data["graph_face_attr"] = graph_face_attr
        group_data["graph_face_grid"] = graph_face_grid
        group_data["graph_edge_attr"] = graph_edge_attr
        group_data["graph_edge_grid"] = graph_edge_grid
        group_data["points"] = points
        group_data["normals"] = normals
        group_data["face_point_map"] = self.point_batches_map(sequence)
        group_data["graph_point_map"] = self.point_batches_map(np.array([points.shape[0]]))
        group_data["graph_node_map"] = self.point_batches_map(np.array([num_nodes]))

        return group_data


def main(file_path):
    step_path = Path(file_path).with_suffix('.step')
    label_path = Path(file_path).with_suffix('.json')
    feature_list_path = 'all.json'
    feature_schema = load_json(feature_list_path)

    result = process_one_file(step_path, feature_schema)

    writer = WriteBatch(label_path)
    data_dict = writer.write_batch(result)

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

    return tensor_data_dict
