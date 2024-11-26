import numpy as np

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
from OCC.Extend import TopologyUtils

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from occwl.graph import face_adjacency


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


# def extract_point_cloud(face, num_srf_u, num_srf_v):
#     """
#     Extract point grid from surface
#     """
#
#     points = uvgrid(face, num_srf_u + 2, num_srf_v + 2, method="point")
#     normals = uvgrid(face, num_srf_u + 2, num_srf_v + 2, method="normal")
#     mask = uvgrid(face, num_srf_u + 2, num_srf_v + 2, method="inside")
#
#     points = points[1:num_srf_u + 1, 1:num_srf_v + 1, :].reshape(-1, 3)
#     normals = normals[1:num_srf_u + 1, 1:num_srf_v + 1, :].reshape(-1, 3)
#     mask = mask[1:num_srf_u + 1, 1:num_srf_v + 1, :].reshape(-1, 1)
#     mask = mask.flatten()
#
#     if points[mask].shape[0] < 3:
#         middle_index = points.shape[0] // 2
#         return points[middle_index - 1:middle_index + 2], normals[middle_index - 1:middle_index + 2]
#     else:
#         return points[mask], normals[mask]


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
