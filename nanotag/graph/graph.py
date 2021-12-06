import numpy as np

from nanotag.graph.dual import faces_to_dual_faces, faces_to_adjacency
from nanotag.graph.representation import faces_to_edges, faces_to_quad_edge
from nanotag.graph.stable_delaunay_graph import stable_delaunay_faces
from nanotag.graph.subgraph import subgraph_adjacency
from nanotag.graph.traverse import connected_components
from nanotag.graph.utils import flatten_list_of_lists, check_clockwise
from nanotag.graph.edges import connect_edges
from nanotag.graph.faces import outer_faces_from_faces
from nanotag.graph.geometry import polygon_area
from collections import defaultdict
import matplotlib.pyplot as plt
from nanotag.graph.visualize import add_edges_to_mpl_plot

class PlanarGraph:

    def __init__(self, faces, points, labels=None):
        self._faces = faces
        self._points = points
        self._labels = labels

    def __len__(self):
        return len(self._points)

    @property
    def edge_indices(self):
        return {frozenset(edge): i for i, edge in enumerate(self.edges)}

    @property
    def face_edge_indices(self):
        edge_indices = self.edge_indices
        face_edge_indices = []
        for face in self.faces:
            face_edge_indices.append([edge_indices[frozenset((face[i], face[i - 1]))] for i in range(len(face))])
        return face_edge_indices

    @property
    def faces(self):
        return self._faces

    @property
    def points(self):
        return self._points

    @property
    def adjacency(self):
        return faces_to_adjacency(self.faces, len(self))

    @property
    def degree(self):
        return np.array([len(value) for value in self.adjacency.values()])

    @property
    def edges(self):
        return np.array(faces_to_edges(self.faces))

    @property
    def polygons(self):
        return [self.points[face] for face in self.faces]

    @property
    def quad_edge(self):
        return faces_to_quad_edge(self.faces)

    def is_clockwise(self):
        return np.array([check_clockwise(polygon) for polygon in self.polygons])

    def make_clockwise(self):
        for i, polygon in enumerate(self.polygons):
            if not check_clockwise(polygon):
                self._faces[i] = self._faces[i][::-1]

    @property
    def faces_incident_nodes(self):
        faces_incident_nodes = defaultdict(list)
        for i, face in enumerate(self.faces):
            for j in face:
                faces_incident_nodes[j].append(i)
        return faces_incident_nodes

    def subgraph_adjacency(self, nodes):
        return subgraph_adjacency(nodes, self.adjacency)

    @property
    def polygon_areas(self):
        return [polygon_area(polygon) for polygon in self.polygons]

    @property
    def outer_faces(self):
        return outer_faces_from_faces(self.faces)

    def boundary(self, indices=None):
        if indices is None:
            faces = self.faces
        else:
            faces = [self.faces[i] for i in indices]

        quad_edge = faces_to_quad_edge(faces)

        boundary = set([frozenset(edge) for edge, face in quad_edge.items() if None in face])

        outer = connect_edges([list(edge) for edge in boundary])
        return outer[np.argmax([self.points[i].max() for i in outer])]

    # def subgraph(self, face_indices, relabel=True):
    #     faces = [self.faces[i] for i in face_indices]
    #     #unique = np.unique(flatten_list_of_lists(faces))[0]
    #
    #     #mapping = {i: j for i, j in enumerate(unique)}
    #
    #     #new_points = self.points[unique]
    #     return self.__class__(faces, self.points)

    def connected_components(self, nodes):
        defect_dual_subgraph = self.subgraph_adjacency(nodes)
        return connected_components(defect_dual_subgraph)

    def dual(self):
        dual_points = np.array([polygon.mean(axis=0) for polygon in self.polygons])
        dual_faces = faces_to_dual_faces(self.faces, len(self))
        return PlanarGraph(dual_faces, dual_points)

    def show(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        add_edges_to_mpl_plot(self.points, edges=self.edges, ax=ax)


class StableDelaunayGraph(PlanarGraph):

    def __init__(self, points, alpha):
        faces = stable_delaunay_faces(points, alpha)
        super().__init__(faces=faces, points=points)
