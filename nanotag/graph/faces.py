from nanotag.graph.edges import connect_edges
from nanotag.graph.representation import faces_to_quad_edge


def outer_faces_from_faces(faces):
    quad_edge = faces_to_quad_edge(faces)
    edges = [list(edge) for edge, faces in quad_edge.items() if faces[0] is None]
    return connect_edges(edges)
