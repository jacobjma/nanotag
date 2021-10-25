def connect_edges(edges):
    def add_next_to_connected_edges(connected_edges, edges):
        found_next_edge = False
        for i, edge in enumerate(edges):
            if connected_edges[-1][-1] == edge[0]:
                connected_edges[-1].append(edge[1])
                found_next_edge = True
                del edges[i]
                break

            elif connected_edges[-1][-1] == edge[1]:
                connected_edges[-1].append(edge[0])
                found_next_edge = True
                del edges[i]
                break

        if found_next_edge == False:
            connected_edges.append([edges[0][1]])
            del edges[0]

        return connected_edges, edges

    connected_edges = [[edges[0][1]]]
    del edges[0]

    while edges:
        connected_edges, edges = add_next_to_connected_edges(connected_edges, edges)

    return connected_edges
