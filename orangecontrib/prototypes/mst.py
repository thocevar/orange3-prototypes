from Orange.distance import Euclidean


def Prim(data, dist=Euclidean()):
    n = len(data)
    dist = dist.fit(data)
    distance_to_tree = [float("inf") for i in range(n)]
    neighbour_in_tree = [None for i in range(n)]
    added = [False for i in range(n)]
    edges = []
    for r in range(n):  # repeat n times (add a new node each time)
        # find the node closest to the tree
        a = -1
        for i in range(n):
            if not added[i] and (a == -1 or distance_to_tree[i] < distance_to_tree[a]):
                a = i
        # add node a to the tree
        if neighbour_in_tree[a] is not None:
            edges.append((a, neighbour_in_tree[a]))
        added[a] = True
        # update distances of nodes to tree
        distance_a = dist(data[a], data)[0]  # distance function
        for i in range(n):
            if not added[i] and distance_a[i] < distance_to_tree[i]:
                distance_to_tree[i] = distance_a[i]
                neighbour_in_tree[i] = a

    return edges