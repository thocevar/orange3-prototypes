import Orange
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui
from Orange.data import Table
from Orange.distance import Euclidean

import orangecontrib.network as network

class OWPseudoTime(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "PseudoTime"
    description = "Build a minimum spanning tree to determine pseudotime."
    icon = "icons/PseudoTime.svg"
    want_main_area = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        network = Output("Network", network.Graph)
        data = Output("Data", Table)

    def __init__(self):
        super().__init__()
        self.lab = gui.widgetLabel(self.controlArea, "Label.")

    @Inputs.data
    def set_data(self, data):
        self.data = data

        data, nx_graph = MST.Prim(data, Euclidean())

        self.Outputs.data.send(data)
        graph = network.Graph(nx_graph)
        self.Outputs.network.send(graph)


class MST:
    @staticmethod
    def distances(adj, x, d, p=None):
        for y in adj[x]:
            if d[y] is not None: continue
            d[y] = d[x] + 1
            if p is not None: p[y] = x
            MST.distances(adj, y, d, p)

    @staticmethod
    def diameter(adj):
        n = len(adj)
        diam, dx, dy = 0, 0, 0
        for x in range(n):
            d = [None if i!=x else 0 for i in range(n)]
            MST.distances(adj, x, d)
            mv, mi = max(zip(d, range(n)))
            if mv > diam:
                diam, dx, dy = mv, x, mi
        return dx, dy

    @staticmethod
    def backbone(adj, x, y, t):
        n = len(adj)
        d = [None if i != x else 0 for i in range(n)]
        p = [None for i in range(n)]
        MST.distances(adj, x, d, p)
        while y!=x:
            t[y] = d[y]
            y = p[y]
        t[x] = 0

    @staticmethod
    def branches(adj, t):
        n = len(adj)
        for x in range(n):
            if t[x] is None: continue
            d = t[:]
            MST.distances(adj, x, d)
            for y in range(n):
                if t[y] is None and d[y] is not None:
                    t[y] = t[x]

    @staticmethod
    def pseudo_time(n, edges):
        adj = [[] for i in range(n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        dx, dy = MST.diameter(adj)
        t = [None for i in range(n)]
        MST.backbone(adj, dx, dy, t)
        MST.branches(adj, t)
        return t

    @staticmethod
    def Prim(data, dist):
        dist = dist.fit(data)
        n = len(data)
        distance_to_tree = [float("inf") for i in range(n)]
        neighbour_in_tree = [None for i in range(n)]
        added = [False for i in range(n)]
        edges = []
        for r in range(n):  # repeat n times (add a new node each time)
            # find the node closest to the tree
            a = -1
            for i in range(n):
                if added[i]: continue
                if a==-1 or distance_to_tree[i] < distance_to_tree[a]:
                    a = i
            # add node a to the tree
            if neighbour_in_tree[a] is not None:
                edges.append((a, neighbour_in_tree[a]))
            added[a] = True
            # update distances of nodes to tree
            if isinstance(dist, list):
                distance_a = dist[a]  # distance matrix
            else:
                distance_a = dist(data[a], data)[0]  # distance function
            for i in range(n):
                if added[i]: continue
                if distance_a[i] < distance_to_tree[i]:
                    distance_to_tree[i] = distance_a[i]
                    neighbour_in_tree[i] = a

        times = MST.pseudo_time(n, edges)
        ptime_var = Orange.data.ContinuousVariable("Pseudo time")
        new_domain = Orange.data.Domain(
            data.domain.attributes,
            data.domain.class_vars,
            data.domain.metas + (ptime_var,))
        new_table = data.transform(new_domain)
        new_table.get_column_view(ptime_var)[0][:] = times

        tree = network.Graph()
        tree.add_nodes_from(range(n))
        tree.add_edges_from(edges)
        tree.set_items(new_table)

        return new_table, tree



if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication
    import Orange

    a = QApplication(sys.argv)
    ow = OWPseudoTime()
    d = Orange.data.Table('iris')
    ow.set_data(d)
    ow.show()
    a.exec_()