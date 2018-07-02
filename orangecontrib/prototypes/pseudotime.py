from math import ceil
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

import Orange
from orangecontrib.prototypes.mst import Prim


tab = Orange.data.Table("iris")
#tab = Orange.data.Table("brown-selected")

#tab = Orange.data.Table("./datasets/real/cell-cycle_leng.tab")
#tab = Orange.data.Table("./datasets/real/mesoderm-development_loh.tab")
#tab = Orange.data.Table("./datasets/real/human-embryos_petropoulos.tab")
#tab = Orange.data.Table("./datasets/real/mESC-differentiation_hayashi.tab")

#tab = Orange.data.Table("./datasets/synthetic/linear_long_2.tab")
#tab = Orange.data.Table("./datasets/synthetic/bifurcating_2.tab")
#tab = Orange.data.Table("./datasets/synthetic/consecutive_bifurcating_2.tab")
#tab = Orange.data.Table("./datasets/synthetic/trifurcating_2.tab")

# dimensionality reduction
comp = 2
pca = Orange.projection.PCA(n_components=comp)
pca_model = pca(tab)
tab = pca_model(tab)

# clustering
n = len(tab)
kmeans = Orange.clustering.KMeans(n_clusters=ceil(0.1*n))
kmeans_model = kmeans(tab)

# list of instance indices by clusters
clusters = [[] for i in range(kmeans_model.k)]
for i, c in enumerate(kmeans_model.labels_):
    clusters[c].append(i)

# minimum spanning tree
tab_centroids = Orange.data.Table.from_numpy(None, kmeans_model.centroids)
edges = Prim(tab_centroids)
adj = [[] for i in range(kmeans_model.k)]
for a,b in edges:
    adj[a].append(b)
    adj[b].append(a)
mst_coord = [[kmeans_model.centroids[a], kmeans_model.centroids[b]] for a,b in edges]

def dist(coord1, coord2):
    return sum((x1-x2)**2 for x1, x2 in zip(coord1, coord2))**0.5

# projection of points onto mst edges
proj = np.zeros((n, comp))
proj_sign = [0 for i in range(n)]
adjp = [[] for i in range(kmeans_model.k)]
for c in range(kmeans_model.k):  # for every cluster
    for i in clusters[c]:  # for each instance in cluster c
        # find nearest adjacent cluster c2 (second nearest cluster)
        c2 = min(adj[c], key=lambda c2: dist(tab[i].x, kmeans_model.centroids[c2]))
        # project instance i onto edge (c, c2)
        vi = [tab[i][d]-tab_centroids[c][d] for d in range(comp)]
        v = [tab_centroids[c2][d]-tab_centroids[c][d] for d in range(comp)]
        a = np.dot(vi,v)/np.dot(v,v)
        proj_sign[i] = 1
        if a<0:
            if len(adj[c])>1: a=0
            else: proj_sign[i] = -1
        if a>1: a=1
        proj[i] = [tab_centroids[c][d] + a*v[d] for d in range(comp)]
        adjp[c].append(i)
        adjp[c2].append(i)

# compute pseudotime of points
def add_tuple(t1, t2):
    return tuple(a+b for a,b in zip(t1,t2))

def farthest(x, p=-1, d=0):
    f = [farthest(z, x, d+dist(tab_centroids[x], tab_centroids[z])) for z in adj[x] if z!=p]
    if f: return max(f)
    else: return (d,x)

start = farthest(0)[1]
pseudotime = [None for i in range(n)]
def order(x, p=-1, d=0):
    for i in adjp[x]:
        if pseudotime[i] is not None: continue
        dxi = dist(tab_centroids[x], proj[i])
        if p==-1 and proj_sign[i]==-1:
            pseudotime[i] = d - dxi
        else:
            pseudotime[i] = d + dxi
    for z in adj[x]:
        if z==p: continue
        order(z, x, d+dist(tab_centroids[x], tab_centroids[z]))

order(start)
ptm, ptM = min(pseudotime), max(pseudotime)
pseudotime = np.array([(pt-ptm)/(ptM-ptm) for pt in pseudotime])


# VISUALIZE

# minimum spanning tree
ax = plt.axes()
ax.set_aspect('equal')
ax.set_adjustable('box')
for a, b in mst_coord:
    ax.plot([a[0], b[0]], [a[1], b[1]], c="black")

x = kmeans_model.centroids[:,0]
y = kmeans_model.centroids[:,1]
ax.scatter(x, y, s=10, c="black", marker='x')

x = np.array([row[0] for row in tab])
y = np.array([row[1] for row in tab])
l = np.array([str(row.get_class()) for row in tab])

# points
colors = list(clr.CSS4_COLORS)
random.seed(1234)
random.shuffle(colors)

labels = sorted(set(l))
for i, label in enumerate(labels):
    mask = (l == label)
    ax.scatter(x[mask], y[mask], s=5, color=colors[i], label=label)

for i in range(n):
    ax.annotate("%.2f" % pseudotime[i], (x[i],y[i]), fontsize=7)

ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", markerscale=3)
ax.autoscale()

# projection lines
for i in range(n):
    ax.plot([x[i], proj[i][0]], [y[i], proj[i][1]], linewidth=1, color='lightgray')

plt.tight_layout()
plt.show()