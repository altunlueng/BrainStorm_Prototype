# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:00:53 2018

@author: Engin
"""

#from github

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Point generation by using normal standard distribution
# 3D plotting and its projection to xy plane

z_offset = 3

# Cluster of points #1
# --------------------
coordinates_c1 = np.array([2, 1, 5])  # center point of the cluster
cluster1 = 0.15 * np.random.standard_normal((50,3)) + coordinates_c1
# Plotting 3D points
ax.plot(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2],
        'ko', alpha=0.6, label='Setosa')
# Plotting projection (notice that the Z coordinates are set to zero)
ax.plot(cluster1[:, 0], cluster1[:, 1], np.zeros_like(cluster1[:, 2])+z_offset,
        'ko')

# Cluster of points #2
# --------------------
coordinates_c2 = np.array([3.5, 2.5, 6])  # center point of the cluster
cluster2 = 0.3 * np.random.standard_normal((50,3)) + coordinates_c2
# Plotting 3D points
ax.plot(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2],
        'ro', alpha=0.6, label='Versicolor')
# Plotting projection (notice that the Z coordinates are set to zero)
ax.plot(cluster2[:, 0], cluster2[:, 1], np.zeros_like(cluster2[:, 2])+z_offset,
        'ro')

# Cluster of points #3
# --------------------
coordinates_c3 = np.array([6, 3, 7])  # center point of the cluster
cluster3 = 0.4 * np.random.standard_normal((50, 3)) + coordinates_c3
ax.plot(cluster3[:, 0], cluster3[:, 1], cluster3[:, 2],
        'go', alpha=0.6, label='Virginica')
# Plotting projection (notice that the Z coordinates are set to zero)
ax.plot(cluster3[:, 0], cluster3[:, 1], np.zeros_like(cluster3[:, 2])+z_offset,
        'go')


# Sphere surface #1
# ------------------
u1 = np.linspace(0, 2 * np.pi, 100)
v1 = np.linspace(0, np.pi, 100)


x_sphere_1 = 1   * np.outer(np.cos(u1), np.sin(v1)) + coordinates_c1[0]
y_sphere_1 = 0.5 * np.outer(np.sin(u1), np.sin(v1)) + coordinates_c1[1]
z_sphere_1 = 1.5 * np.outer(np.ones(np.size(u1)), np.cos(v1)) + coordinates_c1[2]
ax.plot_surface(x_sphere_1, y_sphere_1, z_sphere_1,
                rstride=10, cstride=10, linewidth=0.1, color='b', alpha=0.1)

# Sphere surface #2
# ------------------
u2 = np.linspace(0, 2 * np.pi, 100)
v2 = np.linspace(0, np.pi, 100)

x_sphere_2 = 1.5 * np.outer(np.cos(u2), np.sin(v2)) + coordinates_c2[0]
y_sphere_2 = 1   * np.outer(np.sin(u2), np.sin(v2)) + coordinates_c2[1]
z_sphere_2 = 1.8 * np.outer(np.ones(np.size(u2)), np.cos(v2)) + coordinates_c2[2]
ax.plot_surface(x_sphere_2, y_sphere_2, z_sphere_2,
                rstride=10, cstride=10, linewidth=0.1, color='r', alpha=0.1)

# Sphere surface #3
# -----------------
u3 = np.linspace(0, 2 * np.pi, 100)
v3 = np.linspace(0, np.pi, 100)

x_sphere_3 = 1.5 * np.outer(np.cos(u3), np.sin(v3)) + coordinates_c3[0]
y_sphere_3 = 1   * np.outer(np.sin(u3), np.sin(v3)) + coordinates_c3[1]
z_sphere_3 = 2   * np.outer(np.ones(np.size(u3)), np.cos(v3)) + coordinates_c3[2]
ax.plot_surface(x_sphere_3, y_sphere_3, z_sphere_3,
                rstride=10, cstride=10, linewidth=0.1, color='g', alpha=0.1)

# Limits of the 3D representation
ax.set_xlim3d(0, 8)
ax.set_ylim3d(0, 4)
ax.set_zlim3d(z_offset, 9)


# Labels for the legend
ax.set_xlabel(u'Longitud del pétalo (cm)')
ax.set_ylabel(u'Anchura del pétalo (cm)')
ax.set_zlabel(u'Longitud del sépalo (cm)')

# Show legend
ax.legend()

# If we want to save the figure into PDF o PNG
# fig.savefig(r'/cluster-projection.pdf')
# fig.savefig(r'/cluster-projection.png',dpi=150)

plt.show()

# with alpha shapes

import plotly.plotly as py
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/alpha_shape.csv')
df.head()

scatter = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",    
    x = df['x'], y = df['y'], z = df['z'],
    marker = dict( size=2, color="rgb(23, 190, 207)" )
)
clusters = dict(
    alphahull = 7,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",    
    x = df['x'], y = df['y'], z = df['z']
)
layout = dict(
    title = '3d point clustering',
    scene = dict(
        xaxis = dict( zeroline=False ),
        yaxis = dict( zeroline=False ),
        zaxis = dict( zeroline=False ),
    )
)
fig = dict( data=[scatter, clusters], layout=layout )
# Use py.iplot() for IPython notebook
py.iplot(fig, filename='3d point clustering')













