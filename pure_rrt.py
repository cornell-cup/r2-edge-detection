"""
MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.
"""
import math

import numpy as np
from random import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import collections as mc
from collections import deque
from mpl_toolkits.mplot3d import art3d
from .kinematics import FK
from .rrt import valid_configuration

class Line:
    def __init__(self, p0, p1):
        self.p = np.array(p0)
        self.dirn = np.array(p1) - np.array(p0)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist  # normalize

    def path(self, t):
        return self.p + t * self.dirn


def intersection(line, center, radius):
    """ Check line-sphere (circle) intersection """
    a = np.dot(line.dirn, line.dirn)
    b = 2 * np.dot(line.dirn, line.p - center)
    c = np.dot(line.p - center, line.p - center) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a);
    t2 = (-b - np.sqrt(discriminant)) / (2 * a);

    if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
        return False

    return True


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# TODO: Modify the below methods to work with manifolds representing obstacles
def isInObstacle(vex, obstacles, radius):
    # for obs in obstacles:
        # if distance(obs, vex) < radius:
            # return True
    return False


def isThruObstacle(line, obstacles, radius):
    # for obs in obstacles:
        # if Intersection(line, obs, radius):
            # return True
    return False

# TODO: Implement a greedy search to find the approximate nearest neighbor
def nearest(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def new_vertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(stepSize, length)

    newvex = (nearvex[0] + dirn[0], nearvex[1] + dirn[1], nearvex[2] + dirn[2])
    return newvex


def window(startpos, endpos):
    """ Define seach window - 2 times of start to end rectangle"""
    width = endpos[0] - startpos[0]
    height = endpos[1] - startpos[1]
    winx = startpos[0] - (width / 2.)
    winy = startpos[1] - (height / 2.)
    return winx, winy, width, height


def is_in_window(pos, winx, winy, width, height):
    """ Restrict new vertex insides search window"""
    if winx < pos[0] < winx + width and \
            winy < pos[1] < winy + height:
        return True
    else:
        return False

# problem: a node is only represented by a list [x, y, z]
# solution: use RRT node from rrt.py?
# must change all references to a certain node in the RRT algorithm to new RRT node

# for qrand: make angle configurations random, and call forward kinematics on that
# for qnew: move all angles in step size epsilon towards qrand angles



class RRTNode(object):
    """
    Class to configure the robot precise arm.
    There's a point for each joint and one for each of the arm's ends.
    The starting point is set at (0,0,0) and does not change.
    Args:
        configuration: list of joint angles in radians. Corresponds to the 6 degrees of freedom on the arm
        a1-- the lift angle of the base
        a2-- the pan angle of the base
        a3-- the lift angle of the elbow
        a4-- the pan angle of the elbow
        a5-- the pan angle of the wrist
        a6-- how big to open the end effector
    INSTANCE ATTRIBUTES:
    n_links      [int]      : degrees of freedom.
    yaws         [np array] : yaw angles of joints w.r.t. previous joint.
                              yaw is the counterclockwise angle on the xy plane
                              a.k.a. theta in spherical coordinates.
                              len(yaws) == n_links.
    pitches      [np array] : pitch angles of joints w.r.t. previous joint.
                              pitch is the clockwise angle on the yz plane
                              a.k.a. psi in spherical coordinates.
                              len(pitches) == n_links.
    points       [list]     : coordinates of joints and arm ends.
                              len(points) == n_links + 1.
    """



    def __init__(self, configuration, link_lengths):
        """
        Initialize a configuration of a robot arm with [dof] degrees of freedom.
        """

        self.n_links = len(link_lengths)
        self.dof = 6
        self.link_lengths = link_lengths
        self.configuration = configuration
        self.update_points()


    def get_dof(self):
        """
        Return the degrees of freedom.
        RETURNS: int
        """
        return self.n_links

    def distance_to(self, node2):
        return math.dist(self.points[self.n_links], node2.points[self.n_links])

class Graph:
    # Link lengths


    # TODO: add fields for angles of arm
    def __init__(self, startpos, endpos):
        # Cartesian coordinates
        self.startpos = startpos
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {startpos: 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]
        self.sz = endpos[2] - startpos[2]

        # Specifications of arm
        l1 = 0.066675
        l2 = 0.104775
        l3 = 0.0889
        l4 = 0.1778
        link_lengths = np.array([l1, l2, l3, l4])

    def add_vex(self, pos):
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))

    def random_position(self):
        rx = random()
        ry = random()
        rz = random()

        posx = self.startpos[0] - (self.sx / 2.) + rx * self.sx * 2
        posy = self.startpos[1] - (self.sy / 2.) + ry * self.sy * 2
        posz = self.startpos[2] - (self.sz / 2.) + rz * self.sz * 2
        return posx, posy, posz


def random_angle_config():
    # How to generate random angle configurations within the configuration space?
    # How to bias the angle configurations towards the goal?
    while True:
        a1 = random() * 2 * math.pi
        a2 = random() * 2 * math.pi
        a3 = random() * 2 * math.pi
        a4 = random() * 2 * math.pi
        a5 = random() * 2 * math.pi
        a6 = random() * 2 * math.pi

        if valid_configuration(a1, a2, a3, a4, a5, a6, Graph(0,0)):
            return [a1, a2, a3, a4, a5, a6]

    return []


def steer(qrand, qnearest, epsilon):
    dist = math.dist(qrand, qnearest)
    new = np.subtract(qrand, qnearest)
    new = new * (epsilon / dist)

    return new




def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):
    # Insert kinematics to translate start position cartesian coordinates to angles of the arm

    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.random_position()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = new_vertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            G.success = True
            # print('success')
            # break
    return G


def RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = G.random_position()
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = new_vertex(randvex, nearvex, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex, nearvex)
        G.add_edge(newidx, nearidx, dist)
        G.distances[newidx] = G.distances[nearidx] + dist

        # update nearby vertices distance (if shorter)
        for vex in G.vertices:
            if vex == newvex:
                continue

            dist = distance(vex, newvex)
            if dist > radius:
                continue

            line = Line(vex, newvex)
            if isThruObstacle(line, obstacles, radius):
                continue

            idx = G.vex2idx[vex]
            if G.distances[newidx] + dist < G.distances[idx]:
                G.add_edge(idx, newidx, dist)
                G.distances[idx] = G.distances[newidx] + dist

        dist = distance(newvex, G.endpos)
        if dist < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist)
            try:
                G.distances[endidx] = min(G.distances[endidx], G.distances[newidx] + dist)
            except:
                G.distances[endidx] = G.distances[newidx] + dist

            G.success = True
            # print('success')
            break
    return G


def dijkstra(G):
    """
    Dijkstra algorithm for finding shortest path from start position to end.
    """
    srcIdx = G.vex2idx[G.startpos]
    dstIdx = G.vex2idx[G.endpos]

    # build dijkstra
    nodes = list(G.neighbors.keys())
    dist = {node: float('inf') for node in nodes}
    prev = {node: None for node in nodes}
    dist[srcIdx] = 0

    while nodes:
        curNode = min(nodes, key=lambda node: dist[node])
        nodes.remove(curNode)
        if dist[curNode] == float('inf'):
            break

        for neighbor, cost in G.neighbors[curNode]:
            newCost = dist[curNode] + cost
            if newCost < dist[neighbor]:
                dist[neighbor] = newCost
                prev[neighbor] = curNode

    # retrieve path
    path = deque()
    curNode = dstIdx
    while prev[curNode] is not None:
        path.appendleft(G.vertices[curNode])
        curNode = prev[curNode]
    path.appendleft(G.vertices[curNode])
    return list(path)


def plot(G, obstacles, radius, path=None):
    """
    Plot RRT, obstacles and shortest path
    """
    px = [x for x, y, z in G.vertices]
    py = [y for x, y, z in G.vertices]
    pz = [z for x, y, z in G.vertices]
    fig, ax = plt.subplots()

    for obs in obstacles:
        circle = plt.Circle(obs, radius, color='red')
        ax.add_artist(circle)

    ax.scatter(px, py, pz, c='cyan')
    ax.scatter(G.startpos[0], G.startpos[1], G.startpos[2], c='black')
    ax.scatter(G.endpos[0], G.endpos[1], G.startpos[2], c='black')

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = mc.LineCollection(lines, colors='green', linewidths=2)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lc2 = mc.LineCollection(paths, colors='blue', linewidths=3)
        ax.add_collection(lc2)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()


# TODO: Function to plot arm configurations

def plot_3d(G, obstacles, radius, path=None):
    ax = plt.axes(projection='3d')
    xdata = [x for x, y, z in G.vertices]
    ydata = [y for x, y, z in G.vertices]
    zdata = [z for x, y, z in G.vertices]

    f = FK([r[0], r[1], r[2]])

    lines = [(G.vertices[edge[0]], G.vertices[edge[1]]) for edge in G.edges]
    lc = art3d.Line3DCollection(lines, colors='black', linewidths=1)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lc2 = art3d.Line3DCollection(paths, colors='green', linewidths=3)
        ax.add_collection(lc2)

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues');
    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path


if __name__ == '__main__':
    startpos = (0., 0., 0.)
    endpos = (5., 5., 5.)
    obstacles = [(1., 1.), (2., 2.)]
    n_iter = 200
    radius = 0.7
    stepSize = 0.7



    # G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)

    if G.success:
        path = dijkstra(G)
        print(path)
        plot_3d(G, obstacles, radius, path)
    else:
        plot_3d(G, obstacles, radius)
