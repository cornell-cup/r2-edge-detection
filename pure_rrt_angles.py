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
from kinematics import FK
from rrt import valid_configuration
import time

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


def nearest(G, vex, obstacles, radius):
    Nvex = None
    Nidx = None
    minDist = float("inf")

    for idx, v in enumerate(G.vertices):
        line = Line(v.end_effector_pos, vex)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v.end_effector_pos, vex)
        if dist < minDist:
            minDist = dist
            Nidx = idx
            Nvex = v

    return Nvex, Nidx


def explorable(contenders, min_dist, min_node, goal_node, step_size):
    if not contenders:
        print("No more contenders")
        return min_node
    explorable_nodes = []
    for node in contenders:
        new_dist = distance(node, goal_node)
        if new_dist <= min_dist + step_size/2:
            explorable_nodes.append(node)
        if new_dist <= min_dist:
            min_node = node
            min_dist = distance(node, goal_node)
    new_contenders = []
    for exp_node in explorable_nodes:
        new_contenders += exp_node.children
    return explorable(new_contenders, min_dist, min_node, goal_node, step_size)


def nearest_neighbor(G, vex, obstacles, radius):
    root = G.startpos
    return explorable(G.children, float("inf"), None, vex, radius)


def new_vertex(randvex, nearvex, stepSize):
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(stepSize, length)

    new_angles = (nearvex[0] + dirn[0], nearvex[1] + dirn[1], nearvex[2] + dirn[2], nearvex[3] + dirn[3],
              nearvex[4] + dirn[4], nearvex[5] + dirn[5])
    return RRTNode(new_angles)


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
# Specifications of arm
l1 = 0.066675
l2 = 0.104775
l3 = 0.0889
l4 = 0.1778
link_lengths = np.array([l1, l2, l3, l4])

class RRTNode(object):
    """
    Representation of a node generated in a RRT graph.

    Args:
        configuration: list of joint angles in radians. Corresponds to the 6 degrees of freedom on the arm.
            a1-- the lift angle of the base
            a2-- the pan angle of the base
            a3-- the lift angle of the elbow
            a4-- the pan angle of the elbow
            a5-- the pan angle of the wrist
            a6-- how big to open the end effector
        neighbors: list of nodes with an edge going to this node.

    Instance Attributes:
        end_effector_pos [np array] : [x, y, z] of the end effector.
        angles           [np array] : list of joint angles in radians. [a1 ... a6].
        neighbors        [array]    : list of nodes with an edge going to this node.
    """

    def __init__(self, configuration):
        self.angles = configuration
        self.end_effector_pos = self.forward_kinematics()

    def forward_kinematics(self):
        angles = np.array([[0], [self.angles[0]], [0], [self.angles[1]], [0]])
        alpha = np.array([[self.angles[2]], [0], [0], [self.angles[3]], [0]])
        r = np.array([[0], [link_lengths[1]], [link_lengths[2]], [0], [0]])
        d = np.array([[link_lengths[0]], [0], [0], [0], [link_lengths[3]]])
        return FK(angles, alpha, r, d)


class Graph:
    def __init__(self, start_angles, end_angles):
        self.startpos = RRTNode(start_angles)
        self.endpos = RRTNode(end_angles)

        self.vertices = [self.startpos]
        self.edges = []
        self.success = False

        self.vex2idx = {self.startpos: 0}
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
        self.link_lengths = np.array([l1, l2, l3, l4])

    def add_vex(self, node):
        # make another data structure for cartesian coordinates
        try:
            idx = self.vex2idx[node]
        except:
            idx = len(self.vertices)
            self.vertices.append(node)
            self.vex2idx[node] = idx
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


def random_angle_config(goal_angles, angle_range, graph):
    """ Returns a set of random angles within a range of the goal state angles. """
    rand_angles = [0, 0, 0, 0, 0, 0]

    while True:
        for a in range(0, 6):
            rand_angles[a] = (random() * 2 - 1) * angle_range + goal_angles[a]

        if valid_configuration(rand_angles[0], rand_angles[1], rand_angles[2], rand_angles[3],
                               rand_angles[4], rand_angles[5], graph):
            return rand_angles

    return rand_angles

def six_random_angles():
    """ Returns 6 random numbers in the range [0, 2*pi] """
    return random() * 2 * math.pi, random() * 2 * math.pi, random() * 2 * math.pi, random() * 2 * math.pi,\
           random() * 2 * math.pi, random() * 2 * math.pi

def steer(qrand, qnearest, epsilon):
    dist = math.dist(qrand, qnearest)
    new = np.subtract(qrand, qnearest)
    new = new * (epsilon / dist)

    return new


def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):

    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randvex = RRTNode(random_angle_config(endpos, np.pi, G))
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex.end_effector_pos, obstacles, radius)
        if nearvex is None:
            continue

        newvex = new_vertex(randvex.angles, nearvex.angles, stepSize)

        newidx = G.add_vex(newvex)
        dist = distance(newvex.end_effector_pos, nearvex.end_effector_pos)
        G.add_edge(newidx, nearidx, dist)

        dist_to_goal = distance(newvex.end_effector_pos, G.endpos.end_effector_pos)

        if dist_to_goal < 2 * radius:
            endidx = G.add_vex(G.endpos)
            G.add_edge(newidx, endidx, dist_to_goal)
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

    print(nodes)
    print(dist)

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


def arr_to_int(arr):
    """ The int array representation of an array of arrays. """
    new_array = []
    for i in range(0, len(arr)):
        new_array.append(arr[i][0])

    return new_array

# TODO: Function to plot arm configurations
def plot_3d(G, obstacles, radius, path=None):
    print(len(path))
    ax = plt.axes(projection='3d')
    end_effector_positions = []
    for v in G.vertices:
        end_effector_positions.append(v.end_effector_pos)

    float_vertices = list(map(arr_to_int, end_effector_positions))
    intermediate_vertices = []
    for i in range(1, len(float_vertices) - 1):
        intermediate_vertices.append(float_vertices[i])

    #print(end_effector_positions[0])
    #print(end_effector_positions[-1])
    xdata = [x for x, y, z in intermediate_vertices]
    ydata = [y for x, y, z in intermediate_vertices]
    zdata = [z for x, y, z in intermediate_vertices]

    #for e in G.edges:
    #    print(e)

    lines = [(float_vertices[edge[0]], float_vertices[edge[1]]) for edge in G.edges]
    lc = art3d.Line3DCollection(lines, colors='black', linewidths=1)
    ax.add_collection(lc)

    if path is not None:
        path_vertices = []
        for i in range(0, len(path)):
            path_vertices.append(path[i].end_effector_pos)
        path_vertices = list(map(arr_to_int, path_vertices))
        paths = [(path_vertices[i], path_vertices[i + 1]) for i in range(len(path_vertices) - 1)]
        lc2 = art3d.Line3DCollection(paths, colors='green', linewidths=3)
        ax.add_collection(lc2)

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues')
    ax.scatter3D(G.startpos.end_effector_pos[0], G.startpos.end_effector_pos[1], G.startpos.end_effector_pos[2], c='black')
    ax.scatter3D(G.endpos.end_effector_pos[0], G.endpos.end_effector_pos[1], G.endpos.end_effector_pos[2], c='black')

    ax.set_xlim3d(-.3, .3)
    ax.set_ylim3d(-.3, .3)
    ax.set_zlim3d(-.3, .3)

    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path


def plot_random_points(num):
    """ Plots [num] random points from random angle configurations """
    ax = plt.axes(projection='3d')
    points = []
    for i in range(0, num):
        angles = six_random_angles()
        end_pos = RRTNode(angles).end_effector_pos
        end_pos = arr_to_int(end_pos)
        points.append(end_pos)

    xdata = [x for x, y, z in points]
    ydata = [y for x, y, z in points]
    zdata = [z for x, y, z in points]

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues');
    plt.show()

if __name__ == '__main__':

    startpos = (0., 0., 0., 0., 0., 0.)
    endpos = (2., 2., 2., 2., 2., 2.)

    obstacles = [(1., 1.), (2., 2.)]
    n_iter = 200
    radius = 0.02
    stepSize = 0.7

    start_time = time.time()
    # G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)

    if G.success:
        path = dijkstra(G)
        #print(path[0].angles)
        #print(path[1].angles)
        #print(path[2].angles)
        #print(path)

        plot_3d(G, obstacles, radius, path)
    else:
        # plot_3d(G, obstacles, radius)
        print(":(")
    print("\nTime taken: ", (time.time()-start_time))
    # plot_random_points(50)