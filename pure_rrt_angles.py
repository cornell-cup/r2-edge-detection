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


def nearest(G, node, obstacles, radius):
    new_node = None
    new_node_index = None
    min_dist = float("inf")

    for idx, v in enumerate(G.nodes):
        line = Line(v.end_effector_pos, node)
        if isThruObstacle(line, obstacles, radius):
            continue

        dist = distance(v.end_effector_pos, node)
        if dist < min_dist:
            min_dist = dist
            new_node_index = idx
            new_node = v

    return new_node, new_node_index


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


def nearest_neighbor(G, node, obstacles, radius):
    root = G.start_node
    return explorable(G.children, float("inf"), None, node, radius)


def steer(rand_angles, near_angles, step_size):
    """ Generates a new node based on the random node and the nearest node. """
    dirn = np.array(rand_angles) - np.array(near_angles)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(step_size, length)

    new_angles = (near_angles[0] + dirn[0], near_angles[1] + dirn[1], near_angles[2] + dirn[2],
                  near_angles[3] + dirn[3], near_angles[4] + dirn[4], near_angles[5] + dirn[5])
    return RRTNode(new_angles)


def new_vertex(randvex, nearvex, stepSize):
    """ Generates a new node based on cartesian coordinates. """
    dirn = np.array(randvex) - np.array(nearvex)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(stepSize, length)

    newvex = (nearvex[0] + dirn[0], nearvex[1] + dirn[1], nearvex[2] + dirn[2])
    return newvex


def window(start_coords, end_coords):
    """ Define seach window - 2 times of start to end rectangle"""
    width = end_coords[0] - start_coords[0]
    height = end_coords[1] - start_coords[1]
    winx = start_coords[0] - (width / 2.)
    winy = start_coords[1] - (height / 2.)
    return winx, winy, width, height


def is_in_window(pos, winx, winy, width, height):
    """ Restrict new vertex insides search window"""
    if winx < pos[0] < winx + width and \
            winy < pos[1] < winy + height:
        return True
    else:
        return False


# Specifications of arm
l1 = 0.066675
l2 = 0.104775
l3 = 0.0889
l4 = 0.1778
link_lengths = np.array([l1, l2, l3, l4])


class RRTNode(object):
    """
    A node generated in a RRT graph.

    Args:
        configuration: list of joint angles in radians. Corresponds to the 6 degrees of freedom on the arm.
            a1-- the lift angle of the base
            a2-- the pan angle of the base
            a3-- the lift angle of the elbow
            a4-- the pan angle of the elbow
            a5-- the pan angle of the wrist
            a6-- how big to open the end effector

    Instance Attributes:
        end_effector_pos [np array] : [x, y, z] of the end effector.
        angles           [np array] : list of joint angles in radians. [a1 ... a6].
    """

    def __init__(self, configuration):
        self.angles = configuration
        self.end_effector_pos = self.forward_kinematics()

    def forward_kinematics(self):
        """ Returns the [x, y, z] corresponding the node's angle configuration. """

        angles = np.array([[0], [self.angles[0]], [0], [self.angles[1]], [0]])
        alpha = np.array([[self.angles[2]], [0], [0], [self.angles[3]], [0]])
        r = np.array([[0], [link_lengths[1]], [link_lengths[2]], [0], [0]])
        d = np.array([[link_lengths[0]], [0], [0], [0], [link_lengths[3]]])
        return FK(angles, alpha, r, d)


class Graph:
    """
    An RRT graph.

    Args:
        start_angles: The initial angles of the arm.
        end_angles: The desired angles of the arm.

    Instance Attributes:
        start_node: Node containing cartesian coordinates and arm angles of the start position.
        end_node: Node containing cartesian coordinates and arm angles of the end position.

        nodes: List of all nodes in the graph.
        edges: List of all pairs (n1, n2) for which there is an edge from node n1 to node n2.
        success: True if there is a valid path from start_node to end_node.

        node_to_index: Maps nodes to indexes that are used to find the distance from start_node of each node.
        neighbors: Maps each node to its neighbors.
        distances: Maps each node to its shortest known distance from the start node.

        sx, sy, sz: The distance between the start and end nodes.
    """
    def __init__(self, start_angles, end_angles):
        self.start_node = RRTNode(start_angles)
        self.end_node = RRTNode(end_angles)

        self.nodes = [self.start_node]
        self.edges = []
        self.success = False

        self.node_to_index = {self.start_node: 0}
        self.neighbors = {0: []}
        self.distances = {0: 0.}

        self.sx = endpos[0] - startpos[0]
        self.sy = endpos[1] - startpos[1]
        self.sz = endpos[2] - startpos[2]

    def add_vex(self, node):
        # make another data structure for cartesian coordinates
        try:
            idx = self.node_to_index[node]
        except:
            idx = len(self.nodes)
            self.nodes.append(node)
            self.node_to_index[node] = idx
            self.neighbors[idx] = []
        return idx

    def add_edge(self, idx1, idx2, cost):
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append((idx2, cost))
        self.neighbors[idx2].append((idx1, cost))


def valid_configuration(a1, a2, a3, a4, a5, a6):
    """ Returns true if the given angle configuration is a valid one. """
    #  for a given a1, an a2 is always valid. However, the a3 is not necessarily valid:
    #  use spherical coordinates for validity
    if link_lengths[0] * math.cos(math.radians(a2)) < 0:
        return False, None
    if link_lengths[1] * math.cos(math.radians(a3)) + link_lengths[0] * math.cos(math.radians(a2)) < 0:
        return False, None
    return True, [(a1 + 360) % 360, (a2 + 360) % 360, \
           (a3 + 360) % 360, (a4 + 360) % 360, \
           (a5 + 360) % 360, (a6 + 360) % 360]


def random_angle_config(goal_angles, angle_range, i, amt_iter):
    """ Returns a set of random angles within a range of the goal state angles. """
    rand_angles = [0, 0, 0, 0, 0, 0]

    bias = 1 - i / (amt_iter + 1)

    while True:
        for a in range(0, 6):
            rand_angles[a] = (random() * 2 - 1) * bias * angle_range + goal_angles[a]

        if valid_configuration(rand_angles[0], rand_angles[1], rand_angles[2], rand_angles[3],
                               rand_angles[4], rand_angles[5]):
            return rand_angles

    return rand_angles


def six_random_angles():
    """ Returns 6 random numbers in the range [0, 2*pi] """
    return (random() * 2 - 1) * 2 * math.pi, (random() * 2 - 1) * 2 * math.pi, (random() * 2 - 1) * 2 * math.pi, \
           (random() * 2 - 1) * 2 * math.pi, (random() * 2 - 1) * 2 * math.pi, (random() * 2 - 1) * 2 * math.pi


def rrt(start_angles, end_angles, obstacles, n_iter, radius, stepSize):

    G = Graph(start_angles, end_angles)

    for i in range(n_iter):
        rand_node = RRTNode(random_angle_config(end_angles, math.pi, i, n_iter))
        if isInObstacle(rand_node, obstacles, radius):
            continue

        nearest_node, nearest_node_index = nearest(G, rand_node.end_effector_pos, obstacles, radius)
        if nearest_node is None:
            continue

        new_node = steer(rand_node.angles, nearest_node.angles, stepSize)

        nearest_to_new, nearest_to_new_idx = nearest(G, new_node.end_effector_pos, obstacles, radius)

        newidx = G.add_vex(new_node)
        dist = distance(new_node.end_effector_pos, nearest_to_new.end_effector_pos)
        G.add_edge(newidx, nearest_to_new_idx, dist)

        dist_to_goal = distance(new_node.end_effector_pos, G.end_node.end_effector_pos)

        if dist_to_goal < 2 * radius:
            endidx = G.add_vex(G.end_node)
            G.add_edge(newidx, endidx, dist_to_goal)
            G.success = True
            # print('success')
            # break

    return G


def dijkstra(G):
    """
    Dijkstra algorithm for finding shortest path from start position to end.
    """
    srcIdx = G.node_to_index[G.start_node]
    dstIdx = G.node_to_index[G.end_node]

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
        path.appendleft(G.nodes[curNode])
        curNode = prev[curNode]
    path.appendleft(G.nodes[curNode])
    return list(path)


def arr_to_int(arr):
    """ The int array representation of an array of arrays. """
    new_array = []
    for i in range(0, len(arr)):
        new_array.append(arr[i][0])

    return new_array


def plot_3d(G, path=None):
    print(len(path))
    ax = plt.axes(projection='3d')
    end_effector_positions = []
    for v in G.nodes:
        end_effector_positions.append(v.end_effector_pos)

    float_vertices = list(map(arr_to_int, end_effector_positions))
    intermediate_vertices = []
    for i in range(1, len(float_vertices) - 1):
        intermediate_vertices.append(float_vertices[i])

    xdata = [x for x, y, z in intermediate_vertices]
    ydata = [y for x, y, z in intermediate_vertices]
    zdata = [z for x, y, z in intermediate_vertices]

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

    ax.scatter3D(xdata, ydata, zdata, c=None)
    ax.scatter3D(G.start_node.end_effector_pos[0], G.start_node.end_effector_pos[1], G.start_node.end_effector_pos[2], c='black')
    ax.scatter3D(G.end_node.end_effector_pos[0], G.end_node.end_effector_pos[1], G.end_node.end_effector_pos[2], c='black')

    ax.set_xlim3d(-.3, .3)
    ax.set_ylim3d(-.3, .3)
    ax.set_zlim3d(-.3, .3)

    plt.show()


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


def rrt_graph_list(num_trials, startpos, endpos, n_iter, radius, step_size):
    """ Generates a list of RRT graphs. """
    graphs = []
    for _ in range(0, num_trials):
        G = rrt(startpos, endpos, [], n_iter, radius, step_size)
        graphs.append(G)

    return graphs


def avg_nodes_test(graphs):
    """ The average amount of nodes generated until the end goal is reached. """
    total_nodes = 0
    for i in range(0, len(graphs)):
        total_nodes += len(graphs[i].nodes)

    return total_nodes/len(graphs)


def converge_test(graphs):
    """
    Returns the amount of times the RRT graph converges in num_trials with n_iter iterations, radius, and step size
    step_size.
    """
    num_success = 0
    for i in range(0, len(graphs)):
        if graphs[i].success:
            num_success += 1

    return num_success


if __name__ == '__main__':

    startpos = (0., 0., 0., 0., 0., 0.)
    endpos = (2., 2., 2., 2., 2., 2.)

    obstacles = [(1., 1.), (2., 2.)]
    n_iter = 200
    radius = 0.04
    stepSize = .7

    start_time = time.time()
    G = rrt(startpos, endpos, obstacles, n_iter, radius, stepSize)

    if G.success:
        path = dijkstra(G)
        print("\nTime taken: ", (time.time() - start_time))
        plot_3d(G, path)
    else:
        print("\nTime taken: ", (time.time() - start_time))
        print("Path not found. :(")
        plot_3d(G, [])

    # graphs = rrt_graph_list(500, startpos, endpos, n_iter, radius, stepSize)
    #
    # print("Average nodes generated: ", avg_nodes_test(graphs))
    # print("Num. successes: ", converge_test(graphs))


