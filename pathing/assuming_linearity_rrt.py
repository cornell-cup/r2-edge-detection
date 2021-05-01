"""
MIT License
Copyright (c) 2019 Fanjin Zeng
This work is licensed under the terms of the MIT license, see <https://opensource.org/licenses/MIT>.
"""
import math
from builtins import float, min, enumerate, object, len, range, list, tuple, map
import numpy as np
from random import random
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import art3d
from kinematics import FK
from rrt import valid_configuration


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
# solution: use RRT node from pathing.py?
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

    def __init__(self, configuration, link_lengths, children):
        """
        Initialize a configuration of a robot arm with [dof] degrees of freedom.
        """

        self.n_links = len(link_lengths)
        self.dof = 6
        self.link_lengths = link_lengths
        self.configuration = configuration
        self.update_points()
        self.children = children

    def get_dof(self):
        """
        Return the degrees of freedom.
        RETURNS: int
        """
        return self.n_links

    def distance_to(self, node2):
        return math.dist(self.points[self.n_links], node2.points[self.n_links])

    def add_child(self, child):
        self.children.append(child)


class Graph:
    # Link lengths
    # TODO: add fields for angles of arm
    def __init__(self, startpos, endpos):
        # Cartesian coordinates
        self.startpos = (startpos)
        self.endpos = endpos

        self.vertices = [startpos]
        self.edges = []
        self.success = False

        # self.vex2idx = {startpos: 0}
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

    def add_vex(self, pos):
        # try:
        #     idx = self.vex2idx[pos]
        # except:
        idx = len(self.vertices)
        self.vertices.append(pos)
        # self.vex2idx[pos] = idx
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


def forward_kin(angle_config):
    """ Returns the cartesian coordinates of each of the joints given the angle configuration of the arm. """
    l1 = 0.066675
    l2 = 0.104775
    l3 = 0.0889
    l4 = 0.1778
    L = np.array([l1, l2, l3, l4])

    angles = np.array([[0], [angle_config[0]], [0], [angle_config[1]], [0]])
    alpha = np.array([[angle_config[2]], [0], [0], [angle_config[3]], [0]])
    r = np.array([[0], [L[1]], [L[2]], [0], [0]])
    d = np.array([[L[0]], [0], [0], [0], [L[3]]])
    print(FK(angles, alpha, r, d))
    return FK(angles, alpha, r, d)


def random_angle_config(goal_angles, angle_range, graph):
    """ Returns a set of random angles within a range of the goal state angles. """
    rand_angles = [0, 0, 0, 0, 0, 0]

    while True:
        for i in range(0, 6):
            rand_angles[i] = (random() * 2 - 1) * angle_range + goal_angles[i]

            if valid_configuration(rand_angles[0], rand_angles[1], rand_angles[2], rand_angles[3],
                                   rand_angles[4], rand_angles[5], graph):
                return rand_angles

    return rand_angles


def steer(qrand, qnearest, epsilon):
    dist = math.dist(qrand, qnearest)
    new = np.subtract(qrand, qnearest)
    new = new * (epsilon / dist)

    return new


def RRT(startpos, endpos, obstacles, n_iter, radius, stepSize):
    # Insert kinematics to translate start position cartesian coordinates to angles of the arm

    startpos = tuple(np.concatenate((forward_kin(startpos), startpos)))
    endpos = tuple(np.concatenate((forward_kin(endpos), endpos)))

    G = Graph(startpos, endpos)

    for _ in range(n_iter):
        randangles = random_angle_config(endpos, 1, G)
        randvex = tuple(np.concatenate((forward_kin(randangles), randangles)))
        if isInObstacle(randvex, obstacles, radius):
            continue

        nearvex, nearidx = nearest(G, randvex, obstacles, radius)
        if nearvex is None:
            continue

        newvex = steer(randvex, nearvex, stepSize)

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


# TODO: Function to plot arm configurations

def plot_3d(G, obstacles, radius, path=None):
    ax = plt.axes(projection='3d')
    points = []
    for p in G.vertices:
        points.append((p[0], p[1], p[2]))
    xdata = [x for x, y, z in points]
    ydata = [y for x, y, z in points]
    zdata = [z for x, y, z in points]

    lines = [(points[edge[0]], points[edge[1]]) for edge in G.edges]
    lc = art3d.Line3DCollection(lines, colors='black', linewidths=1)
    ax.add_collection(lc)

    if path is not None:
        paths = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lc2 = art3d.Line3DCollection(paths, colors='green', linewidths=3)
        ax.add_collection(lc2)

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Blues');
    plt.show()


def pathSearch(startpos, endpos, obstacles, n_iter, radius, stepSize):
    G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)
    if G.success:
        path = dijkstra(G)
        # plot(G, obstacles, radius, path)
        return path


def norm(tup):
    result = 0
    for i in tup:
        result += i**2
    return math.sqrt(result)

def temp_valid_configuration(a1, a2, a3, a4, a5, a6):
    l1 = 0.066675
    l2 = 0.104775
    l3 = 0.0889
    l4 = 0.1778
    link_lengths = np.array([l2, l3])
    if link_lengths[0] * math.cos(math.radians(a2)) < 0:
        return False, None
    if link_lengths[1] * math.cos(math.radians(a3)) + link_lengths[0] * math.cos(math.radians(a2)) < 0:
        return False, None
    return True, [(a1 + 360) % 360, (a2 + 360) % 360, \
           (a3 + 360) % 360, (a4 + 360) % 360, \
           (a5 + 360) % 360, (a6 + 360) % 360]



if __name__ == '__main__':
    startpos = (0., 0., 0., 0., 0., 0.)
    endpos = (5., 5., 5., 5., 5., 5.)
    obstacles = [(1., 1.), (2., 2.)]
    n_iter = 200
    radius = 0.7
    stepSize = 5/180*math.pi

    # assumption that start and end are in degrees
    for i in startpos:
        assert 0 <= i <= 358.0*math.pi/180
    for j in endpos:
        assert 0 <= j <= 358.0*math.pi/180

    updates = [startpos]
    deltas = tuple(map(lambda m, n: m - n, endpos, startpos))
    deltas = [i*stepSize/norm(deltas) for i in deltas]
    print(deltas)
    while True:
        acc = True
        for i in range(len(endpos)):
            acc = acc and abs(updates[-1][i] - endpos[i]) < 0.05
        if acc:
            break
        updates.append(tuple(map(lambda m, n: m + n, updates[-1], deltas)))

    # TODO: valid configuration implementation without tree argument
    # assert valid_configuration(startpos[0], startpos[1], startpos[2], startpos[3], startpos[4], startpos[5])
    # assert valid_configuration(endpos[0], endpos[1], endpos[2], endpos[3], endpos[4], endpos[5])

    finalUpdates = updates
    for i in updates:
        if not temp_valid_configuration(i[0], i[1], i[2], i[3], i[4], i[5]):
            for k in updates[updates.index(i)+1:]:
                if temp_valid_configuration(k[0], k[1], k[2], k[3], k[4], k[5]):
                    G = RRT(i, k, obstacles, n_iter, radius, radius)
                    while not G.success and updates.index(i) > 0 and updates.index(k) < len(updates)-1:
                        i = updates[updates.index(i)-1]
                        k = updates[updates.index(i)+1]
                        G = RRT(i, k, obstacles, n_iter, radius, radius)
                    path = dijkstra(G)
                    finalUpdates = finalUpdates[0:updates.index[i]-1] + path + finalUpdates[updates.index[k]+1:]
                    
    for u in finalUpdates:
        print(u)
    # start_time = time.time()
    # # G = RRT_star(startpos, endpos, obstacles, n_iter, radius, stepSize)
    # G = RRT(startpos, endpos, obstacles, n_iter, radius, stepSize)
    #
    # if G.success:
    #     path = dijkstra(G)
    #     print(path)
    #     plot_3d(G, obstacles, radius, path)
    # else:
    #     plot_3d(G, obstacles, radius)
    # print("\nTime taken: ", (time.time()-start_time))
