"""Pure python implementation of the Rapidly Exploring Random Tree
algorithm

author: Nathan Sprague and ...

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import random
import math


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
    link_lengths [np array] : lengths of each link between joints.
                              len(link_lengths) == n_links.
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
        self.yaws = np.array([0. for _ in range(self.n_links)])
        self.pitches = np.array([0. for _ in range(self.dof)])
        # may need angles for end effector, but by OC-RRT may not be necessary
        self.points = np.array([[0., 0., 0.] for _ in range(self.n_links + 1)])
        self.update_points()

    def update_yaw(self, yaws):
        """
        Redefine the yaw angles.
        KEYWORD ARGUMENTS:
        yaws [np array] : new yaw angles for each joint.
        PRECONDITIONS:
        len(yaws) == self.n_links.
        """
        self.yaws = np.array(yaws)
        self.update_points()

    def update_pitch(self, pitches):
        """
        Redefine the pitch angles.
        KEYWORD ARGUMENTS:
        pitches [np array] : new pitch angles for each joint.
        PRECONDITIONS:
        len(pitches) == self.n_links.
        """
        self.pitches = np.array(pitches)
        self.update_points()

    def update_points(self):
        """
        Redefine the points according to yaw and pitch angles.
        """
        for i in range(1, self.n_links + 1):
            yaw = np.sum(self.yaws[i-1])
            pitch = np.sum(self.pitches[i-1])
            r = self.link_lengths[i - 1]
            hyp = r * np.sin(pitch)  # projection of vector on the xy plane
            self.points[i][0] = self.points[i - 1][0] + hyp * np.cos(yaw)
            self.points[i][1] = self.points[i - 1][1] + hyp * np.sin(yaw)
            self.points[i][2] = self.points[i - 1][2] + np.cos(pitch)

    def get_points(self):
        """
        Return the coordinates of the arm's joints and ends.
        RETURNS: np array
        """
        return self.points

    def get_yaws(self):
        """
        Return the arm's yaw angles.
        RETURNS: np array
        """
        return self.yaws

    def get_pitches(self):
        """
        Return the arm's pitch angles.
        RETURNS: np array
        """
        return self.pitches

    def get_dof(self):
        """
        Return the degrees of freedom.
        RETURNS: int
        """
        return self.n_links

    def distance_to(self, node2):
        return math.dist(self.points[self.n_links], node2.points[self.n_links])


class Tree:
    """ Rapidly exploring rapid tree data structure. """
    
    def __init__(self, q_init, link_lengths):
        """ q_init - root, d-dimensional numpy array """
        self.sm = SpatialMap(dim=q_init.size)
        self.root = RRTNode(None, link_lengths)
        self.link_lengths = link_lengths
        self.sm.add(q_init, self.root)
        self.nodes = []
        self.edges = []
        self.t = (self.nodes, self.edges)

    def add_node(self, q_new):
        node = RRTNode(None, None, q_new)
        # return self.sm.add(q_new, node)
        return self.t[0].add(node)

    def add_edge(self, q_near, q_new, u):
        # new_node = self.sm.get_value(q_new)
        # near_node = self.sm.get_value(q_near)
        # assert np.array_equal(new_node.x, q_new)
        # assert np.array_equal(near_node.x, q_near)
        # new_node.parent = near_node
        # new_node.u = u
        return self.t[1].add((q_near, q_new))

    def num_nodes(self):
        return self.sm.num_values

    def __iter__(self):
        return self.sm.__iter__()


def oc_rrt(t):
    """ Constructs an orientation-constrained rapidly exploring random tree (OC-RRT). Searches the constraint manifold for a
    feasible path by growing a space-filling tree T.

    Args:
        t: Contains the set of nodes and the set of edges.

    Returns:
        Tuple (V, E)

    """

    # For i in n, where n is the degrees of freedom (?):
    # Generate qrand with oc_sample_config()
    qrand = oc_sample_config()
    # Select the nearest node (parent node) qnearest from all the nodes on Tr
    qnearest = t.nearest(qrand)
    # Attempt to generate the new node (child node) qnew by moving a step size e from qnearest towards qrand.

    # If this path is collision-free, then qnew is added is added to V, and (qnearest,qnew) is added to E.



def nearest(t, qrand):
    """ Select the nearest node (parent node) from all nodes on T.

    Args:
        t: Tuple of all
    """

def random_angle():
    """ Returns a random angle in radians between 0 and 2pi"""
    return 2 * math.pi * random.random()


def valid_configuration(a1, a2, a3, a4, a5, a6, tree):
    """ Returns true if the given angle configuration is a valid one.

    Args:
        a1 the first dof angle value in radians
        a2 the second dof angle value in radians
        a3 the third dof angle value in radians
        a4 the fourth dof angle value in radians
        a5 the fifth dof angle value in radians
        a6 the sixth dof angle value in radians
        tree the RRT data structure

    Returns:
        true if the given angle configuration is not a self-collision
        false if the given angle configuration is a self-collision

    """
    #  for a given a1, an a2 is always valid. However, the a3 is not necessarily valid:
    #  use spherical coordinates for validity
    if tree.link_lengths[0] * math.cos(math.radians(a2)) < 0:
        return False, None
    if tree.link_lengths[1] * math.cos(math.radians(a3)) + tree.link_lengths[0] * math.cos(math.radians(a2)) < 0:
        return False, None
    return True, [(a1 + 360) % 360, (a2 + 360) % 360, \
           (a3 + 360) % 360, (a4 + 360) % 360, \
           (a5 + 360) % 360, (a6 + 360) % 360]


def oc_sample_config(angles):
    """ Generates a random node uniformly on the constraint manifold.
    Args:
        angles: The euler constraint angles.
    """
    while(True):
        qrand = [random_angle() for i in range(6)]
        if valid_configuration(qrand[0], qrand[1], qrand[2], qrand[3], qrand[4], qrand[5]):
            return RRTNode([qrand[0], qrand[1], qrand[2], qrand[3], qrand[4], qrand[5]])
    # Loops until IK is solved successfully (qrand meets the restrictive requirements of joint angles range and avoiding
    # singularity)
        # Get a random state by repeatedly sampling new random states of the main arm angles
        # Tries to calculate wrist angles for given Euler angles
    # Return a tuple containing these two values


def oc_steer(qnearest, qrand, angles):
    """ Attempts to generate the new node (child node) qnew by moving a step size e from qnearest toward qrand.

    Args:
        qnearest: The nearest node.
        qrand: The node randomly generated on the constraint manifold by oc_sample_config().
        angles: The Euler angles.

    """
    # Loops until IK is solved successfully (qnew meets the restrictive requirements of joint angles range and avoiding
    # singularity)
    # Generates a new node by moving a small amount in the straight line from the nearest node qnearest to the
    # randomly sampled node qrand
    new = steer(qnearest, qrand)
    # Tries to calculate wrist angles for given Euler angles
    # Return a tuple containing these two values
    return new


def steer(qnearest, qrand):
    """ Generate the new node (child node) qnew by moving a step size e from qnearest toward qrand, without checking
    for constraints.

    Args:
        qnearest: The nearest node.
        qrand: The node randomly generated on the constraint manifold by oc_sample_config().
        angles: The Euler angles.

    """

    dist = math.dist(qrand, qnearest)
    new = np.subtract(qrand.get_points()[-1], qnearest.get_points()[-1])
    new = new * (Tree.epsilon / dist)
    
    return new


if __name__=="__main__":
    test_simple_rrt()

