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
    Class to configure the robot precise arm arm.
    There's a point for each joint and one for each of the arm's ends.
    The starting point is set at (0,0,0) and does not change.
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

    def __init__(self, parent):
        """
        Initialize a configuration of a robot arm with [dof] degrees of freedom.
        """
        self.parent = parent
        self.n_links = 2
        self.dof = 6
        self.link_lengths = np.array([2 for _ in range(dof)])
        self.yaws = np.array([0. for _ in range(self.n_links)])
        self.pitches = np.array([0. for _ in range(dof)])
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


class Tree:
    """ Rapidly exploring rapid tree data structure. """
    
    def __init__(self, q_init):
        """ q_init - root, d-dimensional numpy array """
        self.sm = SpatialMap(dim=q_init.size)
        self.root = RRTNode(None, None, q_init)
        self.sm.add(q_init, self.root)
        
    def add_node(self, q_new):
        node = RRTNode(None, None, q_new)
        return self.sm.add(q_new, node)

    def add_edge(self, q_near, q_new, u):
        new_node = self.sm.get_value(q_new)
        near_node = self.sm.get_value(q_near)
        assert np.array_equal(new_node.x, q_new)
        assert np.array_equal(near_node.x, q_near)
        new_node.parent = near_node
        new_node.u = u

    def num_nodes(self):
        return self.sm.num_values

    def __iter__(self):
        return self.sm.__iter__()


class RRTStraightLines(object):
    """RRT problem class for searching in straight line steps directly
        toward intermediate target locations.  All searching happens
        within the unit square.

    """

    def __init__(self, step_size=.03):
        self.step_size = step_size

    def random_state(self):
        return np.random.random((2,))

    def select_input(self, q_rand, q_near):
        u = q_rand - q_near
        length = np.linalg.norm(u)
        if length > self.step_size:
            u = u / length * self.step_size
        return u

    def new_state(self, q_near, u):
        return q_near + u


def rrt(problem, q_init, q_goal, tolerance, max_tree_size=float('inf')):
    """ Path search using RRT.

    Args:
        problem:   a problem instance that provides three methods:

            problem.random_state() -
               Return a randomly generated configuration
            problem.select_input(q_rand, q_near) -
               Return an action that would move the robot from
               q_near in the direction of q_rand
            problem.new_state(q_near, u) -
               Return the state that would result from taking
               action u in state q_near

        q_init:         the initial state
        q_goal:         the goal state
        tolerance:      how close the path needs to get to the goal.
        max_tree_size:  the maxmimum number of nodes to add to the tree
    Returns:
       (path, tree) tuple

    """
    #UNFINISHED!
    pass

def draw_tree(rrt):
    """ Draw a full RRT using matplotlib. """
    import matplotlib.pyplot as plt
    for node in rrt.sm:
        if node.parent is not None:
            plt.plot([node.parent.x[0], node.x[0]],
                     [node.parent.x[1], node.x[1]],
                     'r.-')

def test_simple_rrt():
    """ Demo the rrt algorithm on a simple 2d search task. """
    import matplotlib.pyplot as plt
    x_start = np.array([.5, .5])
    x_goal = np.array([.9, .9])
    lines = RRTStraightLines(.03)
    path, tree = rrt(lines, x_start, x_goal, .03)
    print(path)
    result = np.zeros((len(path), 2))
    for i in range(len(path)):
        result[i, :] = path[i].x
    draw_tree(tree)
    plt.plot(result[:, 0], result[:, 1], '.-')
    plt.show()

def oc_rrt(V, E):
    """ Constructs an orientation-constrained rapidly exploring random tree (OC-RRT). Searches the constraint manifold for a
    feasible path by growing a space-filling tree T.

    Args:
        V: Set of nodes. Initially contains the start node qinit.
        E: Set of edges. Initially is empty.

    Returns:
        Tuple (V, E)

    """

    # For i in n, where n is the degrees of freedom (?):
        # Generate qrand with oc_sample_config()
        # Select the nearest node (parent node) qnearest from all the nodes on T
        # Attempt to generate the new node (child node) qnew by moving a step size e from qnearest towards qrand.
        # If this path is collision-free, then qnew is added is added to V, and (qnearest,qnew) is added to E.

def random_angle():
    """ Returns a random angle in radians between 0 and 2pi"""
    return 2 * math.pi * random.random()


def valid_configuration(a1, a2, a3, a4, a5, a6):
    """ Returns true if the given angle configuration is a valid one.

    Args:
        a1 the first dof angle value in radians
        a2 the second dof angle value in radians
        a3 the third dof angle value in radians
        a4 the fourth dof angle value in radians
        a5 the fifth dof angle value in radians
        a6 the sixth dof angle value in radians

    Returns:
        true if the given angle configuration is not a self-collision
        false if the given angle configuration is a self-collision

    """
    #  for a given a1, an a2 is always valid. However, the a3 is not necessarily valid:
    #  use spherical coordinates for validity
    if self.l1 * math.cos(math.radians(a2)) < 0:
        return False, None
    if self.l2 * math.cos(math.radians(a3)) + self.l1 * math.cos(math.radians(a2)) < 0:
        return False, None
    return True, [(a1 + 360) % self.j1, (a2 + 360) % self.j2, \
           (a3 + 360) % self.j3, (a4 + 360) % self.j4, \
           (a5 + 360) % self.j5, (a6 + 360) % self.j6]


def oc_sample_config(angles):
    """ Generates a random node uniformly on the constraint manifold.
    Args:
        angles: The euler constraint angles.
    """
    while(True):
        qrand = [random_angle() for i in range(6)]
        if valid_configuration(qrand[0], qrand[1], qrand[2], qrand[3], qrand[4], qrand[5]):
            return 
    # Loops until IK is solved successfully (qrand meets the restrictive requirements of joint angles range and avoiding
    # singularity)
        # Get a random state by repeatedly sampling new random states of the main arm angles
        # Tries to calculate wrist angles for given Euler angles
    # Return a tuple containing these two values

def oc_steer(qnear, qrand, angles):
    """ Attempts to generate the new node (child node) qnew by moving a step size e from qnearest toward qrand.

    Args:
        qnear: The nearest node.
        qrand: The node randomly generated on the constraint manifold by oc_sample_config().
        angles: The euler angles.

    """
    # Loops until IK is solved successfully (qnew meets the restrictive requirements of joint angles range and avoiding
    # singularity)
        # Generates a new node by moving a small amount in the straight line from the nearest node qnearest to the
            # randomly sampled node qrand
        # Tries to calculate wrist angles for given Euler angles
    # Return a tuple containing these two values




if __name__=="__main__":
    test_simple_rrt()

