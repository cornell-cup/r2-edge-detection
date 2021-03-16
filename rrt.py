"""Pure python implementation of the Rapidly Exploring Random Tree
algorithm

author: Nathan Sprague and ...

"""
import numpy as np
from spatial_map import SpatialMap

class RRTNode(object):
    """ RRT Tree node. """
    def __init__(self, parent, u, x):
        """ Constructor.
        Arguments:
            parent - Parent node in the tree.
            u      - control signal that moves from the parents state to x.
            x      - state associated with this node.
        """
        self.parent = parent
        self.u = u
        self.x = x

    def __repr__(self):
        """ String representation for debugging purposes. """
        return "<u: {}, x: {}>".format(self.u, self.x)


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


if __name__=="__main__":
    test_simple_rrt()
   
