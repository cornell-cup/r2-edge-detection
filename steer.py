import math
import numpy as np
epsilon = 0.2


def oc_steer(qnearest, qrand):
    """ Attempts to generate the new node (child node) qnew by moving a step size e from qnearest toward qrand.

    Args:
        qnearest: The nearest node.
        qrand: The node randomly generated on the constraint manifold by oc_sample_config().

    """
    # Loops until IK is solved successfully (qnew meets the restrictive requirements of joint angles range and avoiding
    # singularity)
    # Generates a new node by moving a small amount in the straight line from the nearest node qnearest to the
    # randomly sampled node qrand
    dist = math.dist(qrand, qnearest)
    new = np.subtract(qrand, qnearest)
    new = new * (epsilon / dist)
    # Tries to calculate wrist angles for given Euler angles
    # Return a tuple containing these two values
    return new


new = oc_steer([0,1,1], [0,2,2])

print(new)
print(np.linalg.norm(new))
