"""
Written by Simon Kapen and Alison Duan.
Adapted from Fanjin Zeng on github, 2019.
"""

import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import art3d
from kinematics import FK, kinematics
import time
import random
import visualization
import collision_detection
import line
from rrtnode import RRTNode
from rrtgraph import Graph


def point_is_colliding(node, obstacles):
    for obs in obstacles:
        colliding_x = obs[0] <= node.end_effector_pos[0] <= obs[0] + obs[3]
        colliding_y = obs[1] <= node.end_effector_pos[1] <= obs[1] + obs[3]
        colliding_z = obs[2] <= node.end_effector_pos[2] <= obs[2] + obs[3]

        if colliding_x and colliding_y and colliding_z:
            return True

    return False


def arm_is_colliding(node, obstacles):
    arm = node.to_nlinkarm()

    for obs in obstacles:
        if collision_detection.arm_is_colliding(arm, obs):
            return True
    return False


def nearest(G, node):
    """ Finds the nearest node to [node] in cartesian space. """

    new_node = None
    new_node_index = None
    min_dist = float("inf")

    for idx, v in enumerate(G.nodes):
        dist = line.distance(v.end_effector_pos, node)
        if dist < min_dist:
            min_dist = dist
            new_node_index = idx
            new_node = v

    return new_node, new_node_index


def steer(rand_angles, near_angles, step_size):
    """ Generates a new node based on the random node and the nearest node. """

    dirn = np.array(rand_angles) - np.array(near_angles)
    length = np.linalg.norm(dirn)
    dirn = (dirn / length) * min(step_size, length)

    new_angles = (near_angles[0] + dirn[0], near_angles[1] + dirn[1], near_angles[2] + dirn[2],
                  near_angles[3] + dirn[3], near_angles[4], near_angles[5])
    return RRTNode(new_angles)


def extend_heuristic(G, rand_node, step_size, threshold, obstacles):
    """ Extends RRT T from the node closest to the end node, in the direction of [rand_node]. If the node
        being extended from has failed too many times (generates an colliding configuration or does not get closer
        to the end node), it is removed from the ranking."""
    near_node = G.ranking[0]
    new_node = steer(rand_node.angles, near_node.angles, step_size)

    if G.dist_to_end(new_node) < G.dist_to_end(near_node) and not arm_is_colliding(new_node, obstacles):
        nearest_to_new, nearest_to_new_idx = nearest(G, new_node.end_effector_pos)

        newidx = G.add_vex(new_node)
        dist = line.distance(new_node.end_effector_pos, nearest_to_new.end_effector_pos)
        G.add_edge(newidx, nearest_to_new_idx, dist)
        return new_node, G.node_to_index[new_node]

    near_node.inc_fail_count()
    near_idx = G.node_to_index[near_node]
    if near_node.fail_count > threshold:
        G.ranking.remove(near_node)

        parent = G.get_parent(near_idx)
        parent.inc_fail_count()

    return near_node, near_idx


def valid_configuration(a1, a2, a3, a4, a5, a6):
    """ Returns true if the given angle configuration is a valid one. """

    link_lengths = [visualization.r_1, visualization.r_2]
    #  for a given a1, an a2 is always valid. However, the a3 is not necessarily valid:
    #  use spherical coordinates for validity
    if link_lengths[0] * math.cos(a2) < 0:
        return False, None
    if link_lengths[1] * math.cos(a3) + link_lengths[0] * math.cos(a2) < 0:
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
            # Random number from -2pi to 2pi
            rand_angles[a] = (random.random() * 2 - 1) * 2 * np.pi

        if valid_configuration(rand_angles[0], rand_angles[1], rand_angles[2], rand_angles[3],
                               rand_angles[4], rand_angles[5]):

            return rand_angles

    return rand_angles


def rrt(start_angles, end_angles, obstacles, n_iter, radius, stepSize, threshold):

    G = Graph(start_angles, end_angles, startpos, endpos)

    for i in range(n_iter):
        rand_node = RRTNode(random_angle_config(end_angles, math.pi, i, n_iter))
        if arm_is_colliding(rand_node, obstacles):
            continue

        if i % 2 == 0 or G.ranking == []:
            nearest_node, nearest_node_index = nearest(G, rand_node.end_effector_pos)
            if nearest_node is None:
                continue

            new_node = steer(rand_node.angles, nearest_node.angles, stepSize)

            if arm_is_colliding(new_node, obstacles):
                continue

            nearest_to_new, nearest_to_new_idx = nearest(G, new_node.end_effector_pos)

            newidx = G.add_vex(new_node)
            dist = line.distance(new_node.end_effector_pos, nearest_to_new.end_effector_pos)
            G.add_edge(newidx, nearest_to_new_idx, dist)

        else:
            new_node, newidx = extend_heuristic(G, rand_node, stepSize, threshold, obstacles)
            if arm_is_colliding(new_node, obstacles):
                continue

        end_eff_dist_to_goal = line.distance(new_node.end_effector_pos, G.end_node.end_effector_pos)
        elbow_dist_to_goal = line.distance(new_node.joint_positions[0], G.end_node.joint_positions[0])

        if end_eff_dist_to_goal < 2 * radius and not G.success:
            endidx = G.add_vex(G.end_node)
            G.add_edge(newidx, endidx, end_eff_dist_to_goal)
            G.success = True
            # Comment this out to run all iterations
            break

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
        new_array.append(arr[i])

    return new_array


def plot_3d(G, path=None):
    ax = plt.axes(projection='3d')

    end_effector_positions = []
    for v in G.nodes:
        if arm_is_colliding(v, obstacles):
            if (v == G.end_node):
                print("End node is colliding")
            else:
                print("RRT-generated node is colliding")
        end_effector_positions.append(v.end_effector_pos)

    float_vertices = list(map(arr_to_int, end_effector_positions))

    plot_edges(ax, G, float_vertices)

    plot_nodes(ax, float_vertices)

    if path is not None:
        plot_path(ax, path)
        plot_arm_configs(ax, path)

    ax.scatter3D(G.start_node.end_effector_pos[0], G.start_node.end_effector_pos[1], G.start_node.end_effector_pos[2], c='black')
    ax.scatter3D(G.end_node.end_effector_pos[0], G.end_node.end_effector_pos[1], G.end_node.end_effector_pos[2], c='black')

    # obstacles
    for obs in obstacles:
        collision_detection.plot_linear_cube(ax, obs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim3d(-.3, .35)
    ax.set_ylim3d(-.3, .3)
    ax.set_zlim3d(-.3, .4)

    plt.show()


def plot_nodes(ax, float_vertices):
    intermediate_vertices = []
    for i in range(1, len(float_vertices) - 1):
        intermediate_vertices.append(float_vertices[i])

    xdata = [x for x, y, z in intermediate_vertices]
    ydata = [y for x, y, z in intermediate_vertices]
    zdata = [z for x, y, z in intermediate_vertices]

    ax.scatter3D(xdata, ydata, zdata, c=None)


def plot_edges(ax, G, float_vertices):
    lines = [(float_vertices[edge[0]], float_vertices[edge[1]]) for edge in G.edges]

    lc = art3d.Line3DCollection(lines, colors='black', linewidths=1)
    ax.add_collection(lc)


def plot_path(ax, path):
    path_vertices = []
    for i in range(0, len(path)):
        path_vertices.append(path[i].end_effector_pos)
    path_vertices = list(map(arr_to_int, path_vertices))
    paths = [(path_vertices[i], path_vertices[i + 1]) for i in range(len(path_vertices) - 1)]
    lc2 = art3d.Line3DCollection(paths, colors='green', linewidths=3)
    ax.add_collection(lc2)


def plot_arm_configs(ax, path):
    path_arm_edges = []
    for i in range(0, len(path)):
        path_arm_edges.append(([0, 0, 0], path[i].joint_positions[0]))
        path_arm_edges.append((path[i].joint_positions[0], path[i].joint_positions[1]))
    lc3 = art3d.Line3DCollection(path_arm_edges, colors='green', linewidths=1)
    ax.add_collection(lc3)


def rrt_graph_list(num_trials, startpos, endpos, n_iter, radius, step_size, threshold):
    """ Generates a list of RRT graphs. """
    graphs = []
    for i in range(0, num_trials):
        print("Trial: ", i)
        G = rrt(startpos, endpos, [], n_iter, radius, step_size, threshold)
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

    x, y, z = (.2, -.2, -.2)

    angles = [round(x[0], 5) for x in kinematics(x, y, z).tolist()]

    endpos = (math.radians(angles[1]), math.radians(angles[0]), math.radians(angles[3]), math.radians(angles[2]), 0, 0)
    # print("angle 1: ", math.radians(angles[1]))
    # print("angle 2: ", math.radians(angles[0]))
    # print("angle 3: ", math.radians(angles[3]))
    # print("angle 4: ", math.radians(angles[2]))
    # endpos = (1, 1, 0, 1, 0, 0)

    obstacles = [[0.1, -0.1, 0, 0.2]]
    #obstacles = []
    n_iter = 1000
    radius = 0.01
    stepSize = .5
    threshold = 2
    start_time = time.time()
    print("RRT started")
    G = rrt(startpos, endpos, obstacles, n_iter, radius, stepSize, threshold)

    if G.success:
        path = dijkstra(G)
        print("\nTime taken: ", (time.time() - start_time))
        plot_3d(G, path)
    else:
        print("\nTime taken: ", (time.time() - start_time))
        print("Path not found. :(")
        plot_3d(G, [])

    # trials = 50
    # graphs = rrt_graph_list(trials, startpos, endpos, n_iter, radius, stepSize, threshold)
    #
    # print("Average nodes generated: ", avg_nodes_test(graphs))
    # print("Num. successes: ", converge_test(graphs))
    # total_time = time.time() - start_time
    # print("Time taken: ", total_time)
    # print("Average time per graph: ", total_time / trials)
