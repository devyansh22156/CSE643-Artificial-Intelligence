import heapq
import math
import time
import tracemalloc
from collections import deque

import matplotlib
import numpy as np
import pickle
import matplotlib.pyplot as plt


# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def dfs(adj_list, node, goal_node, depth, visited):
    if node == goal_node:
        return [node]

    if depth == 0:
        return None

    visited.add(node)

    for adjNode, cost in adj_list.get(node, []):
        if adjNode not in visited:
            result = dfs(adj_list, adjNode, goal_node, depth - 1, visited)
            if result is not None:
                return [node] + result

    visited.remove(node)  # on backtracking unmarked the node as visited
    return None


def ids(adj_list, start_node, goal_node, max_depth):
    for d in range(max_depth + 1):
        visited = set()
        result = dfs(adj_list, start_node, goal_node, d, visited)
        if result is not None:
            return result
    return None


def dfsCheck(adj_list, start_node, goal_node):
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node == goal_node:
            return True
        visited.add(node)
        for adjNode, cost in adj_list.get(node, []):
            if adjNode not in visited:
                stack.append(adjNode)

    return False


def get_ids_path(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)
    m = len(adj_matrix[0])
    adj_list = {}

    for i in range(n):
        adj_list[i] = []
        for j in range(m):
            if adj_matrix[i][j] > 0:
                adj_list[i].append((j, adj_matrix[i][j]))

    if start_node == goal_node:
        return [start_node]

    if not dfsCheck(adj_list, start_node, goal_node):  # checks if path exists else returns null
        return []

    max_depth = n - 1  # n = number of total nodes present in the graph
    path = ids(adj_list, start_node, goal_node, max_depth)

    if path is not None:
        return path

    return []


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def bfs(adj_list, queue, visited, tempVisited, parents):
    # tempVisited is the visited array for goal if queue is of start
    while (queue):
        node = queue.popleft()
        for adjNode, cost in adj_list.get(node, []):
            if adjNode not in visited:
                visited.add(adjNode)
                parents[adjNode] = node
                queue.append(adjNode)
                if adjNode in tempVisited:
                    # parents[adjNode] = node
                    return adjNode
    return None


def getPath(startParents, goalParents, temp):
    def loop(parents, node):
        path = []
        while node is not None:
            path.append(node)
            node = parents[node]
        return path

    path1 = loop(startParents, temp)
    path1.reverse()
    path2 = loop(goalParents, goalParents[temp])

    return path1 + path2


def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)
    m = len(adj_matrix[0])

    adj_list = {}
    for i in range(n):
        adj_list[i] = []
        for j in range(m):
            if adj_matrix[i][j] > 0:
                adj_list[i].append((j, adj_matrix[i][j]))

    if start_node == goal_node:
        return [start_node]

    startQueue = deque([start_node])
    goalQueue = deque([goal_node])

    startVisited = {start_node}
    goalVisited = {goal_node}

    startParents = {start_node: None}
    goalParents = {goal_node: None}

    while startQueue and goalQueue:
        meeting_node = bfs(adj_list, startQueue, startVisited, goalVisited, startParents)
        if meeting_node is not None:
            return getPath(startParents, goalParents, meeting_node)

        meeting_node = bfs(adj_list, goalQueue, goalVisited, startVisited, goalParents)
        if meeting_node is not None:
            return getPath(startParents, goalParents, meeting_node)

    return []  # path not found


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def getDist(u, v, node_attributes):
    x1 = node_attributes[u]['x']
    y1 = node_attributes[u]['y']
    x2 = node_attributes[v]['x']
    y2 = node_attributes[v]['y']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def getPath2(parent, goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)
    m = len(adj_matrix[0])

    adj_list = {}
    for i in range(n):
        adj_list[i] = []
        for j in range(m):
            if adj_matrix[i][j] > 0:
                adj_list[i].append((j, adj_matrix[i][j]))

    pq = []

    gn = {i: float('inf') for i in range(n)}
    gn[start_node] = 0
    parents = {start_node: None}

    fn_start = getDist(start_node, goal_node, node_attributes)
    heapq.heappush(pq, (fn_start, start_node))

    while pq:
        node_fn, node = heapq.heappop(pq)

        if node == goal_node:
            return getPath2(parents, goal_node)

        for adjNode, cost in adj_list.get(node, []):
            if gn[node] + cost < gn[adjNode]:
                gn[adjNode] = gn[node] + cost
                parents[adjNode] = node
                fn = gn[node] + cost + getDist(adjNode, goal_node, node_attributes)
                heapq.heappush(pq, (fn, adjNode))

    return []


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def bfs2(adj_list, heapq, node, end_node, pq, gn, parents, visited, node_attributes):
    for adjNode, cost in adj_list.get(node, []):
        new_gn = gn[node] + cost
        if new_gn < gn[adjNode]:
            gn[adjNode] = new_gn
            parents[adjNode] = node
            fn = new_gn + getDist(adjNode, end_node, node_attributes)
            heapq.heappush(pq, (fn, adjNode))
            visited.add(adjNode)


def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)
    m = len(adj_matrix[0])

    adj_list = {}
    for i in range(n):
        adj_list[i] = []
        for j in range(m):
            if adj_matrix[i][j] > 0:
                adj_list[i].append((j, adj_matrix[i][j]))

    if start_node == goal_node:
        return [start_node]

    startPQ = []
    goalPQ = []

    gn_start = {i: float('inf') for i in range(n)}
    gn_goal = {i: float('inf') for i in range(n)}
    gn_start[start_node] = 0
    gn_goal[goal_node] = 0

    fn_start = getDist(start_node, goal_node, node_attributes)
    fn_goal = getDist(goal_node, start_node, node_attributes)

    startParents = {start_node: None}
    goalParents = {goal_node: None}

    heapq.heappush(startPQ, (fn_start, start_node))
    heapq.heappush(goalPQ, (fn_goal, goal_node))

    startVisited = {start_node}
    goalVisited = {goal_node}

    while startPQ and goalPQ:

        minHeuristic, node = heapq.heappop(startPQ)
        if node in goalVisited:
            return getPath(startParents, goalParents, node)

        bfs2(adj_list, heapq, node, goal_node, startPQ, gn_start, startParents, startVisited, node_attributes)

        minHeuristic, node = heapq.heappop(goalPQ)
        if node in startVisited:
            return getPath(startParents, goalParents, node)

        bfs2(adj_list, heapq, node, start_node, goalPQ, gn_goal, goalParents, goalVisited, node_attributes)

    return []


# Bonus Problem

# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def getPairs(adj_list, n):
    pairNodes = []
    visited = set()
    parents = {}
    discoveryTime = {}
    lowTime = {}
    timer = [0]

    for start_node in range(n):
        if start_node not in visited:
            stack = [(start_node, None)]
            while stack:
                node, parent = stack[-1]

                if node not in visited:
                    visited.add(node)
                    discoveryTime[node] = lowTime[node] = timer[0]
                    timer[0] += 1
                    parents[node] = parent

                    for v, weight in adj_list[node]:
                        if v not in visited:
                            stack.append((v, node))
                        elif v != parents.get(node):
                            lowTime[node] = min(lowTime[node], discoveryTime[v])
                else:
                    stack.pop()
                    if parent is not None:
                        lowTime[parent] = min(lowTime[parent], lowTime[node])
                        if lowTime[node] > discoveryTime[parent]:
                            pairNodes.append((parent, node))

    pairNodes.sort()
    return pairNodes


def bonus_problem(adj_matrix):
    n = len(adj_matrix)
    m = len(adj_matrix[0])
    adj_list = {}

    for i in range(n):
        adj_list[i] = []
        for j in range(m):
            if adj_matrix[i][j] > 0:
                adj_list[i].append((j, adj_matrix[i][j]))

    return getPairs(adj_list, n)


# -----------------------------------------------------------------------------------------------------------------------


def getTimeMemory(func, *args):
    tracemalloc.start()
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_time, peak

def compare_searches(adj_matrix):
    n = len(adj_matrix)
    results = []
    idsTimeExecution = 0
    idsMemorySpace = 0
    bibfsTimeExecution = 0
    bibfsMemorySpace = 0
    astarTimeExecution = 0
    astarMemorySpace = 0
    bihsTimeExecution = 0
    bihsMemorySpace = 0

    for source in range(n):
        for end in range(n):
            if source != end:
                print(f"\nFor {source} To {end}")
                path1, time, memory = getTimeMemory(get_ids_path, adj_matrix, source, end)
                idsTimeExecution += time
                idsMemorySpace += memory
                print(f"IDS Path: {path1}, Time: {time}s, Memory: {memory}bytes")
                results.append({
                    'startNode': source,
                    'goalNode': end,
                    'idsPath': path1,
                    'idsTime': time,
                    'idsMemory': memory,
                })

                path2, time, memory = getTimeMemory(get_bidirectional_search_path, adj_matrix, source, end)
                bibfsTimeExecution += time
                bibfsMemorySpace += memory
                print(f"Bidirectional Search Path: {path2}, Time: {time}s, Memory: {memory}bytes")
                results.append({
                    'startNode': source,
                    'goalNode': end,
                    'bidirectionalPath': path2,
                    'bidirectionalTime': time,
                    'bidirectionalMemory': memory,
                })

                path3, time, memory = getTimeMemory(get_astar_search_path, adj_matrix, node_attributes, source, end)
                astarTimeExecution += time
                astarMemorySpace += memory
                print(f"A* Search Path: {path3}, Time: {time}s, Memory: {memory}bytes")
                results.append({
                    'startNode': source,
                    'goalNode': end,
                    'astarPath': path3,
                    'astarTime': time,
                    'astarMemory': memory,
                })

                path4, time, memory = getTimeMemory(get_bidirectional_heuristic_search_path, adj_matrix, node_attributes, source, end)
                bihsTimeExecution += time
                bihsMemorySpace += memory
                print(f"Bidirectional Heuristic Search Path: {path4}, Time: {time}s, Memory: {memory}bytes ")
                results.append({
                    'startNode': source,
                    'goalNode': end,
                    'bidirectional_heuristicPath': path4,
                    'bidirectional_heuristicTime': time,
                    'bidirectional_heuristicMemory': memory,
                })

    print("\nComparison Results for all the node pairs")
    print(f"Total Execution Time for IDS: {idsTimeExecution:.4f}s")
    print(f"Total memory usage for IDS: {idsTimeExecution/1024:.2f}KB")
    print(f"Total Execution Time for Bidirectional BFS: {bibfsTimeExecution:.4f}s")
    print(f"Total memory usage for Bidirectional BFS: {bibfsMemorySpace/1024:.2f}KB")
    print(f"Total Execution Time for A*: {astarTimeExecution:.4f}s")
    print(f"Total memory usage for A*: {astarMemorySpace/1024:.2f}KB")
    print(f"Total Execution Time for BiHS: {bihsTimeExecution:.4f}s")
    print(f"Total memory usage for BiHS: {bihsTimeExecution/1024:.2f}KB")
    plotGraphs(results)

def getPathCost(path, adj_matrix):
    if not path or len(path)<2:
        return 0
    cost = 0
    for i in range(len(path)-1):
        cost += adj_matrix[path[i]][path[i + 1]]
    return cost

def plotGraphs(results):
    algorithms = ['ids', 'bidirectional', 'astar', 'bidirectional_heuristic']
    colors = {'ids': 'r', 'bidirectional': 'g', 'astar': 'b', 'bidirectional_heuristic': 'm'}
    labels = {'ids': 'IDS', 'bidirectional': 'Bidirectional', 'astar': 'A*', 'bidirectional_heuristic': 'Bidirectional Heuristic'}

    runTimes = {alg: [] for alg in algorithms}
    memoryCosts = {alg: [] for alg in algorithms}
    pathCosts = {alg: [] for alg in algorithms}

    for result in results:
        for alg in algorithms:
            runTimes[alg].append(result.get(f'{alg}Time', None))
            memoryCosts[alg].append(result.get(f'{alg}Memory', None))
            pathCosts[alg].append(getPathCost(result.get(f'{alg}Path', None), adj_matrix))

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    for alg in algorithms:
        plt.scatter(memoryCosts[alg], runTimes[alg], color=colors[alg], label=labels[alg])
    plt.xlabel('Memory Usage (KB)')
    plt.ylabel('Execution Time (s)')
    plt.title('Time vs Memory Usage')
    plt.legend()

    plt.subplot(1, 2, 2)
    for alg in algorithms:
        plt.scatter(pathCosts[alg], runTimes[alg], color=colors[alg], label=labels[alg])
    plt.xlabel('Path Cost')
    plt.ylabel('Execution Time (s)')
    plt.title('Cost vs Execution Time')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    adj_matrix = np.load('IIIT_Delhi.npy')
    with open('IIIT_Delhi.pkl', 'rb') as f:
        node_attributes = pickle.load(f)
    compare_searches(adj_matrix)