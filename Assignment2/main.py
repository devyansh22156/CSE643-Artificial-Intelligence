# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}  # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)  # Count of trips for each stop
fare_rules = {}  # Mapping of route IDs to fare information
merged_fare_df = None  # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')


# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.

    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    for index, trip in df_trips.iterrows():
        trip_to_route[trip['trip_id']] = trip['route_id']

    # Map route_id to a list of stops in order of their sequence
    # Ensure each route only has unique stops
    df_stop_times_sorted = df_stop_times.sort_values(by=['trip_id', 'stop_sequence'])
    for trip_id, group in df_stop_times_sorted.groupby('trip_id'):
        route_id = trip_to_route[trip_id]
        stop_list = list(group['stop_id'])
        route_to_stops[route_id].extend(stop_list)
    route_to_stops = {route: list(set(stops)) for route, stops in route_to_stops.items()}

    # Count trips per stop
    for index, stop_time in df_stop_times.iterrows():
        stop_trip_count[stop_time['stop_id']] += 1

    # Create fare rules for routes
    for index, fare_rule in df_fare_rules.iterrows():
        route_id = fare_rule['route_id']
        fare_info = {
            'fare_id': fare_rule['fare_id'],
            'origin_id': fare_rule.get('origin_id'),
            'destination_id': fare_rule.get('destination_id'),
            'contains_id': fare_rule.get('contains_id')
        }
        fare_rules[route_id] = fare_info

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='left')

    return {
        'route_to_stops': route_to_stops,
        'trip_to_route': trip_to_route,
        'stop_trip_count': stop_trip_count,
        'fare_rules': fare_rules,
        'merged_fare_df': merged_fare_df
    }

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    #pass

    routeCounter = defaultdict(int)

    for tID, rID in trip_to_route.items():
        routeCounter[rID] += 1

    busiest_routes = list(routeCounter.items())
    for i in range(len(busiest_routes)):
        for j in range(len(busiest_routes)-i-1):
            if busiest_routes[j][1] < busiest_routes[j+1][1]:
                busiest_routes[j], busiest_routes[j+1] = busiest_routes[j+1], busiest_routes[j]

    ans = busiest_routes[:5]
    return ans


# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    #pass

    sol = list(stop_trip_count.items())

    for i in range(len(sol)):
        for j in range(0, len(sol) - i - 1):
            if sol[j][1] < sol[j + 1][1]:
                sol[j], sol[j + 1] = sol[j + 1], sol[j]

    ans = sol[:5]
    return ans


# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    #pass

    stop_routes = defaultdict(set)
    for ID, stops in route_to_stops.items():
        for stop_id in stops:
            stop_routes[stop_id].add(ID)

    cnt = {ID: len(routes) for ID, routes in stop_routes.items()}

    busiest_stops = list(cnt.items())

    for i in range(len(busiest_stops)):
        for j in range(0, len(busiest_stops)-i-1):
            if busiest_stops[j][1] < busiest_stops[j+1][1]:
                busiest_stops[j], busiest_stops[j+1] = busiest_stops[j+1], busiest_stops[j]

    ans = busiest_stops[:5]
    return ans


# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route.
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    #pass


    stopRoutes = defaultdict(list)

    for route_id, stops in route_to_stops.items():
        stop_pairs = list(zip(stops, stops[1:]))

        for start_stop, end_stop in stop_pairs:
            stopRoutes[(start_stop, end_stop)].append(route_id)

    L = []

    for stopPair, routes in stopRoutes.items():
        if len(routes) == 1:
            ID = routes[0]
            start_stop, end_stop = stopPair
            freq = stop_trip_count[start_stop] + stop_trip_count[end_stop]
            L.append((stopPair, ID, freq))

    for i in range(len(L)):
        length = len(L) - i - 1
        for j in range(0, length):
            if L[j][2] < L[j+1][2]:
                L[j], L[j+1] = L[j+1], L[j]

    sol = [(stop_pair, route_id) for stop_pair, route_id, route in L[:5]]
    return sol


# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df


# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    #pass
    graph = nx.Graph()
    for route, stops in route_to_stops.items():
        graph.add_edges_from(zip(stops[:-1], stops[1:]), route=route)

    loc = nx.spring_layout(graph, seed=42)
    edgeX, edgeY = [], []

    for path in graph.edges():
        a, _ = loc[path[0]]
        _, b = loc[path[0]]
        c, _ = loc[path[1]]
        _, d = loc[path[1]]
        edgeX.extend([a, b, None])
        edgeY.extend([b, d, None])

    edgeTrace = go.Scatter(x=edgeX, y=edgeY, line=dict(width=1, color='#888'), mode='lines')
    nodeX, nodeY, nodeText = [], [], []

    for node in graph.nodes():
        x, _ = loc[node]
        nodeX.append(x)
        
        _, y = loc[node]
        nodeY.append(y)
        nodeText.append(f'Stop ID: {node}')

    nodeTrace = go.Scatter(
        x=nodeX, y=nodeY, mode='markers', text=nodeText,
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, line_width=2)
    )

    fig = go.Figure(data=[edgeTrace, nodeTrace], layout=go.Layout(
        title='Bus-Stops-Graph', showlegend=False, hovermode='closest',
        margin=dict(b=0.5, l=0.3, r=0.7, t=38), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
    ))
    fig.show()



# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of r oute IDs (int) that connect the two stops directly.
    """
    #pass

    sol = []

    for ID, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            idx1, idx2 = stops.index(start_stop), stops.index(end_stop)
            if idx1 < idx2:
                sol.append(ID)
    return sol



# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')

def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    #pass


    pyDatalog.clear()
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")

    create_kb()
    add_route_data(route_to_stops)



# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    #pass

    DirectRoute(R, X, Y) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y))
    OptimalRoute(R1, R2, X, Y, Z) <= (DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (X != Z) & (Z != Y) & (R1 != R2))

    for ID, stops in route_to_stops.items():
        for stop in stops:
            +RouteHasStop(ID, stop)



# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    #pass


    query = pyDatalog.ask(f"DirectRoute(R, {start}, {end})")

    if query:
        sol = sorted({result[0] for result in query.answers})
        return sol
    else:
        return []



# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    #pass
    ans = []
    query = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include)

    for r in query:
        first = r[0]
        mid = stop_id_to_include
        sec = r[1]
        transfers = max_transfers
        ans.append((first, mid, sec))

    return list(set(ans))



# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    #pass
    ans = []
    query = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include)

    for r in query:
        first = r[0]
        mid = stop_id_to_include
        sec = r[1]
        transfers = max_transfers
        ans.append((sec, mid, first))

    return list(set(ans))


# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    pass


# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass


# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass


# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass
