import numpy as np
from pyvis.network import Network
from itertools import permutations

def generate_route_delay_table(N, num_edges=6, max_base_delay=3, accident_prob=0.02, min_accident_duration=2, max_accident_duration=5, min_accident_impact=20, max_accident_impact=40, seed=None):
    """
    Generates a table T with smooth background traffic and sudden accidents.
    
    Parameters:
    - N: Number of time windows.
    - num_edges: Number of edges.
    - max_base_delay: Normal peak traffic delay.
    - accident_prob: Chance (0 to 1) of an accident starting in any time window/edge.
    - min_accident_duration: Minimum duration (in time windows) of an accident.
    - max_accident_duration: Maximum duration (in time windows) of an accident.
    - min_accident_impact: Minimum additional delay (in minutes) caused by an accident_prob
    - max_accident_impact: Maximum additional delay (in minutes) caused by an accident_prob.
    """
    np.random.seed(seed)  
    
    # --- STEP 1: Generate Smooth Background Traffic ---
    window_size = 4
    raw_noise = np.random.uniform(0, max_base_delay, size=(N + window_size, num_edges))
    kernel = np.ones(window_size) / window_size
    
    T = np.zeros((N, num_edges))
    for i in range(num_edges):
        smoothed = np.convolve(raw_noise[:, i], kernel, mode='valid')
        T[:, i] = smoothed[:N]

    # --- STEP 2: Inject Sudden Accidents (Jumps) ---
    for e in range(num_edges):
        for t in range(N):
            # Roll the dice: Does an accident happen here?
            if np.random.random() < accident_prob:
                # Duration: how many windows the accident blocks the road (2 to 5)
                duration = np.random.randint(min_accident_duration, max_accident_duration)
                # Severity: how many extra minutes are added (the "jump")
                impact = np.random.uniform(min_accident_impact, max_accident_impact)
                
                # Apply the impact to the current and subsequent time windows
                end_t = min(t + duration, N)
                T[t:end_t, e] += impact

    return T 

def build_features(delays, visited_set, num_edges):
    visited_flags = [1.0 if i in visited_set else 0.0 for i in range(num_edges)]
    return list(delays) + visited_flags

def plotNetwork(individual, filename, justUsedNodes = False, justUsedEdges = False):
    
    net = Network(notebook=False, directed=True, cdn_resources="in_line")
    net.force_atlas_2based()
    processingFunctionNames = ["Hallein", "Golling", "Itzling", "Salzburg Altstadt", "Eugendorf", "Berchtesgaden", "Wait"]
    judgmentFunctionNames = ["T1", "T2", "T3", "T4", "T5", "T6"]
    
    # adding start node
    net.add_node(-1, label=individual.startNode.type, color = "#635b3e")
    
    # adding inner nodes
    for node in individual.innerNodes:
        if justUsedNodes == True and node.used == False:
            continue
        # set color
        if node.type == "J":
            if node.used == True:
                color = "#3e6341"
            else:
                color = "#e0ebe1"
            net.add_node(node.id, label=f"ID: {node.id} F: {judgmentFunctionNames[node.f]}", color=color)
        elif node.type == "P":
            if node.used == True:
                color = "#372f61"
            else:
                color = "#e0ddee"
            net.add_node(node.id, label=f"ID: {node.id} F: {processingFunctionNames[node.f]}", color=color)
        else:
            color = None

    # adding edges
    for node in individual.innerNodes:
        if justUsedNodes == True and node.used == False:
            continue
            
        for idx, edge in enumerate(node.edges):
            if justUsedEdges == True and individual.innerNodes[edge].used == False:
                continue
                
            if node.type == "J":
                edgeLabel = f"{node.boundaries[idx]} bis {node.boundaries[idx+1]}"
                net.add_edge(node.id, edge, title = edgeLabel,
                        font={'size': 14, 'color': '#3e6341', 'align': 'horizontal'})
            else:
                net.add_edge(node.id, edge,
                        font={'size': 14, 'color': '#372f61', 'align': 'horizontal'})
            
    # adding start node edge 
    net.add_edge(-1, individual.startNode.edges[0])
    net.save_graph(filename)

def simulate_route(route, distances=[]):
    return sum(distances[route[i], route[i+1]] for i in range(len(route) - 1))


def bruteforce_fastest_route(start=0, num_stations=0, distances=[]):
    other_stations = [s for s in range(num_stations) if s != start]
    best_cost  = float('inf')
    best_route = None

    for perm in permutations(other_stations):
        route = [start] + list(perm)
        cost  = simulate_route(route, distances)
        if cost < best_cost:
            best_cost  = cost
            best_route = route

    return best_route, best_cost
