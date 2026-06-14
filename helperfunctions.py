import numpy as np
from pyvis.network import Network
from itertools import permutations

def generate_route_delay_table(N, num_edges=6, max_base_delay=3, accident_prob=0.02, 
                                min_accident_duration=2, max_accident_duration=5, 
                                min_accident_impact=20, max_accident_impact=40, seed=None):
    np.random.seed(seed)

    T = np.zeros((N, num_edges))

    for e in range(num_edges):
        # --- Jede Spalte bekommt eigene zufällige Parameter ---
        col_max_base   = np.random.uniform(0.5, max_base_delay)       # unterschiedliches Basisrauschen
        col_acc_prob   = np.random.uniform(0.1, accident_prob)        # unterschiedliche Unfallhäufigkeit
        col_min_impact = np.random.uniform(min_accident_impact * 0.5, min_accident_impact)
        col_max_impact = np.random.uniform(col_min_impact, max_accident_impact)
        col_window     = np.random.randint(2, 10)                      # unterschiedliche Glättung

        # --- STEP 1: Glattes Basisrauschen pro Spalte ---
        raw_noise = np.random.uniform(0, col_max_base, size=N + col_window)
        kernel    = np.ones(col_window) / col_window
        smoothed  = np.convolve(raw_noise, kernel, mode='valid')
        T[:, e]   = smoothed[:N]

        # --- STEP 2: Unfälle pro Spalte injizieren ---
        t = 0
        while t < N:
            if np.random.random() < col_acc_prob:
                duration = np.random.randint(min_accident_duration, max_accident_duration)
                impact   = np.random.uniform(col_min_impact, col_max_impact)
                end_t    = min(t + duration, N)
                T[t:end_t, e] += impact
                t = end_t  # nach Unfall überspringen → keine überlappenden Unfälle
            else:
                t += 1

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
