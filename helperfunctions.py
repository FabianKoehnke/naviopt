import numpy as np

def generate_route_delay_table(N, num_edges=6, max_base_delay=3, accident_prob=0.02, min_accident_duration=2, max_accident_duration=5, min_accident_impact=20, max_accident_impact=40):
    """
    Generates a table T with smooth background traffic and sudden accidents.
    
    Parameters:
    - N: Number of time windows.
    - num_edges: Number of edges.
    - max_base_delay: Normal peak traffic delay.
    - accident_prob: Chance (0 to 1) of an accident starting in any time window/edge.
    """
    
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
