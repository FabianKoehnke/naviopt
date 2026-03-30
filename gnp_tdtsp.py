import helperfunctions as hf 
import numpy as np

# 1. Genrate random T table 
T= hf.generate_route_delay_table(
        N=100, 
        num_edges=6, 
        max_base_delay=0,
        accident_prob=0.03,
        min_accident_duration=2, 
        max_accident_duration=5, 
        min_accident_impact=20, 
        max_accident_impact=40
        )

print(np.round(T,2))

# 2. Read distance matrix



# 3. Running GNP 
