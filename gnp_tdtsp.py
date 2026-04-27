import helperfunctions as hf 
import numpy as np
import pandas as pd 
import fracnetics as fn
import math

# 0. Parameters
N = 100 # Table size (number of waiting times)
num_edges = 6 # Number of stations
individuals = 50 # Number of individuals in the population
judgment_nodes = 2 # Number of judgment nodes
processing_nodes = 2 # Number of processing nodes
maxWaitingIndices = 5 # maximum waiting time index of judgment nodes 
minitues_per_index = 5 # number of minutes per waiting time index
generations = 10 # number of generations to run the algorithm
worstFitness = -1000 # worst fitness value for invalid individuals

# 1. Generate random T table 
timetablesEachNode = []
for i in range(num_edges-1):

    T = hf.generate_route_delay_table(
            N=N, 
            num_edges=num_edges-1, 
            max_base_delay=1,
            accident_prob=0.03,
            min_accident_duration=minitues_per_index, 
            max_accident_duration=minitues_per_index*10,
            min_accident_impact=2, 
            max_accident_impact= minitues_per_index # waiting time cant be less than the time it takes to get to the next station
            )
    timetablesEachNode.append(T)
print(np.round(timetablesEachNode[0],2))

# 2. Read distance matrix
distances = pd.read_csv('data/distances.csv', sep=";", index_col=0).values
print(distances)

# 3. Running GNP 
pop = fn.Population(
    seed=17,
    ni=individuals, # number of individuals
    jn=judgment_nodes, # judgment nodes
    jnf=num_edges-1, # judgment node functions
    pn=processing_nodes, # processing nodes
    pnf=num_edges+1, # processing node functions (stations and wait)
    fractalJudgment=False,
    nFeatureValues=[num_edges-1 for _ in range(num_edges-1)]
)

# 3.1 Setting boundaries of nodes
minFeatures = []
maxFeatures = []
for i in range(num_edges):
    minFeatures.append(0)
    maxFeatures.append(maxWaitingIndices)

pop.setAllNodeBoundaries(minFeatures, maxFeatures)


for generation in range(generations):
    # 3.2 Evaluate fitness of individuals
    for ind in pop.individuals:
        ind.initPathTraversal()
        visited_processing_nodes = set()
        visited_processing_nodes.add(num_edges) # waiting node is considered as visited at the beginning
        currentTime = 0
        currentStation = 0 # start at station 0
        fitness = 0
        while len(visited_processing_nodes) < num_edges and currentTime < N:
            delays = timetablesEachNode[currentStation][currentTime]
            dec = ind.decisionAndNextNode(delays,dMax=2) # maximal delay is set to 1, which means that only one judgment node can be activated at a time. 
            print(f"decision: {dec}, delays: {delays}")
            if dec == num_edges: # decision is to wait 
               currentTime += 1 
               fitness += minitues_per_index
            else:
                visited_processing_nodes.add(dec)
                fitness += distances[currentStation, dec] # add distance to fitness
                fitness += delays[currentStation] #  add delay to fitness
                dist = math.ceil(distances[currentStation, dec] / minitues_per_index) # add distance to current time
                print(f"Current station: {currentStation}, Current time: {currentTime}, Decision: {dec}, Delay: {delays[currentStation]}, Distance: {dist}")
                currentTime += dist
                currentStation = dec 

            print(f"Current station: {currentStation}, Current time: {currentTime}, Fitness: {fitness}")

        if ind.invalid == True:
            ind.fitness = worstFitness
        else:
            ind.fitness = fitness * -1



    # 3.3 Selection
    pop.tournamentSelection(N=2, E=10)
    best = pop.individuals[pop.indicesElite[0]]

    # 3.4 Mutation
    pop.callEdgeMutation(
        probInnerNodes = 0.1, # probability of changing an edge of jn or pn
        probStartNode = 0.1,  # probability of changing an edge of the start node
        justUsedNodes = True
    )

    print(f"Generation: {generation}, Best fitness: {best.fitness}")
