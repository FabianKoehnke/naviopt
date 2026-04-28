import helperfunctions as hf 
import numpy as np
import pandas as pd 
import fracnetics as fn
import math

# 0. Parameters
seed=17
N = 100 # Table size (number of waiting times)
num_edges = 6 # Number of stations
individuals = 200 # Number of individuals in the population
judgment_nodes = 10 # Number of judgment nodes
processing_nodes = 10 # Number of processing nodes
maxWaitingIndices = 5 # each index in the T table corresponds to a given waiting time 
minitues_per_index = 5 # number of minutes per waiting time index
generations = 100 # number of generations to run the algorithm
worstFitness = -1000 # worst fitness value for invalid individuals

# 1. Generate random T table 
timetablesEachNode = []
for i in range(num_edges):

    T = hf.generate_route_delay_table(
            N=N, 
            num_edges=num_edges,
            max_base_delay=1,
            accident_prob=0.03,
            min_accident_duration=minitues_per_index, 
            max_accident_duration=minitues_per_index*5,
            min_accident_impact=2, 
            max_accident_impact= minitues_per_index, # waiting time cant be less than the time it takes to get to the next station
            seed=seed+1
            )
    timetablesEachNode.append(T)
print(np.round(timetablesEachNode[0],2))

# 2. Read distance matrix
distances = pd.read_csv('data/distances.csv', sep=";", index_col=0).values
print(distances)

# 3. Running GNP 
pop = fn.Population(
    seed=seed,
    ni=individuals, # number of individuals
    jn=judgment_nodes, # judgment nodes
    jnf=num_edges, # judgment node functions
    pn=processing_nodes, # processing nodes
    pnf=num_edges+1, # processing node functions (stations and wait)
    fractalJudgment=False,
    nFeatureValues=[num_edges-1 for _ in range(num_edges-1)]
)

# 3.1 Setting boundaries of nodes
minFeatures = []
maxFeatures = [0 for _ in range(num_edges)]

for table in timetablesEachNode:
    minFeatures.append(0)
    for i in range(num_edges):
        if np.max(table[:,i]) > maxFeatures[i]:
            maxFeatures[i] = np.max(table[:,i])
print("minFeatures: ", minFeatures)
print("maxFeatures: ", maxFeatures)
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
        stop = False
        decisions = []
        while len(visited_processing_nodes) <= num_edges and currentTime < N:
            delays = timetablesEachNode[0][currentTime] # TODO this should be the timetable of the current station, but for now we just use the first one for testing
            dec = ind.decisionAndNextNode(delays,dMax=2) # maximal delay is set to 1, which means that only one judgment node can be activated at a time. 
            decisions.append(dec)
            if ind.invalid == True:
                stop = True
                break
            if dec == num_edges or currentStation == dec: # decision is to wait 
               currentTime += 1 
               fitness += minitues_per_index
            else:
                visited_processing_nodes.add(dec)
                fitness += distances[currentStation, dec] # add distance to fitness
                fitness += delays[currentStation] #  add delay to fitness
                dist = math.ceil(distances[currentStation, dec] / minitues_per_index) # add distance to current time
                currentTime += dist
                currentStation = dec 


        # stop if invalid or if the last decision is not the same station the startnode is pointing to
        if stop == True:
            ind.fitness = worstFitness
            ind.fitnessValues = decisions
        else:
            ind.fitness = fitness * -1
            ind.fitnessValues = decisions



    # 3.3 Selection
    pop.tournamentSelection(N=2, E=1)
    best = pop.individuals[pop.indicesElite[0]]

    #pop.callAddDelNodes(
    #    minFeatures, 
    #    maxFeatures, 
    #    junk=0.5
    #)

    # Crossover (recombination)
    pop.crossover(
        probability = 0.05, 
        type = "uniform",
    )

    # 3.4 Mutation
    pop.callEdgeMutation(
        probInnerNodes = 0.15, # probability of changing an edge of jn or pn
        probStartNode = 0.05,  # probability of changing an edge of the start node
        justUsedNodes = True
    )

    pop.callBoundaryMutationUniform(
        probability = 0.02,
        justUsedNodes = True
        )

    print(f"Generation: {generation}, Best fitness: {best.fitness}")

hf.plotNetwork(best, "best_solution.html", justUsedNodes = True , justUsedEdges = True)

# print decisions of best individuals
print("Best individual's decisions: ", best.fitnessValues)
