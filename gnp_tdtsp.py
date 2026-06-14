import helperfunctions as hf 
import numpy as np
import pandas as pd 
import fracnetics as fn
import math
import matplotlib.pyplot as plt

# 0. Parameters
seed=17
N = 5000 # Table size (number of waiting times)
num_edges = 6 # Number of stations
individuals = 1000 # Number of individuals in the population
judgment_nodes = 1 # Number of judgment nodes
processing_nodes = 6 # Number of processing nodes
minitues_per_index = 1 # number of minutes per waiting time index
generations = 300 # number of generations to run the algorithm
worstFitness = -1000 # worst fitness value for invalid individuals

# 1. Generate random T table 
timetablesEachNode = []
for i in range(num_edges):

    T = hf.generate_route_delay_table(
            N=N, 
            num_edges=num_edges,
            max_base_delay=2,
            accident_prob=0.2,
            min_accident_duration=20,
            max_accident_duration=100,
            min_accident_impact=10,
            max_accident_impact=100, # waiting time cant be less than the time it takes to get to the next station
            seed=seed+i
            )
    timetablesEachNode.append(T)
print(np.round(timetablesEachNode[1],2))

# 2. Read distance matrix
distances = pd.read_csv('data/distances.csv', sep=";", index_col=0).values
print(distances)

# 3. Running GNP 
pop = fn.Population(
    seed=seed,
    ni=individuals, # number of individuals
    jn=6, # judgment nodes
    jnf=1, # judgment node functions
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


fitness_over_time = []
for generation in range(generations):

    # 3.2 Evaluate fitness of individuals
    for ind in pop.individuals:
        ind.initPathTraversal()
        visited_processing_nodes = set()
        visited_processing_nodes.add(num_edges) # waiting node is considered as visited at the beginning
        currentTime = 0
        currentStation = 0 # start at station 0
        visited_processing_nodes.add(currentStation) # waiting node is considered as visited at the beginning
        fitness = 0
        stop = False
        decisions = [currentStation]

        while len(visited_processing_nodes) < num_edges+1 and currentTime < N:
            delays = timetablesEachNode[currentStation][currentTime]
            dec = ind.decisionAndNextNode([delays[currentStation]],dMax=5) 
            decisions.append(dec)

            if ind.invalid == True or (dec in visited_processing_nodes and dec != currentStation):
                stop = True
                break
            
            if dec == num_edges or currentStation == dec: # decision is to wait 
               currentTime += minitues_per_index 
               fitness += minitues_per_index
            else:
                visited_processing_nodes.add(dec)
                fitness += distances[currentStation, dec] # add distance to fitness
                fitness += delays[currentStation] #  add delay to fitness
                #dist = math.ceil(distances[currentStation, dec] / minitues_per_index) # add distance to current time
                dist = distances[currentStation, dec] # add distance to current time
                currentTime += dist
                currentTime += math.ceil(delays[currentStation]) # add delay to current time
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

    pop.callAddDelNodes(
        minFeatures, 
        maxFeatures, 
        junk=0.5
    )

    # Crossover (recombination)
    pop.crossover(
        probability = 0.05, 
        type = "uniform",
    )

    # 3.4 Mutation
    pop.callEdgeMutation(
        probInnerNodes = 0.05, # probability of changing an edge of jn or pn
        probStartNode = 0.05,  # probability of changing an edge of the start node
        justUsedNodes = True
    )

    pop.callBoundaryMutationUniform(
        probability = 0.05,
        justUsedNodes = True
        )

    usedJN = 0 
    for node in best.innerNodes:
        if node.used == True and node.type == "J":
            usedJN += 1
    print(f"Generation: {generation} | Best fitness: {best.fitness} | NN: {len(best.innerNodes)} | usedJN: {usedJN}")
    fitness_over_time.append([ind.fitness for ind in pop.individuals])

#best.innerNodes[best.currentNodeID].used = False # set used to false for the best individual to prevent plotting the last used node and edge (which is not used in the best route)
hf.plotNetwork(best, f"best_generation_{generation}_fitness_{best.fitness}.html", justUsedNodes = False, justUsedEdges = False)

# print decisions of best individuals
print("Best individual's decisions: ", best.fitnessValues)
routeTime = 0
currentStation = int(best.fitnessValues[0])
stations = list(dict.fromkeys(best.fitnessValues))
print(stations)
for dec in stations[1:]:
    routeTime += distances[currentStation, int(dec)]
    currentStation = int(dec)

print("Route time without delay: ", routeTime)

bestRoute, bestCost = hf.bruteforce_fastest_route(start=0, num_stations=num_edges, distances=distances)
print("Best route: ", bestRoute)
print("Best cost no delay: ", bestCost)

sumDelay = 0
currentTime = 0
currentStation = 0
for dec in bestRoute[1:]:
    delay = timetablesEachNode[currentStation][currentTime][currentStation]
    sumDelay += delay #  add delay to fitness
    #dist = math.ceil(distances[currentStation, dec] / minitues_per_index) # add distance to current time
    dist = distances[currentStation, dec] # add distance to current time
    currentTime += dist
    currentTime += math.ceil(delay) # add delay to current time
    currentStation = dec

print("Total delay: ", sumDelay+bestCost)

# Plot with matplotlib the fitness values as boxplot over generations

#for i in range(len(fitness_over_time)):
    #plt.boxplot(fitness_over_time[i], positions=[i], widths=0.5)

plt.plot([i for i in range(generations)], [fit[-1] for fit in fitness_over_time])
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness over generations")

#plt.show()
