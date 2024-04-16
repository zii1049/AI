import numpy as np, random, operator, pandas as pd
# matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"



class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
        
###############################################################################
    
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population
    # print(population)


################################################################################

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#################################################################################################

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults



def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


#############################################################################################

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

############################################################################################

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


################################################################################################

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#############################################################################################

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

##############################################################################################

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


##########################################################################################
##########################################################################################


# city1 = City(2 ,3)
# city2 = City(20 ,30)
# city3 = City(12 ,13)
# city4 = City(22 ,33)
# city5 = City(25 ,53)

# city_list = [city1,city2,city3,city4,city5]
# print(city_list)

# population = initialPopulation(4 , city_list)
# # print(initial_pop)

# fitness_result = rankRoutes(population)
# print(fitness_result)

# selection_result = selection(fitness_result ,2)
# # print(selection_result)

# mating_pool = matingPool(population, selection_result)
# # print(mating_pool)

# children = breedPopulation(mating_pool,2)
# # print(children)

# mutate_pop = mutatePopulation(children , 0.1)
# # print(mutate_pop)

# next_generation = nextGeneration(mutate_pop , 2 , 0.1)
# # print(next_generation)

# print(rankRoutes(next_generation))
city1 = City(2 ,3)
city2 = City(20 ,30)
city3 = City(12 ,13)
city4 = City(22 ,33)
city5 = City(25 ,53)
city_list = [city1,city2,city3,city4,city5]

# Constants
POPULATION_SIZE = 4
MUTATION_RATE = 0.1
# TOURNAMENT_SIZE = 5
NUM_GENERATIONS = 3
NUM_ELITE = 2

fittest_list = []
for i in range(NUM_GENERATIONS):
    population = initialPopulation(POPULATION_SIZE , city_list)
    fitness_result = rankRoutes(population)
    selection_result = selection(fitness_result ,NUM_ELITE)
    mating_pool = matingPool(population, selection_result)
    children = breedPopulation(mating_pool,NUM_ELITE)
    mutate_pop = mutatePopulation(children , MUTATION_RATE)
    next_generation = nextGeneration(mutate_pop , NUM_ELITE , MUTATION_RATE)
    population = next_generation

    fitness_list = list(map(lambda x:x[1] , fitness_result))
    fittest_list.append(fitness_list[-1])

print(fittest_list)
fittest_list.sort()
print(fittest_list[0])