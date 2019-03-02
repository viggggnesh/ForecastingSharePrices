import pandas_datareader.data as dr
import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sp
import csv
from itertools import zip_longest

ALLELE_SIZE = 5
POPULATION_SIZE = 8
ELITE_CHROMOSOMES = 1
SELECTION_SIZE = 4
MUTATION_RATE = 0.25
TARGET_FITNESS = 0.9
ALPHA = 0.1

R = [175, 280, 320, 340, 360, 385]


symbol = "TSLA"
start = datetime.datetime(2016, 10, 19)
end = datetime.date.today()
data = dr.DataReader(symbol, "google", start, end)


PRICE = data["Close"].values.tolist()
SIGMA0 = np.std(PRICE)


#class

class Chromosome:
    """
    Candidate solution.
    """

    def __init__(self):
        self._genes =
        self._fitness = 0
        self._genes.append(random.randint(R[0], R[1]))
        self._genes.append(random.randint(R[1], R[2]))
        self._genes.append(random.randint(R[2], R[3]))
        self._genes.append(random.randint(R[3], R[4]))
        self._genes.append(random.randint(R[4], R[5]))

    def get_genes(self):
        return self._genes

    def get_fitness(self):
        L =
        D =
        for i in range(len(PRICE) - 2):
            if self._genes[0] <= PRICE[i]:
                if self._genes[1] <= PRICE[i + 1] or PRICE[i + 1] <= self._genes[2]:
                    if self._genes[3] <= PRICE[i + 2] or PRICE[i + 2] <= self._genes[4]:
                        L.append(PRICE[i + 1])
                        D.append(i)
        sigma = np.std(L)
        nc = len(L)
        self._fitness = -np.log2(sigma / SIGMA0) - ALPHA / nc
        return self._fitness

    def __str__(self):
        return self._genes.__str__()


class Population:

    def __init__(self, size):
        self._chromosomes =
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome())
            i += 1

    def get_chromosomes(self):
        return self._chromosomes


class GeneticAlgorithm:

    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))

    @staticmethod
    def _crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop

    @staticmethod
    def _mutate_population(pop):
        for i in range(ELITE_CHROMOSOMES, POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosomes(pop.get_chromosomes()[i])
        return pop

    @staticmethod
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        for i in range(ALLELE_SIZE):
            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    crossover_chrom.get_genes()[i] = chromosome1.get_genes()


[i] else:
crossover_chrom.get_genes()[i] = chromosome2.get_genes()
[i]
return crossover_chrom


@staticmethod
def _mutate_chromosomes(chromosome):
    if random.random() < MUTATION_RATE:
        chromosome.get_genes()[0] = random.randint(R[0], R[1])
    elif random.random() < MUTATION_RATE:
        chromosome.get_genes()[1] = random.randint(R[1], R[2])
    elif random.random() < MUTATION_RATE:
        chromosome.get_genes()[2] = random.randint(R[2], R[3])
    elif random.random() < MUTATION_RATE:
        chromosome.get_genes()[3] = random.randint(R[3], R[4])
    elif random.random() < MUTATION_RATE:
        chromosome.get_genes()[4] = random.randint(R[4], R[5])


@staticmethod
def _select_population(pop):
    select_pop = Population(0)
    i = 0
    while i < SELECTION_SIZE:
        select_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
        i += 1
    select_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    return select_pop


#fn

def print_population(pop, gen_number):

    print("Generation #", gen_number, ": Fittest chromosome fitness: %3.2f" % pop.get_chromosomes()[0].get_fitness())
    i = 0
    for x in pop.get_chromosomes():
        print("Chromosome #", i, " :", x, "| Fitness: %3.2f" % x.get_fitness())
        i += 1


#main

population = Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(),
                                  reverse=True)  # Population with the highest fitness is the first
print_population(population, 0)

gen = 1
while population.get_chromosomes()[0].get_fitness() < TARGET_FITNESS:
    population = GeneticAlgorithm.evolve(population)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    print_population(population, gen)
    gen += 1

genes =
for i in range(5):
    genes.append(population.get_chromosomes()[0].get_genes()[i])

for i in range(len(PRICE) - 2):
    if genes[0] <= PRICE[i]:
        if genes[1] <= PRICE[i + 1] or PRICE[i + 1] <= genes[2]:
            if genes[3] <= PRICE[i + 2] or PRICE[i + 2] <= genes[4]:
                L.append(PRICE[i + 1])
                D.append(i)