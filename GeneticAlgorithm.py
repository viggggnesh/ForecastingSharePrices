import numpy
import GA

equation_inputs = [4, -2, 3.5, 5, -11, -4.7]

num_weights = 6


sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop,
            num_weights)

new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

num_generations = 5

for generation in range(num_generations):
    print("Generation : ", generation)

    fitness = GA.cal_pop_fitness(equation_inputs,new_population)

    parents = GA.select_mating_pool(new_population,fitness,num_parents_mating)

    offspring_crossover = GA.crossover(parents,offspring_size=(pop_size[0] - parents.shape[0], num_weights))

    offspring_mutation = GA.mutation(offspring_crossover)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    print("Best result : ", numpy.max(numpy.sum(new_population * equation_inputs, axis=1)))

fitness = GA.cal_pop_fitness(equation_inputs, new_population)

best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])