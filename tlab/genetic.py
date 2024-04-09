import numpy as np

def objective_function(x):
    return x**2 - 4*x + 4

solution_space = (-10, 10)
population_size = 50
num_generations = 100
mutation_rate = 0.1

def initialize_population():
    return np.random.uniform(*solution_space, size=population_size)
def calculate_fitness(population):
    return np.array([-objective_function(x) for x in population])  # Minimize the objective function  # Minimize the objective function

def selection(population, fitness):
    probabilities = fitness / np.sum(fitness)
    selected_indices = np.random.choice(len(population), size=2, p=probabilities)
    return [population[index] for index in selected_indices]
def crossover(parent1, parent2):
    # Single-point crossover
    if np.isscalar(parent1) and np.isscalar(parent2):
        # If both parents are scalar values, return them as children
        return parent1, parent2
    else:
        if np.isscalar(parent1):
            parent1 = np.array([parent1])
        else:
            parent1 = np.array(parent1)
        if np.isscalar(parent2):
            parent2 = np.array([parent2])
        else:
            parent2 = np.array(parent2)
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
def mutation(child):
    # Random mutation by adding a small random value
    if np.random.rand() < mutation_rate:
        return child + np.random.uniform(-0.5, 0.5)
    return child

def genetic_algorithm():
    population = initialize_population()
    for generation in range(num_generations):
        # Evaluate fitness
        fitness = calculate_fitness(population)
        # Select parents
        parents = [selection(population, fitness) for _ in range(population_size // 2)]
        # Perform crossover and mutation
        offspring = []
        for parent_pair in parents:
            parent1, parent2 = parent_pair
            child1, child2 = crossover(parent1, parent2)
            offspring.append(mutation(child1))
            offspring.append(mutation(child2))
        # Combine parents and offspring
        population = np.concatenate((population, offspring))
        # Select top individuals for the next generation
        population = sorted(population, key=objective_function)[:population_size]
        # Print best fitness in current generation
        best_fitness = -max(fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
    # Select best solution from final population
    best_solution = min(population, key=objective_function)
    best_fitness = objective_function(best_solution)
    print(f"Optimal Solution: {best_solution}, Optimal Fitness: {best_fitness}")

# Run the genetic algorithm
genetic_algorithm()