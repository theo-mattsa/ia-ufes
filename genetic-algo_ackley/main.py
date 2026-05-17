import numpy as np


def ackley(x: np.ndarray) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)


def fitness(individual: np.ndarray):
    return ackley(individual)


def create_population(pop_size: int) -> list:
    return np.random.uniform(low=-32, high=32, size=(pop_size, 30))


def selection(population: np.ndarray) -> np.ndarray:
    selected = []
    for _ in range(len(population)):
        i, j = np.random.choice(len(population), 2)
        ind1 = population[i]
        ind2 = population[j]
        if fitness(ind1) < fitness(ind2):
            selected.append(ind1)
        else:
            selected.append(ind2)
    return np.array(selected)


def crossover(parent1: np.ndarray, parent2: np.ndarray, rate: int = 0.8):
    if np.random.random() <= rate:
        beta = np.random.normal()
        child1 = beta * parent1 + (1 - beta) * parent2
        child2 = (1 - beta) * parent1 + beta * parent2
        return child1, child2
    return parent1, parent2


def mutation(individual: np.ndarray, rate: int = 0.1) -> np.ndarray:
    if np.random.random() <= rate:
        alpha = np.random.normal()
        return alpha * individual
    return individual


def genetic_algorithm(
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
):
    population = create_population(population_size)

    for generation in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])

        best_idx = np.argmin(fitnesses)
        score = fitnesses[best_idx]

        population = selection(population)

        new_population = list()

        # Assumption: A população é par
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.append(mutation(child1, mutation_rate))
            new_population.append(mutation(child2, mutation_rate))

        print(f"Generation {generation} | Fitness = {score:.2f}")

        population = np.array(new_population)


genetic_algorithm()
