import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PENALTY = 10_000
DIMENSION = 30


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
    cost = ackley(individual)
    penalty = PENALTY if np.any(individual < -32) or np.any(individual > 32) else 0
    return cost + penalty


def create_population(population_size: int) -> np.ndarray:
    return np.random.uniform(-32, 32, size=(population_size, DIMENSION))


def mutation(population: np.ndarray, factor: float):
    i1, i2, i3 = np.random.choice(len(population), 3, replace=False)
    return population[i1] + factor * (population[i2] - population[i3])


def crossover(population: np.ndarray, mutant: np.ndarray, rate: float):
    new_i = []
    i1_idx = np.random.choice(len(population), replace=False)
    i1 = population[i1_idx]
    for i in range(len(i1)):
        if np.random.random() <= rate:
            new_i.append(mutant[i])
        else:
            new_i.append(i1[i])

    return np.array(new_i)


def de(
    generations: int = 100,
    population_size: int = 50,
    mutation_factor: float = 0.4,
    crossover_rate: float = 0.7,
):
    history = []
    population = create_population(population_size)

    for generation in range(generations):

        fitnesses = np.array([fitness(ind) for ind in population])

        best_idx = np.argmin(fitnesses)
        score = fitnesses[best_idx]
        history.append(score)

        # print(f"Generation: {generation}, score: {score}")

        new_population = []

        for _ in range(population_size):

            # Mutation
            mutant = mutation(population, mutation_factor)

            # Crossover
            new_individual = crossover(population, mutant, crossover_rate)

            # Selection
            random_idx = np.random.choice(len(population))
            random_ind = population[random_idx]

            if fitness(new_individual) < fitness(random_ind):
                new_population.append(new_individual)
            else:
                new_population.append(random_ind)

        population = new_population
    return history


def experiment(
    iterations: int,
    generations: int,
    population_size: int,
    cross_rate: float,
    mutation_factor: float,
):
    results = []
    print(
        "Experiment: Generations:",
        generations,
        "Crossover Rate:",
        cross_rate,
        "Mutation Factor:",
        mutation_factor,
        "Population Size:",
        population_size,
    )
    for _ in tqdm(range(iterations)):
        history = de(
            generations=generations,
            population_size=population_size,
            mutation_factor=mutation_factor,
            crossover_rate=cross_rate,
        )
        results.append(history)

    return results


if __name__ == "__main__":

    results = experiment(
        iterations=5,
        generations=120,
        population_size=80,
        cross_rate=0.7,
        mutation_factor=0.4,
    )

    final_scores = np.array([h[-1] for h in results])

    print("Mean:", np.mean(final_scores))
    print("Std:", np.std(final_scores))
    print("Best:", np.min(final_scores))

    # Plot
    for history in results:
        plt.plot(history)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Differential Evolution")

    plt.show()
