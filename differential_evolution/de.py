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


def differential_evolution(
    func,
    bounds,
    dimensions,
    population_size=50,
    F=0.4,
    CR=0.7,
    generations=100,
):
    rng = np.random.default_rng()

    min_bound, max_bound = bounds

    population = rng.uniform(min_bound, max_bound, size=(population_size, dimensions))

    fitness = np.asarray([func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = population[best_idx]

    for gen in range(generations):
        for i in range(population_size):

            # Choose 3 random individuals
            idxs = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[rng.choice(idxs, 3, replace=False)]

            # Mutation
            mutant = np.clip(a + F * (b - c), min_bound, max_bound)

            # Crossover
            mask = rng.random(dimensions) < CR
            if not np.any(mask):
                mask[rng.randint(0, dimensions)] = True

            trial = np.where(mask, mutant, population[i])

            # Selection
            f = func(trial)

            if f < fitness[i]:
                population[i] = trial
                fitness[i] = f
                if f < fitness[best_idx]:
                    best_idx = i
                    best = trial
        print(f"Generation: {gen + 1}: Best fitness = {fitness[best_idx]}")

    return best, fitness[best_idx]


if __name__ == "__main__":
    best, fitness = differential_evolution(
        ackley, bounds=(-32, 32), dimensions=30, population_size=100, generations=250
    )
    print(best)
    print(fitness)
