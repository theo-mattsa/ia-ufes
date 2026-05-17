import math
import numpy as np
import random
from constants import CLIENTS, DEPOSIT


def fitness(x: tuple):

    distance = 0
    path = [CLIENTS[i] for i in x]
    path.insert(0, DEPOSIT)
    path.append(DEPOSIT)

    for i in range(len(path) - 1):
        distance += math.dist(path[i], path[i + 1])

    return distance


def create_population(N: int) -> list:
    NUM_CLIENTS = len(CLIENTS)
    return [
        np.random.choice(NUM_CLIENTS, size=NUM_CLIENTS, replace=False) for _ in range(N)
    ]


def selection(pop: list) -> list:
    """
    Get the intermediate population
    """

    pop_i = list()
    N = len(pop)

    for _ in range(N):
        ind_1 = random.choice(pop)
        ind_2 = random.choice(pop)

        fit_1 = fitness(ind_1)
        fit_2 = fitness(ind_2)

        if fit_1 < fit_2:
            pop_i.append(ind_1)
        else:
            pop_i.append(ind_2)

    return pop_i


def crossover(pop_i: list, rate: float = 0.7):
    pop_ii = []
    N = len(pop_i)

    def generate_children(p_1, p_2):
        chrom_size = len(p_1)
        start, end = sorted(random.sample(range(chrom_size), 2))
        c = [None] * chrom_size
        c[start:end] = p_1[start:end]
        used = set(c[start:end])
        p2_i = 0
        for i in range(chrom_size):
            if c[i] is None:
                while p_2[p2_i] in used:
                    p2_i += 1
                c[i] = p_2[p2_i]
                used.add(p_2[p2_i])
                p2_i += 1
        return c

    while len(pop_ii) < N:
        p_1 = random.choice(pop_i)
        p_2 = random.choice(pop_i)
        if random.random() <= rate:
            c_1 = generate_children(p_1, p_2)
            c_2 = generate_children(p_2, p_1)
        else:
            c_1 = p_1.copy()
            c_2 = p_2.copy()
        pop_ii.append(c_1)
        pop_ii.append(c_2)

    return pop_ii


def mutation(pop_ii: list, rate: float = 0.1):
    pop_iii = []

    for _ in range(len(pop_ii)):
        p_1 = random.choice(pop_ii).copy()
        if random.random() <= rate:
            size = len(p_1)
            i, j = sorted(random.sample(range(size), 2))
            p_1[i:j] = p_1[i:j][::-1]
        pop_iii.append(p_1)
    return pop_iii


def genetic_alg(num_epochs: int = 50):
    N = 50

    best_score = None
    pop_p = create_population(N)

    for i in range(num_epochs):

        pop_i = selection(pop_p)
        pop_ii = crossover(pop_i)
        pop_iii = mutation(pop_ii)

        pop_p = pop_iii

        best_i = min(pop_p, key=fitness)

        print(f"Best score for generation {i}: {fitness(best_i)}")

    best = min(pop_p, key=fitness)
    return best, fitness(best)


if __name__ == "__main__":
    genetic_alg()
