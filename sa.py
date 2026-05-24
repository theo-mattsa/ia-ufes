import numpy as np
import math
import random

MAX_INVESTMENT = 50
MAX_YT_LINK = 25
MAX_HOURS = 80


def fitness(x: tuple):
    x1, x2, x3, x4 = x
    return (
        50 * x1
        - 1.2 * x1**2
        + 45 * x2
        - 0.8 * x2**2
        + 40 * x3
        - 0.8 * x3**2
        + 55 * x4
        - 1.5 * x4**2
    )


def is_feasible(x: tuple):
    if min(x) < 0:
        return False

    x1, x2, x3, x4 = x

    total_invested = sum(x)
    total_yt_link = x3 + x4
    total_hours_invested = 2 * x1 + x2 + 3 * x3 + 2 * x4

    return (
        total_invested <= MAX_INVESTMENT
        and total_yt_link <= MAX_YT_LINK
        and total_hours_invested <= MAX_HOURS
    )


def neighbors(x: tuple):
    nbrs = []
    deltas = np.arange(-1, 1.25, 0.25)
    deltas = deltas[deltas != 0]
    for i in range(len(x)):
        for delta in deltas:
            y = list(x)
            y[i] += delta
            if is_feasible(y):
                nbrs.append(y)
    return nbrs


def simulated_annealing(T=100.0, T_min=0.01, cooling=0.99):

    current = (0, 0, 0, 0)
    current_score = fitness(current)

    best = current
    best_score = current_score

    while T > T_min:
        candidate = random.choice(neighbors(current))
        candidate_score = fitness(candidate)
        delta = candidate_score - current_score
        if delta > 0:
            current = candidate
            current_score = candidate_score
        else:
            prob = math.exp(delta / T)
            if random.random() < prob:
                current = candidate
                current_score = candidate_score
        if current_score > best_score:
            best = current
            best_score = current_score

        T *= cooling
    return best


if __name__ == "__main__":
    print(fitness(simulated_annealing()))
