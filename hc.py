import numpy as np
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


# Vanilla
def hill_climb(*, random_current: bool = False):
    current = np.random.uniform(0, 1, size=4) if random_current else np.zeros(4)
    current = np.round(current, decimals=2)

    while True:
        nbrs = neighbors(current)
        if not nbrs:
            break
        best_nbr = max(nbrs, key=fitness)
        if fitness(current) >= fitness(best_nbr):
            return current
        current = best_nbr
    return current


# Random Restart Hill Climb
def rr_hill_climb(n: int):
    best = None
    for _ in range(n):
        candidate = hill_climb(random_current=True)
        if best is None or fitness(candidate) > fitness(best):
            best = candidate
    return best


# Stochastic Hill Climbing
def sh_hill_climb():
    current = np.random.uniform(0, 1, size=4)
    while True:
        nbrs = neighbors(current)
        better = [x for x in nbrs if fitness(x) > fitness(current)]
        if not better:
            return current
        current = random.choice(better)


if __name__ == "__main__":
    hc = sh_hill_climb()
    print(fitness(hc))
