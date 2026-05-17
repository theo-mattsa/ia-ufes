import math
import random
from constants import CLIENTS, DEPOSIT


def fitness(individual):
    distance = 0
    path = [CLIENTS[i] for i in individual]
    path.insert(0, DEPOSIT)
    path.append(DEPOSIT)
    for i in range(len(path) - 1):
        distance += math.dist(path[i], path[i + 1])
    return distance


def create_population(pop_size):
    n_clients = len(CLIENTS)
    return [random.sample(range(n_clients), n_clients) for _ in range(pop_size)]


def selection(population):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        ind1 = random.choice(population)
        ind2 = random.choice(population)
        if fitness(ind1) < fitness(ind2):
            selected.append(ind1)
        else:
            selected.append(ind2)
    return selected


def generate_child(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    used = set(child[start:end])
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_idx] in used:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            used.add(parent2[p2_idx])

    return child


def crossover(population, crossover_rate=0.8):
    children = []
    random.shuffle(population)
    for i in range(0, len(population), 2):
        p1 = population[i]
        p2 = population[i + 1]
        if random.random() <= crossover_rate:
            c1 = generate_child(p1, p2)
            c2 = generate_child(p2, p1)
        else:
            c1 = p1.copy()
            c2 = p2.copy()
        children.append(c1)
        children.append(c2)
    return children


def mutation(population, mutation_rate=0.1):
    mutated_population = []
    for individual in population:
        ind = individual.copy()
        if random.random() <= mutation_rate:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        mutated_population.append(ind)
    return mutated_population


def genetic_algorithm(
    pop_size=100, generations=200, crossover_rate=0.8, mutation_rate=0.1
):

    population = create_population(pop_size)

    best_solution = None
    best_fitness = float("inf")

    for generation in range(generations):

        current_best = min(population, key=fitness)
        current_fitness = fitness(current_best)

        if current_fitness < best_fitness:
            best_solution = current_best.copy()
            best_fitness = current_fitness

        selected = selection(population)
        children = crossover(selected, crossover_rate)
        population = mutation(children, mutation_rate)

        if generation % 10 == 0:
            print(
                f"Generation {generation:03d} | " f"Best Distance = {best_fitness:.2f}"
            )

    return best_solution, best_fitness


if __name__ == "__main__":

    best_route, best_distance = genetic_algorithm()

    print("\nBest Route:")
    print(best_route)

    print("\nBest Distance:")
    print(best_distance)
