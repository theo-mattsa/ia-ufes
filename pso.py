import numpy as np


def objective_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return term1 + term2 + a + np.e


class PSO:
    def __init__(
        self,
        num_particles: int,
        dimensions: int,
        num_iterations: int,
        bounds: tuple,
        objective_fun,
        w: float,
        c1: float,
        c2: float,
    ):

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.num_iterations = num_iterations
        self.objective_fun = objective_fun
        self.bounds = bounds

        self.positions = np.random.uniform(
            low=bounds[0],
            high=bounds[1],
            size=(num_particles, dimensions),
        )

        self.velocity = np.random.uniform(
            low=-1,
            high=1,
            size=(num_particles, dimensions),
        )

        # Particles best position and score
        self.pbest = self.positions.copy()
        self.pbest_score = np.apply_along_axis(
            self.objective_fun,
            1,
            self.positions,
        )

        # Global best positions and score
        idx = np.argmin(self.pbest_score)
        self.gbest = self.positions[idx].copy()
        self.gscore = self.pbest_score[idx]

    def _update_velocity(self):
        r1 = np.random.random(self.positions.shape)
        r2 = np.random.random(self.positions.shape)
        cognitive = self.c1 * r1 * (self.pbest - self.positions)
        social = self.c2 * r2 * (self.gbest - self.positions)
        self.velocity = self.w * self.velocity + cognitive + social

    def _update_positions(self):
        self.positions += self.velocity

        # Ensure (-32, 32)
        self.positions = np.clip(
            self.positions,
            self.bounds[0],
            self.bounds[1],
        )

    def _update_best_positions(self):

        scores = np.apply_along_axis(
            self.objective_fun,
            1,
            self.positions,
        )

        improved = scores < self.pbest_score

        self.pbest[improved] = self.positions[improved]
        self.pbest_score[improved] = scores[improved]

        best_idx = np.argmin(scores)
        if scores[best_idx] < self.gscore:
            self.gscore = scores[best_idx]
            self.gbest = self.positions[best_idx].copy()

    def optimize(self):

        for iteration in range(self.num_iterations):
            self._update_velocity()
            self._update_positions()
            self._update_best_positions()
            print(f"Iteration {iteration + 1:3d} | " f"Best Score: {self.gscore:.10f}")

        return self.gbest, self.gscore


if __name__ == "__main__":

    pso = PSO(
        num_particles=100,
        dimensions=30,
        num_iterations=150,
        bounds=(-32, 32),
        objective_fun=objective_function,
        w=0.7,
        c1=2.5,
        c2=2.5,
    )

    best_position, best_score = pso.optimize()

    print("Optimization Finished")
    print("Best Position:")
    print(best_position)

    print("Best Score:")
    print(best_score)
