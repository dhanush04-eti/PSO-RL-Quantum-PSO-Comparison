import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def pso_rl(func, num_particles, dim, max_iter, bounds):
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    p_best = particles.copy()
    g_best = particles[np.argmin([func(p) for p in particles])]
    Q = np.zeros((num_particles, 3))

    for _ in range(max_iter):
        for i in range(num_particles):
            action = np.argmax(Q[i])

            r1, r2 = np.random.rand(2)
            velocities[i] = (0.7 * velocities[i] +
                             2 * r1 * (p_best[i] - particles[i]) +
                             2 * r2 * (g_best - particles[i]))

            particles[i] += velocities[i]
            particles[i] += (action - 1) * 0.1
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            reward = -func(particles[i])
            Q[i, action] += 0.1 * (reward + 0.9 * np.max(Q[i]) - Q[i, action])

            if func(particles[i]) < func(p_best[i]):
                p_best[i] = particles[i]

            if func(particles[i]) < func(g_best):
                g_best = particles[i]

    return g_best

num_particles = 30
dim = 10
max_iter = 500
num_runs = 30

functions = [sphere, rastrigin, ackley]
func_names = ['Sphere', 'Rastrigin', 'Ackley']
bounds = [(-5.12, 5.12), (-5.12, 5.12), (-32.768, 32.768)]

for func, name, bound in zip(functions, func_names, bounds):
    results = []
    for _ in range(num_runs):
        best = pso_rl(func, num_particles, dim, max_iter, bound)
        results.append(func(best))
    
    print(f"{name} function:")
    print(f"Average: {np.mean(results)}")
    print(f"Best: {np.min(results)}")
    print(f"Worst: {np.max(results)}")
    print()

    plt.figure()
    plt.hist(results, bins=10)
    plt.title(f"PSO-RL on {name} Function")
    plt.xlabel("Function Value")
    plt.ylabel("Count")
    plt.show()