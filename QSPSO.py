import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute

def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def quantum_update(position, p_best, g_best, func):
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1, 2])
    qc.cx(0, 1)
    qc.cx(1, 2)

    angle = np.pi * (func(position) / (func(p_best) + func(g_best)))
    qc.ry(angle, [0, 1, 2])

    qc.measure([0, 1, 2], [0, 1, 2])

    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1)
    result = job.result()
    counts = result.get_counts(qc)
    measured_state = list(counts.keys())[0]

    new_position = position.copy()
    if measured_state[2] == '1':
        new_position += np.random.uniform(-0.1, 0.1, len(position))
    if measured_state[1] == '1':
        new_position = (new_position + p_best) / 2
    if measured_state[0] == '1':
        new_position = (new_position + g_best) / 2

    return new_position

def qpso(func, num_particles, dim, max_iter, bounds):
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    p_best = particles.copy()
    g_best = particles[np.argmin([func(p) for p in particles])]

    for _ in range(max_iter):
        for i in range(num_particles):
            particles[i] = quantum_update(particles[i], p_best[i], g_best, func)
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

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
        best = qpso(func, num_particles, dim, max_iter, bound)
        results.append(func(best))
    
    print(f"{name} function:")
    print(f"Average: {np.mean(results)}")
    print(f"Best: {np.min(results)}")
    print(f"Worst: {np.max(results)}")
    print()

    plt.figure()
    plt.hist(results, bins=10)
    plt.title(f"QPSO on {name} Function")
    plt.xlabel("Function Value")
    plt.ylabel("Count")
    plt.show()