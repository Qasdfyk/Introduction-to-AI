import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from solver import gradient_descent
from tabulate import tabulate

def plot_2d(problem, x0, beta, eps):
    x1 = np.arange(-2, 2, 0.01)
    x2 = np.arange(-2, 2, 0.01)
    x1, x2 = np.meshgrid(x1, x2)
    z = problem([x1, x2])
    x, it, minimum = gradient_descent(problem, x0, beta, eps)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x1, x2, z, cmap=cm.gist_earth)
    ax.scatter(x[0], x[1], minimum, c='darkred', s=30)
    plt.show()
    fig, ax = plt.subplots()
    ax.tricontourf(x1.flatten(), x2.flatten(), z.flatten())
    ax.scatter(x[0], x[1], color='black', marker='x', s=100)
    plt.show()

def plot_1d(problem, x0, beta, eps):
    fig, ax = plt.subplots()
    x = np.linspace(-10, 10, 1000)
    y = problem(x)
    xt, it, min = gradient_descent(problem, x0, beta, eps)
    ax.plot(x, y, linewidth=2.0,zorder=0)
    ax.set(xlim=(-10, 10), xticks=np.arange(-10, 10),
        ylim=(-5, 10), yticks=np.arange(-5, 10))    
    ax.scatter(xt, min, c='darkred', s=30, zorder=1)
    plt.show()

# beta < 1.5 i beta > 0.2
def test_beta_2d_avg(problem, eps):
    results_avg = {}
    for i in range(1000):
        beta = 0.2
        good_random_start_point = np.array([np.random.uniform(-0.3, 2.5), np.random.uniform(-2, 1.8)])
        while beta < 1.5:
            _, it, _ = gradient_descent(problem, good_random_start_point, beta, eps)
            if round(beta,2) not in results_avg:
                results_avg[round(beta,2)] = it
            else:
                results_avg[round(beta,2)] += it
            beta += 0.05
    return {k: round(v /  i)for k, v in results_avg.items()}

def test_beta_2d_std(problem, eps):
    results_std = {}
    beta = 0.2
    while beta < 1.5:
        for _ in range(1000):
            good_random_start_point = np.array([np.random.uniform(-0.3, 2.5), np.random.uniform(-2, 1.8)])
            _, it, _ = gradient_descent(problem, good_random_start_point, beta, eps)
            if round(beta,2) not in results_std:
                results_std[round(beta,2)] = [it]
            else:
                results_std[round(beta,2)].append(it)
        beta += 0.05
    return {k: round(np.std(np.array(v))) for k, v in results_std.items()}

# beta < 0.45
def test_beta_1d(problem, eps):
    results = {}
    for i in range(1000):
        beta = 0.1
        random_start_point = [np.random.uniform(-1000, 1000)]
        while beta < 0.45:
            _, it, _ = gradient_descent(problem, random_start_point, beta, eps)
            if round(beta,2) not in results:
                results[round(beta,2)] = it
            else:
                results[round(beta,2)] += it
            beta += 0.01
    return {k: round(v / i) for k, v in results.items()}

def test_beta_1d_std(problem, eps):
    results_std = {}
    beta = 0.1
    while beta < 0.45:
        for _ in range(1000):
            good_random_start_point = [np.random.uniform(-100, 100)]
            _, it, _ = gradient_descent(problem, good_random_start_point, beta, eps)
            if round(beta,2) not in results_std:
                results_std[round(beta,2)] = [it]
            else:
                results_std[round(beta,2)].append(it)
        beta += 0.01
    return {k: round(np.std(np.array(v))) for k, v in results_std.items()}

def make_tables():
    g_avg={0.2: 130, 0.25: 103, 0.3: 85, 0.35: 72, 0.4: 63, 0.45: 55, 0.5: 49, 0.55: 44, 0.6: 40, 0.65: 36, 0.7: 33, 0.75: 31, 0.8: 28, 0.85: 26, 0.9: 24, 0.95: 24, 1.0: 25, 1.05: 25, 1.1: 25, 1.15: 26, 1.2: 27, 1.25: 29, 1.3: 31, 1.35: 34, 1.4: 38, 1.45: 45}
    g_std = {0.2: 267, 0.25: 298, 0.3: 129, 0.35: 124, 0.4: 150, 0.45: 110, 0.5: 110, 0.55: 71, 0.6: 93, 0.65: 93, 0.7: 80, 0.75: 84, 0.8: 93, 0.85: 65, 0.9: 70, 0.95: 75, 1.0: 61, 1.05: 60, 1.1: 53, 1.15: 39, 1.2: 49, 1.25: 50, 1.3: 51, 1.35: 44, 1.4: 40, 1.45: 46}
    f_avg = {0.1: 37, 0.11: 33, 0.12: 29, 0.13: 26, 0.14: 23, 0.15: 21, 0.16: 19, 0.17: 17, 0.18: 15, 0.19: 14, 0.2: 12, 0.21: 11, 0.22: 9, 0.23: 8, 0.24: 6, 0.25: 1, 0.26: 6, 0.27: 8, 0.28: 9, 0.29: 11, 0.3: 12, 0.31: 14, 0.32: 15, 0.33: 17, 0.34: 19, 0.35: 21, 0.36: 23, 0.37: 26, 0.38: 29, 0.39: 33, 0.4: 37, 0.41: 43, 0.42: 49, 0.43: 58, 0.44: 69}
    f_std = {0.1: 2, 0.11: 2, 0.12: 1, 0.13: 1, 0.14: 1, 0.15: 1, 0.16: 1, 0.17: 1, 0.18: 1, 0.19: 1, 0.2: 1, 0.21: 1, 0.22: 1, 0.23: 0, 0.24: 0, 0.25: 0, 0.26: 0, 0.27: 0, 0.28: 1, 0.29: 1, 0.3: 1, 0.31: 1, 0.32: 1, 0.33: 1, 0.34: 1, 0.35: 1, 0.36: 1, 0.37: 1, 0.38: 2, 0.39: 2, 0.4: 2, 0.41: 2, 0.42: 3, 0.43: 3, 0.44: 4}
    table_g = []
    table_f = []
    for (k1,v1), (_, v2)  in zip(g_avg.items(), g_std.items()):
        table_g.append([k1,v1,v2])
    print(tabulate(table_g, headers=["beta", "avg", "std"]))
    print()
    for (k1,v1), (_, v2)  in zip(f_avg.items(), f_std.items()):
        table_f.append([k1,v1,v2])
    print(tabulate(table_f, headers=["beta", "avg", "std"]))

# GIVEN FUNCTIONS 
f = lambda x: 2*x**2 + 3*x - 1
g = lambda x: 1 - 0.6*np.exp(-x[0]**2 - x[1]**2) - 0.4*np.exp(-(x[0]+1.75)**2 - (x[1]-1)**2)

# print(test_beta_2d_avg(g, 0.00001))
# print(test_beta_2d_std(g, 0.00001))
# plot_2d(g, np.array([1,1]), 0.1, 0.00001)
# plot_1d(f, np.array([5]), 0.25, 0.00001)
# print(test_beta_1d(f, 0.00001))
# print(test_beta_1d_std(f, 0.00001))
# make_tables()