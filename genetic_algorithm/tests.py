from genetic_algorithm_solver import GeneticAlgorithm, generate_population, evaluate
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tabulate import tabulate

def avaliable(x):
    n = int(len(x) ** (0.5))
    checked = [[False] * n for _ in range(n)]
    avaliable = [[False] * n for _ in range(n)]
    to_check = []
    for i in range(n): # accessible from outside
        to_check.append((i, 0))
        to_check.append((i, n - 1))
        to_check.append((0, i))
        to_check.append((n - 1, i))
    while to_check:
        i, j = to_check.pop()
        if checked[i][j]:
            continue
        checked[i][j] = True
        if x[i * n + j]: # parking spot
            avaliable[i][j] = True
        else: # road
            if i > 0:
                to_check.append((i - 1, j))
            if i < n - 1:
                to_check.append((i + 1, j))
            if j > 0:
                to_check.append((i, j - 1))
            if j < n - 1:
                to_check.append((i, j + 1))
    return avaliable

def change_to_red(x, data):
    places = avaliable(x)
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i][j] != places[i][j]:
                data[i][j] = 0.666666
    return data

def test_random_population():
    x = GeneticAlgorithm()
    evals = []
    for i in range(100):
        population = generate_population(1000)
        _, eval, _ = x.solve(evaluate, population, t_max=0, pc=1, pm=1)
        evals.append(eval)
    avg = np.average(evals)
    med = np.median(evals)
    std = np.std(evals)
    print(f"avg = {avg}; med = {med}; std = {std}")


def test_pc_parameter(population, t_max, pm):
    all_evaluations = []
    pcs = []
    x = GeneticAlgorithm()
    for i in range(85, 96, 1):
        pc = i/100
        temp_eval = []
        for _ in range(25):
            _, eval, _ = x.solve(evaluate, population, t_max=t_max, pc=pc, pm=pm)
            temp_eval.append(eval)
        pcs.append(pc)
        all_evaluations.append(temp_eval)
    avg = []
    med = []
    std = []
    for row in all_evaluations:
        avg.append(np.average(row))
        med.append(np.median(row))
        std.append(np.std(row))
    print(tabulate({"pc": pcs, "avg": avg, "med": med,"std": std}, headers="keys"))

def draw_finding_best(population, t_max, pc, pm):
    plt.rcParams.update({'font.size': 22})
    x = GeneticAlgorithm()
    _, _, bests = x.solve(evaluate, population, t_max=t_max, pc=pc, pm=pm)
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.stairs(bests, linewidth=2.5)
    ax.set(xlim=(0, len(bests)), xticks=np.arange(1,1),
        ylim=(50, 63), yticks=np.arange(51, 63))
    plt.show()

def draw_population(x):
    data = []
    for i in range(1,11):
        data.append(x[10*(i-1):10*i])
    data = change_to_red(x, data)
    data = pd.DataFrame(np.array(data))
    sns.heatmap(data, annot=True, annot_kws={'color':'black','fontsize':12}, cmap=ListedColormap(["white", "red", "green"]), fmt='.0f', cbar=False, square=True)
    plt.show()

# x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# test_random_population()
test_pc_parameter(generate_population(1000), 100, 0.01)
# draw_population(x)
# draw_finding_best(generate_population(1000), 100, 0.9, 0.01)
    