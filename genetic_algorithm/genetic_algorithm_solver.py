from template import Solver
import random
import numpy as np

def generate_random_individual():
    return np.random.choice([0, 1], size=100).tolist()

def generate_population(size):
    population = []
    for _ in range(size):
        population.append(generate_random_individual())
    return population

def evaluate(x):
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
    return sum(sum(row) for row in avaliable)


class GeneticAlgorithm(Solver):
    def get_parameters(self):
        return self.parameters
    
    def solve(self, q, start_population, *args, **kwargs):
        self.population = start_population
        self.parameters = kwargs
        size = len(start_population)
    
        evaluated_population = {}
        list_of_best = []

        for i in range(size): # first evaluation and find best
            evaluated_population[i] = evaluate(self.population[i])
        best_key = max(evaluated_population, key=evaluated_population.get)
        best_individual = self.population[best_key]
        best_individual_eval = evaluated_population[best_key]
        list_of_best.append(best_individual_eval)

        for _ in range(self.parameters["t_max"]):
            # selection
            total = sum(evaluated_population.values())
            new_population = random.choices(self.population, weights=list([v/total for _,v in evaluated_population.items()]), k=size)

            # one-point crossover
            for _ in range(size//2):
                if random.random() < self.parameters["pc"]:
                    x1 = new_population.pop(0)
                    x2 = new_population.pop(0)
                    cut = random.randint(1, len(x1)-1)
                    new_x1 = x1[:cut] + x2[cut:]
                    new_x2 = x1[cut:] + x2[:cut]
                    new_population.append(new_x1)
                    new_population.append(new_x2)
                else:
                    x1 = new_population.pop(0)
                    x2 = new_population.pop(0)
                    new_population.append(x1)
                    new_population.append(x2)               

            # mutation
            for i in range(len(new_population)):
                for j in range(len(new_population[i])):
                    if random.random() < self.parameters["pm"]:
                        if new_population[i][j] == 1:
                            new_population[i][j] = 0
                        else:
                            new_population[i][j] = 1

            # succession
            self.population = new_population

            # evaluation
            for i in range(size): 
                evaluated_population[i] = q(self.population[i])

            # finding best
            best_key = max(evaluated_population, key=evaluated_population.get)
            if evaluated_population[best_key] > best_individual_eval:
                best_individual = self.population[best_key]
                best_individual_eval = evaluated_population[best_key]
            list_of_best.append(best_individual_eval)
        
        return best_individual, best_individual_eval, list_of_best

if __name__ == "__main__":
    x = GeneticAlgorithm()
    population = generate_population(1000)
    ind, eva, _ = x.solve(evaluate, population, t_max=100, pc=0.9, pm=0.01)
    print(ind)
    print(eva)
    print(x.get_parameters())
