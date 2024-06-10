import numpy as np
import numdifftools as nd

def gradient_descent(problem, x0, beta, eps):
    x_t = x0
    iterations = 0
    while np.any(abs(nd.Gradient(problem)(x_t)) >= eps):
        x_t = x_t - (beta*np.array(nd.Gradient(problem)(x_t)))
        iterations += 1
    return x_t, iterations, problem(x_t)

# Przykładowe wywołanie dla dobrych parametrów
if __name__ == "__main__":
    f = lambda x: 2*x**2 + 3*x - 1
    g = lambda x: 1 - 0.6*np.exp(-x[0]**2 - x[1]**2) - 0.4*np.exp(-(x[0]+1.75)**2 - (x[1]-1)**2)
    print(gradient_descent(f, np.array([35]), 0.25, 0.00001))
    print(gradient_descent(g, np.array([1, 1]), 0.95, 0.00001))
