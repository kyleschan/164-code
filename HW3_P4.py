import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[5, 2],
              [2, 1]])

b = np.array([3, 1])


def f(x):
    y = 0.5 * np.dot(x, Q.dot(x)) - np.dot(b, x)
    return y


def grad_f(x):
    y = Q.dot(x) - b
    return y


def conjugate_gradient_method(initial_x, iterations):
    f_plot = np.zeros([iterations, 1])
    x = initial_x
    f_plot[0] = f(x)
    direction = grad_f(x)
    n = 1
    while n < iterations:
        print("This is iteration", n)
        alpha = -np.dot(grad_f(x), direction) / np.dot(direction, Q.dot(direction))
        x = x + alpha * direction
        beta = np.dot(grad_f(x), Q.dot(direction)) / np.dot(direction, Q.dot(direction))
        direction = -grad_f(x) + beta * direction
        print("x is", x, "and f(x) is", f(x))
        f_plot[n] = f(x)
        n = n + 1

    return f_plot


def gradient_descent_method(initial_x, iterations):
    f_plot = np.zeros([iterations, 1])
    x = initial_x
    f_plot[0] = f(x)
    direction = grad_f(x)
    n = 1
    while n < iterations:
        print("This is iteration", n)
        alpha = np.dot(grad_f(x), grad_f(x)) / np.dot(grad_f(x), Q.dot(grad_f(x)))
        x = x - alpha * grad_f(x)
        print("x is", x, "and f(x) is", f(x))
        f_plot[n] = f(x)
        n = n + 1

    return f_plot


n = 5
x_plot = np.zeros([n, 1])
for i in range(n):
    x_plot[i] = i

cg_plot = conjugate_gradient_method([0, 0], n)
gd_plot = gradient_descent_method([0, 0], n)
plt.plot(x_plot, cg_plot, 'ro', x_plot, gd_plot, 'bo')
plt.xticks(np.arange(0, 5, step=1))
plt.xlabel("Iteration k")
plt.ylabel("f(k)")
plt.legend(("Conjugate Gradient", "Gradient Descent"))
plt.show()
