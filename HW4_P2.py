import numpy as np
import matplotlib.pyplot as plt

init_x = np.array([2, 2])


def f(x):
    y = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    return y


def grad_f(x):
    y = np.array([2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0] ** 2)])
    return y


def secant_ls(x, d):
    init_ddot = np.dot(grad_f(x), d)
    curr_ddot = init_ddot
    alpha = 0.001
    curr_alpha = 0
    epsilon = 1e-4
    max_its = 100
    i = 1
    while abs(curr_ddot) > epsilon * abs(init_ddot) and i < max_its:
        prev_alpha = curr_alpha
        curr_alpha = alpha
        prev_ddot = curr_ddot
        curr_ddot = np.dot(grad_f(x + curr_alpha * d), d)
        alpha = (curr_ddot * prev_alpha - prev_ddot * curr_alpha) / (curr_ddot - prev_ddot)
        i = i + 1

    return alpha


def conjugate_gradient_method(initial_x, iterations, update_method):
    # Initialize everything
    f_plot = np.zeros([iterations, 1])
    x = initial_x
    x_prev = initial_x / 2
    direction = grad_f(x)
    beta = 0
    n = 0
    while n < iterations:
        # Update parameters
        print("This is iteration", n + 1)
        alpha = secant_ls(x, direction)
        x_prev = x
        grad_prev = grad_f(x_prev)
        f_prev = f(x_prev)
        x = x + alpha * direction
        grad = grad_f(x)

        # Update beta depending on update type given
        if update_method == 0:  # Hestenes-Stiefel
            beta = np.dot(grad, grad - grad_prev) / np.dot(direction, grad - grad_prev)
        elif update_method == 1:    # Polak-Ribiere
            beta = np.dot(grad, grad - grad_prev) / np.dot(grad_prev, grad_prev)
        elif update_method == 2:    # Fletcher-Reeves
            beta = beta = np.dot(grad, grad) / np.dot(grad_prev, grad_prev)
        elif update_method == 3:    # Powell
            beta = max(0, np.dot(grad, grad - grad_prev) / np.dot(grad_prev, grad_prev))

        # Revert to negative gradient every 6 iterations
        if n % 6 == 0:
            direction = -grad_f(x)
        else:
            direction = -grad_f(x) + beta * direction

        print("x is", x, "and f(x) is", f(x))
        f_plot[n] = f(x)
        n = n + 1

    return f_plot


n = 20
x_plot = np.zeros([n, 1])
for i in range(n):
    x_plot[i] = i

cg_hs_plot = conjugate_gradient_method(init_x, n, 0)
cg_pr_plot = conjugate_gradient_method(init_x, n, 1)
cg_fr_plot = conjugate_gradient_method(init_x, n, 2)
cg_p_plot = conjugate_gradient_method(init_x, n, 3)

plt.plot(x_plot, cg_hs_plot, 'r-')
plt.plot(x_plot, cg_pr_plot, 'b-')
plt.plot(x_plot, cg_fr_plot, 'm-')
plt.plot(x_plot, cg_p_plot, 'g-')
plt.xticks(np.arange(1, 20, step=1))
plt.xlabel("Iteration k")
plt.ylabel("f(k)")
plt.legend(("Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Powell"))
plt.savefig(f'HW4_P2_plot.png')
plt.show()


