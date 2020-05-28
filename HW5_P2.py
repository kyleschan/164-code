import numpy as np
import matplotlib.pyplot as plt

init_x = np.array([-2, 2])
init_H = np.identity(2)


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


def quasi_newton_method(initial_x, iterations, update_method, H):
    # Initialize everything
    f_plot = np.zeros([iterations, 1])
    x = initial_x
    f_plot[0] = f(x)
    grad = grad_f(x)
    direction = -np.dot(H, grad)
    n = 0
    while n < iterations:
        # Update parameters
        print("This is iteration", n + 1)
        x_prev = x
        alpha = secant_ls(x_prev, direction)
        x = x_prev + alpha * direction
        grad_prev = grad
        grad = grad_f(x)
        delta_grad = grad - grad_prev
        delta_x = alpha * direction
        print("H is", H)
        # Revert to negative gradient every 6 iterations
        if n % 6 == 0:
            direction = -grad
        # Update beta depending on update type given
        elif update_method == 0:  # Rank One Correction Method
            H = H + np.outer((delta_x - np.dot(H, delta_grad)), delta_x - np.dot(H, delta_grad)) / np.dot(
                delta_grad, delta_x - np.dot(H, delta_grad))
        elif update_method == 1:  # DFP
            H = H + np.outer(delta_x, delta_x) / np.dot(delta_x, delta_grad) - np.outer(
                np.dot(H, delta_grad),
                np.dot(H, delta_grad)) / np.dot(
                delta_grad, np.dot(H, delta_grad))
        elif update_method == 2:  # BFGS
            H = H + (1 + np.dot(delta_grad,
                                H.dot(delta_grad)) / np.dot(delta_x, delta_grad)) * np.outer(delta_x,
                                                                                                 delta_x) / np.dot(delta_x, delta_grad) - (np.dot(H, np.outer(delta_grad, delta_x)) + np.transpose(np.dot(H, np.outer(delta_grad, delta_x)))) / np.dot(delta_x, delta_grad)
        direction = -np.dot(H, grad)
        print("x is", x, "and f(x) is", f(x))
        f_plot[n] = f(x)
        n = n + 1

    return f_plot


n = 20
x_plot = np.zeros([n - 1, 1])
for i in range(n - 1):
    x_plot[i] = i

qn_r1c_plot = quasi_newton_method(init_x, n, 0, init_H)
qn_dfp_plot = quasi_newton_method(init_x, n, 1, init_H)
qn_bfgs_plot = quasi_newton_method(init_x, n, 2, init_H)

plt.plot(x_plot, qn_r1c_plot[1:], 'r-')
plt.plot(x_plot, qn_dfp_plot[1:], 'b-')
plt.plot(x_plot, qn_bfgs_plot[1:], 'm-')
plt.xticks(np.arange(1, 20, step=1))
plt.xlabel("Iteration k")
plt.ylabel("f(k)")
plt.legend(("Rank-1 Correction", "DFP", "BFGS"))
plt.savefig(f'HW4_P3_plot.png')
plt.show()
