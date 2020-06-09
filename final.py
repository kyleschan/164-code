import numpy as np
import matplotlib.pyplot as plt

init_x = np.array([-1, 1])
init_H = np.identity(2)

Q = np.array([[2, 3],
              [3, 10]])
b = np.array([-1, 1])


def f(x):
    y = 0.5 * np.dot(x, Q.dot(x)) - np.dot(b, x)
    return y


def grad_f(x):
    y = Q.dot(x) - b
    return y


def quasi_newton_method(initial_x, iterations, update_method, H):
    # Initialize everything
    f_plot = np.zeros([iterations, 1])
    x = initial_x
    f_plot[0] = f(x)
    grad = grad_f(x)
    direction = -np.dot(H, grad)
    n = 1
    while n < iterations:
        # Update parameters
        print("This is iteration", n)
        x_prev = x
        alpha = -np.dot(grad_f(x), direction) / np.dot(direction, Q.dot(direction))
        x = x_prev + alpha * direction
        grad_prev = grad
        grad = grad_f(x)
        delta_grad = grad - grad_prev
        delta_x = alpha * direction
        print("H is", H)
        # Update beta depending on update type given
        if update_method == 0:  # Rank One Correction Method
            H = H + np.outer((delta_x - np.dot(H, delta_grad)), delta_x - np.dot(H, delta_grad)) / np.dot(
                delta_grad, delta_x - np.dot(H, delta_grad))
        elif update_method == 1:  # BFGS
            H = H + (1 + np.dot(delta_grad,
                                H.dot(delta_grad)) / np.dot(delta_x, delta_grad)) * np.outer(delta_x,
                                                                                             delta_x) / np.dot(delta_x, delta_grad) - (np.dot(H, np.outer(delta_grad, delta_x)) + np.transpose(np.dot(H, np.outer(delta_grad, delta_x)))) / np.dot(delta_x, delta_grad)
        direction = -np.dot(H, grad)
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
n = 3
x_plot = np.zeros([n, 1])
for i in range(n):
    x_plot[i] = i

gd_plot = gradient_descent_method(init_x, n)
qn_bfgs_plot = quasi_newton_method(init_x, n, 1, init_H)

plt.plot(x_plot, gd_plot, 'r-')
plt.plot(x_plot, qn_bfgs_plot, 'b-')
plt.xticks(np.arange(1, 3, step=1))
plt.xlabel("Iteration k")
plt.ylabel("f(k)")
plt.legend(("Steepest Descent", "BFGS"))
plt.savefig(f'final_plot.png')
plt.show()