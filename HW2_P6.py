import numpy as np
import math


def f(x):
    y = 8 * math.exp(1 - x) + 7 * np.log(x)
    return y


def f_prime(x):
    y_prime = 7 / x - 8 * math.exp(1 - x)
    return y_prime


def f_double_prime(x):
    y_double_prime = 8 * math.exp(1 - x) - 7 / (x ** 2)
    return y_double_prime


def golden_section(a, b, uncertainty):
    if a == b:
        print("[", a, ",", b, "]")
    if b > a:
        a, b = b, a
    golden = (1 + 5 ** 0.5) / 2
    rho = golden - 1
    left = rho * b + (1 - rho) * a
    f_left = f(left)
    n = 1
    while abs(b - a) > uncertainty:
        print("This is iteration", n)
        right = rho * a + (1 - rho) * b
        f_right = f(right)
        if f_right < f_left:
            b, left, f_left = left, right, f_right
        else:
            a, b = b, right
        if a < b:
            print("[", a, ",", b, "]")
        else:
            print("[", b, ",", a, "]")
        n = n + 1


def bisection_method(a, b, uncertainty):
    if a == b:
        print("[", a, ",", b, "]")
    if b > a:
        a, b = b, a
    f_prime_a = f_prime(a)
    f_prime_b = f_prime(b)
    if f_prime_a == 0:
        return f(a)
    if f_prime_b == 0:
        return f(b)
    n = 1
    while abs(b - a) > uncertainty:
        print("This is iteration", n)
        x = (a + b) / 2
        deriv_x = f_prime(x)
        if deriv_x == 0:
            print("[", x, ",", x, "]")
        elif np.sign(deriv_x) == np.sign(f_prime_a):
            a = x
        else:
            b = x

        if a < b:
            print("[", a, ",", b, "]")
        else:
            print("[", b, ",", a, "]")
        n = n + 1


def Newton_method(a, b, uncertainty):
    if a == b:
        print("x is", a)
    n = 1
    x = (a + b) / 2
    while True:
        print("This is iteration", n)
        deriv_x = f_prime(x)
        second_deriv_x = f_double_prime(x)
        prev_x = x
        x = x - deriv_x / second_deriv_x

        if abs(prev_x - x) < uncertainty:
            print("x is", x)
            return
        print("x is", x)
        n = n + 1


def secant_method(a, b, uncertainty):
    if a == b:
        print("x is", a)
    n = 1
    x = 2 * (a + b) / 3
    prev_x = (a + b) / 3

    while True:
        print("This is iteration", n)
        deriv_x = f_prime(x)
        deriv_prev_x = f_prime(prev_x)
        prev_x, x, deriv_prev_x = x, x - ((x - prev_x) / (deriv_x - deriv_prev_x)) * deriv_x, deriv_x
        if abs(prev_x - x) < uncertainty:
            print("x is", x)
            return
        print("x is", x)
        n = n + 1


print("Golden Section Method:")
golden_section(1, 2, 0.23)
print("\n")
print("Bisection Method:")
bisection_method(1, 2, 0.23)
print("\n")
print("Newton's Method:")
Newton_method(1, 2, 0.23)
print("\n")
print("Secant Method:")
secant_method(1, 2, 0.23)