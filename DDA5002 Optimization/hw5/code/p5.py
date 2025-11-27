import math


# Bisection method (on derivative g(x) = e^x - 2)
def bisection():
    a, b = 0.0, 1.0
    iterations = 0
    while b - a > 1e-4:
        c = (a + b) / 2.0
        g_c = math.exp(c) - 2
        g_a = math.exp(a) - 2
        if g_c == 0:
            break
        elif g_a * g_c < 0:
            b = c
        else:
            a = c
        iterations += 1
    return (a + b) / 2.0, iterations


# Golden Section method (minimization of f(x) = e^x - 2x)
def golden_section():
    a, b = 0.0, 1.0
    phi = (math.sqrt(5) - 1) / 2  # Golden ratio conjugate
    iterations = 0
    while b - a > 1e-4:
        c = a + (1 - phi) * (b - a)
        d = a + phi * (b - a)
        f_c = math.exp(c) - 2 * c
        f_d = math.exp(d) - 2 * d
        if f_c < f_d:
            b = d
        else:
            a = c
        iterations += 1
    return (a + b) / 2.0, iterations


# Run methods
sol_bisect, iter_bisect = bisection()
sol_golden, iter_golden = golden_section()

print(f"Bisection Method: Solution = {sol_bisect:.4f}, Iterations = {iter_bisect}")
print(f"Golden Section Method: Solution = {sol_golden:.4f}, Iterations = {iter_golden}")
