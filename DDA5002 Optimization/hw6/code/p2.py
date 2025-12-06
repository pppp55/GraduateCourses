# -*- coding: utf-8 -*-
"""
DDA 5002 Homework 6, Problem 2
Implementation of Gradient Descent with different step sizes
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic function f(x) = x^T A x
A = np.array([[2, 0], [0, 5]])


def f(x):
    """Objective function"""
    return x.T @ A @ x


def grad_f(x):
    """Gradient of f: 2*A*x"""
    return 2 * A @ x


def constant_step_descent(x0, alpha, max_iter=10):
    """Gradient descent with constant step size"""
    x = x0.copy()
    trajectory = [x.copy()]
    values = [f(x)]

    for k in range(max_iter):
        g = grad_f(x)
        x = x - alpha * g
        trajectory.append(x.copy())
        values.append(f(x))

    return np.array(trajectory), np.array(values)


def exact_line_search_descent(x0, max_iter=10):
    """Gradient descent with exact line search"""
    x = x0.copy()
    trajectory = [x.copy()]
    values = [f(x)]

    for k in range(max_iter):
        g = grad_f(x)
        d = -g  # descent direction

        numerator = d.T @ A @ x
        denominator = d.T @ A @ d
        if abs(denominator) < 1e-14:
            break
        alpha = -numerator / denominator

        x = x + alpha * d
        trajectory.append(x.copy())
        values.append(f(x))

    return np.array(trajectory), np.array(values)


def backtracking_line_search_descent(x0, gamma=0.5, sigma=0.2, max_iter=10):
    """Gradient descent with backtracking line search (Armijo rule)"""
    x = x0.copy()
    trajectory = [x.copy()]
    values = [f(x)]

    for k in range(max_iter):
        g = grad_f(x)
        norm_g_sq = g.T @ g

        # Start with alpha = 1
        alpha = 1.0
        while True:
            x_new = x - alpha * g
            left = f(x_new)
            right = f(x) - gamma * alpha * norm_g_sq

            if left <= right:
                break
            else:
                alpha *= sigma  # alpha = sigma * alpha
                if alpha < 1e-10:  # prevent infinite loop
                    break

        x = x_new
        trajectory.append(x.copy())
        values.append(f(x))

    return np.array(trajectory), np.array(values)


# Initial point
x0 = np.array([1.0, 1.0])

# Run all three methods
print("Running Gradient Descent for 10 iterations...\n")

# Method 1: Constant step size 0.1
traj_const, f_const = constant_step_descent(x0, alpha=0.1, max_iter=10)
print("Constant Step (ω=0.1):")
print(f"  Final x: {traj_const[-1]}")
print(f"  Final f: {f_const[-1]:.6f}\n")

# Method 2: Exact line search
traj_exact, f_exact = exact_line_search_descent(x0, max_iter=10)
print("Exact Line Search:")
print(f"  Final x: {traj_exact[-1]}")
print(f"  Final f: {f_exact[-1]:.6f}\n")

# Method 3: Backtracking line search
traj_back, f_back = backtracking_line_search_descent(
    x0, gamma=0.5, sigma=0.2, max_iter=10
)
print("Backtracking Line Search (γ=0.5, σ=0.2):")
print(f"  Final x: {traj_back[-1]}")
print(f"  Final f: {f_back[-1]:.6f}\n")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(f_const, label="Constant Step (α=0.1)", marker="o")
plt.semilogy(f_exact, label="Exact Line Search", marker="s")
plt.semilogy(f_back, label="Backtracking (γ=0.5, σ=0.2)", marker="^")
plt.xlabel("Iteration k")
plt.ylabel("f(x_k) (log scale)")
plt.title("Convergence of Gradient Descent with Different Step Sizes")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()

# Save plot
plt.savefig("p2.png")
plt.show()

# Output the first few iterations for verification
print("First few function values (Constant):", [f"{v:.4f}" for v in f_const[:5]])
print("First few function values (Exact):", [f"{v:.4f}" for v in f_exact[:5]])
print("First few function values (Back):", [f"{v:.4f}" for v in f_back[:5]])
