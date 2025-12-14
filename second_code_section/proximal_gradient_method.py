import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv("../all_stocks_5yr.csv")
data = data.sort_values(["Name", "date"])
data["prev_close"] = data.groupby("Name")["close"].shift(1)
data["return"] = np.log(data["close"] / data["prev_close"])

returns = data.pivot(index="date", columns="Name", values="return").dropna()
mu = 252 * returns.mean().values
sigma = 252 * returns.cov().values
sigma += 1e-6 * np.eye(len(mu))

# Define the projection onto the simplex
def project_simplex(v):
    """Projection of a vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

# Define gradient and soft threshold
def compute_gradient(sigma, mu, lamb, w_k):
    return sigma @ w_k - lamb * mu

def soft_threshold(w, threshold):
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

# Projected gradient descent
def projected_gradient_descent(sigma, lamb, mu, w0, w_prev, c, step_size, max_iter = 1000):
    w = w0.copy()
    f_values = []

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        w = project_simplex( soft_threshold(w - step_size * gradient - w_prev, c*step_size) + w_prev )
        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu + c * np.linalg.norm(w - w_prev, 1)
        f_values.append(f_value)
    
    return w, f_values

# Example parameters
n_assets = len(mu)
w0 = np.ones(n_assets) / n_assets
w_prev = w0.copy()
lamb = 0.5
c = 0.1
step_size = 0.01
max_iter = 200

# Run the algorithm
w_opt, f_values = projected_gradient_descent(sigma, lamb, mu, w0, w_prev, c, step_size, max_iter)

# Plot the objective function over iterations
plt.figure(figsize=(8,5))
plt.plot(f_values, label='Objective Function')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Objective Function vs Iteration")
plt.grid(True)
plt.legend()
plt.show()




