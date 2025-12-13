import random

def projected_randomized_coordinate_descent(sigma, lamb, mu, w, step_size, max_iter = 1000):
    n = len(w)

    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]

    for i in range(max_iter):
        i_k = random.randint(0, n-1)
        grad_ik = sigma.values[i_k, :] @ w - lamb * mu[i_k]
        
        w[i_k] = w[i_k] - step_size * grad_ik
        w = project_simplex(w)
    
        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)
    return w, f_values
