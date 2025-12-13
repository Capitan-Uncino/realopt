def compute_subgradient(sigma, mu, lamb, c, w_k, w_prev):
    grad = sigma @ w_k - lamb * mu

    l1_subgrad = np.sign(w_k - w_prev)
    l1_subgrad[w_k == w_prev] = 0  

    return grad + c * l1_subgrad

def projected_subgradient_descent(sigma, lamb, mu, w0, w_prev, c, initial_step_size, max_iter = 1000):
    w = w0.copy()

    f_values = []

    for i in range(max_iter):
        gradient = compute_subgradient(sigma, mu, lamb, c,  w, w_prev)
        w = project_simplex(w - (initial_step_size)/sqrt(i+1) * gradient)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu + c * np.linalg.norm(w-w_prev, 1) 
        f_values.append(f_value)
    return w, f_values
