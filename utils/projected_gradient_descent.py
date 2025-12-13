def compute_gradient(sigma, mu, lamb, w_k):
    return sigma @ w_k - lamb * mu

# projected gradient descent with fixed step_size
def projected_gradient_descent(sigma, lamb, mu, w0, step_size, max_iter = 1000):
    w = w0.copy()

    f_values = []

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        w = project_simplex(w - step_size * gradient)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)
    return w, f_values
        f_value  = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)

    return w, f_values
