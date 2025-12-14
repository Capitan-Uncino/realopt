def compute_gradient(sigma, mu, lamb, w_k):
    return sigma @ w_k - lamb * mu


def soft_threshold(w, threshold):
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)


def projected_gradient_descent(sigma, lamb, mu, w0, w_prev, c, step_size, max_iter = 1000):
    w = w0.copy()

    f_values = []

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        w = project_simplex( soft_threshold(w - step_size * gradient-w_prev, c*step_size)+ w_prev)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu + c * np.linalg.norm(w-w_prev, 1) 
        f_values.append(f_value)
    return w, f_values
