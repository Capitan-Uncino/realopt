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

#projected gradient descent with Armijo line-search
def armijo_step(sigma, mu, lamb, w0, gradient, alpha0=1.0, max_iter=1000):
    f_w0 = 0.5 * w0.T @ sigma @ w0 - lamb * w0.T @ mu
    l = 0

    while l < max_iter:
        step = (0.5 ** l) * alpha0
        w_new = project_simplex(w0 - step * gradient)
        f_w_new = 0.5 * w_new.T @ sigma @ w_new - lamb * w_new.T @ mu

        if f_w0 - f_w_new >= (step * 0.5) * gradient.T @ (w0 - w_new):
            return step

        l += 1
    return (0.5 ** l) * alpha0

def projected_gradient_descent_armijo(sigma, lamb, mu, w0, max_iter=1000):
    w = w0.copy()
    f_values = []

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        step_size = armijo_step(sigma, mu, lamb, w, gradient)
        w = project_simplex(w - step_size * gradient)

        f_value  = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)

    return w, f_values
