def projected_gradient_descent(sigma, lamb, mu, w0, step_size, max_iter = 1000):
    w = w0.copy()

    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        w = project_simplex(w - step_size * gradient)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)
    return w, f_values
  

def projected_gradient_descent_armijo(sigma, lamb, mu, w, max_iter=1000, alpha0=1.0, max_l=50):
    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]
    
    alpha_k = alpha0
    
    for k in range(max_iter):
        g_k = sigma @ w - lamb * mu
        
        l = 0
        while l < max_l:
            step = (0.5 ** l) * alpha_k
            w_new = project_simplex(w - step * g_k)

            f_w = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
            f_w_new = 0.5 * w_new.T @ sigma @ w_new - lamb * w_new.T @ mu
            if f_w - f_w_new >= 0.5 * g_k.T @ (w - w_new):
                break
            l += 1
        
        w = w_new
        f_w = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_w)
        
        alpha_k = (0.5 ** max(l-1, 0)) * alpha_k
    return w, f_values
