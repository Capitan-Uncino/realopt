
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv("all_stocks_5yr.csv")
data = data.sort_values(["Name", "date"])
data["prev_close"] = data.groupby("Name")["close"].shift(1)
data["return"] = np.log(data["close"] / data["prev_close"])

returns = data.pivot(index="date", columns="Name", values="return").dropna()
mu = 252 * returns.mean().values
sigma = 252 * returns.cov().values
sigma += 1e-6 * np.eye(len(mu))

def project_simplex(v):
    """Projection of a vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(n) + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def compute_gradient(sigma, mu, lamb, w_k):
    return sigma @ w_k - lamb * mu


# Example parameters
n_assets = len(mu)
w0 = np.ones(n_assets) / n_assets
w_prev = w0.copy()
lamb = 0.5
c = 0.1 
eigenvalues, eigenvectors = np.linalg.eig(sigma) 
L = np.max(eigenvalues)
#step_size = 1/L
step_size = 0.01

max_iter = 200


#___________________________________________________________ PROJECTED GRADIENT DESCENT

def projected_gradient_descent(sigma, lamb, mu, w0, step_size, max_iter = 1000):
    w = w0.copy()

    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        w = project_simplex(w - step_size * gradient)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)
    return w, f_values
  

w_pgd, f_values_pgd = projected_gradient_descent(sigma, lamb, mu, w0, step_size, max_iter)

#___________________________________________________________ PROJECTED GRADIENT DESCENT ARMIJO

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



w_pgda, f_values_pgda = projected_gradient_descent_armijo(sigma, lamb, mu, w0, max_iter)



#___________________________________________________________ PROJECTED GRADIENT DESCENT MOMENTUM

def projected_gradient_descent_momentum(sigma, lamb, mu, w, step_size, max_iter = 1000 , beta = 0.9):
    m = np.zeros(len(w))
    
    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]
    
    for i in range(max_iter):
        grad = compute_gradient(sigma, lamb, mu, w)


        m = beta * m + grad
        w = w - step_size * m

        
        w = project_simplex(w)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)
    return w, f_values


w_pgdm, f_values_pgdm = projected_gradient_descent_momentum(sigma, lamb, mu, w0, step_size, max_iter)



#_____________________________________________________________  PROJECTED COORDINATE DESCENT

import random

def projected_randomized_coordinate_descent(sigma, lamb, mu, w, step_size, max_iter = 1000):
    n = len(w)

    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]

    for i in range(max_iter):
        i_k = random.randint(0, n-1)
        grad_ik = sigma[i_k, :] @ w - lamb * mu[i_k]
        
        w[i_k] = w[i_k] - step_size * grad_ik
        w = project_simplex(w)
    
        if i%n == 0:
            f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
            f_values.append(f_value)
    return w, f_values


w_pcd, f_values_pcd = projected_randomized_coordinate_descent(sigma, lamb, mu, w0, step_size, max_iter*len(w0))

#_____________________________________________________________  PROJECTED SUBGRADIENT DESCENT
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
        w = project_simplex(w - (initial_step_size)/np.sqrt(i+1) * gradient)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu + c * np.linalg.norm(w-w_prev, 1) 
        f_values.append(f_value)
    return w, f_values

w_sub, f_values_sub = projected_subgradient_descent(sigma, lamb, mu, w0, w_prev, c, step_size, max_iter)




#_____________________________________________________________  PROXIMAL GRADIEN DESCENT


def soft_threshold(w, threshold):
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

# Projected gradient descent
def proximal_gradient_descent(sigma, lamb, mu, w0, w_prev, c, step_size, max_iter = 1000):
    w = w0.copy()
    f_values = []

    for i in range(max_iter):
        gradient = compute_gradient(sigma, mu, lamb, w)
        w = project_simplex( soft_threshold(w - step_size * gradient - w_prev, c*step_size) + w_prev )
        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu + c * np.linalg.norm(w - w_prev, 1)
        f_values.append(f_value)
    
    return w, f_values



# Run the algorithm
w_prox, f_values_prox = proximal_gradient_descent(sigma, lamb, mu, w0, w_prev, c, step_size, max_iter)


#_____________________________________________________________  INTERIOR POINT METHOD


def make_simplex_param(n):
    """Uses invariance under affine transforms to enforce equality constraint"""
    c = np.ones(n) / n
    A = np.zeros((n, n-1))
    for i in range(n-1):
        A[i, i] = 1.0
        A[n-1, i] = -1.0
    return c, A

def G_value(y, c, A, Sigma, mu, t, lam, wprev, c_l1):
    yw = y[:A.shape[1]]
    z  = y[A.shape[1]:]
    w = c + A @ yw
    d = w - wprev
    if np.any(w <= 0):
        raise ValueError("w out of domain")
    if np.any(z - d <= 0) or np.any(z + d <= 0):
        raise ValueError("z out of domain")
    s = t - 0.5 * w @ Sigma @ w + lam * w @ mu - c_l1 * np.sum(z)
    if s <= 0:
        raise ValueError("quadratic epigraph violated")
    return (
        -np.log(s)
        -np.sum(np.log(w))
        -np.sum(np.log(z - d))
        -np.sum(np.log(z + d))
    )

def G_grad(y, c, A, Sigma, mu, t, lam, wprev, c_l1):
    yw = y[:A.shape[1]]
    z  = y[A.shape[1]:]
    w = c + A @ yw
    d = w - wprev
    s = t - 0.5 * w @ Sigma @ w + lam * w @ mu - c_l1 * np.sum(z)
    v = Sigma @ w - lam * mu
    grad_w = (
        v / s
        - 1.0 / w
        + 1.0 / (z - d)
        - 1.0 / (z + d)
    )
    grad_z = (
        c_l1 / s
        - 1.0 / (z - d)
        - 1.0 / (z + d)
    )
    return np.concatenate([A.T @ grad_w, grad_z])

def G_hess(y, c, A, Sigma, mu, t, lam, wprev, c_l1):
    yw = y[:A.shape[1]]
    z  = y[A.shape[1]:]
    w = c + A @ yw
    d = w - wprev
    s = t - 0.5 * w @ Sigma @ w + lam * w @ mu - c_l1 * np.sum(z)
    v = Sigma @ w - lam * mu
    # ww block
    H_ww = (
        Sigma / s
        + np.outer(v, v) / s**2
        + np.diag(1.0 / w**2)
        + np.diag(1.0 / (z - d)**2 + 1.0 / (z + d)**2)
    )
    # zz block
    H_zz = np.diag(
        1.0 / (z - d)**2
        + 1.0 / (z + d)**2
        + (c_l1**2) / s**2
    )
    # wz block
    H_wz = np.diag(
        -1.0 / (z - d)**2
        + 1.0 / (z + d)**2
        + c_l1 * v / s**2
    )
    # reduce ww block
    H_yy = A.T @ H_ww @ A
    H_yz = A.T @ H_wz
    return np.block([
        [H_yy, H_yz],
        [H_yz.T, H_zz]
    ])

def backtracking_line_search(y, dy, c, A, Sigma, mu, t, lam, wprev, c_l1, obj, eta, 
                              alpha=0.3, beta=0.8, max_iter=50):
    """Backtracking line search to find a step size that decreases the objective"""
    try:
        f0 = -obj @ y + eta * G_value(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
        grad = -obj + eta * G_grad(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
        directional_deriv = grad @ dy
        
        step = 1.0
        for _ in range(max_iter):
            y_new = y + step * dy
            try:
                f_new = -obj @ y_new + eta * G_value(y_new, c, A, Sigma, mu, t, lam, wprev, c_l1)
                # Armijo condition
                if f_new <= f0 + alpha * step * directional_deriv:
                    return step
            except ValueError:
                # Step goes outside domain
                pass
            step *= beta
        return step
    except:
        return 0.1  # Fallback small step

def newton_step(y, c, A, Sigma, mu, t, lam, wprev, c_l1, obj, eta):
    g = -obj + eta * G_grad(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
    H = eta * G_hess(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
    
    # Add small regularization for numerical stability
    H += np.eye(H.shape[0]) * 1e-8
    
    dy = np.linalg.solve(H, -g)
    lambda_sq = g @ (-dy)  # Newton decrement squared
    
    # Make sure lambda_sq is non-negative
    lambda_sq = max(0, lambda_sq)
    delta = np.sqrt(lambda_sq)
    
    return dy, delta


import numpy as np

def interior_point_method(Sigma, lam, mu_param, w0, wprev, c_l1):
    f_values = []  # 1. Initialize list to store values
    
    n = len(w0)
    c_const, A = make_simplex_param(n)
    
    # Pre-calculate the dimension of the subspace yw for slicing later
    k_dim = A.shape[1] 
    
    t = 1.0  
    eta = 10.0  
    tau = 0.5
    epsilon = 1e-4  
    eta_stop = 1e-6
    max_newton_iters = 50  
    
    A_pinv = np.linalg.pinv(A)
    yw = A_pinv @ (w0 - c_const)
    z = np.ones(n) * 0.3  
    y = np.concatenate([yw, z])
    
    # obj seems to be external, assuming it is defined or derived from inputs
    # If obj is not defined in this scope, ensure you define it before this line.
    # For this example, I assume obj relates to linear terms:
    # obj_y = A.T @ obj
    # obj_full = np.concatenate([obj_y, np.zeros(n)])
    
    # Placeholder for obj_full if not provided in snippet:
    # You likely have this defined elsewhere, but ensure it is passed or calculated.
    
    outer_iter = 0
    while eta > eta_stop and outer_iter < 100:
        print(f"\n{'='*60}")
        print(f"OUTER ITERATION {outer_iter}, eta = {eta:.6e}")
        print(f"{'='*60}")
        
        for inner_iter in range(max_newton_iters):
            # --- START MODIFICATION ---
            # 2. Extract yw from current y (first k_dim elements)
            yw_curr = y[:k_dim]
            
            # 3. Reconstruct w
            w_curr = c_const + A @ yw_curr
            
            # 4. Calculate Objective Function
            # f(w) = 0.5 * w^T Sigma w - lambda * w^T mu + c_l1 * ||w - w_prev||_1
            term1 = 0.5 * w_curr.T @ Sigma @ w_curr
            term2 = lam * w_curr.T @ mu_param
            term3 = c_l1 * np.linalg.norm(w_curr - wprev, 1)
            
            f_value = term1 - term2 + term3
            f_values.append(f_value)
            # --- END MODIFICATION ---

            try:
                # Assuming obj_full is available or passed to these functions
                g_val = G_value(y, c_const, A, Sigma, mu_param, t, lam, wprev, c_l1)
                dy, delta = newton_step(y, c_const, A, Sigma, mu_param, t, lam, wprev, c_l1, obj_full, eta)
                
                if delta < epsilon:
                    break
                
                step = backtracking_line_search(y, dy, c_const, A, Sigma, mu_param, t, lam, wprev, 
                                                c_l1, obj_full, eta)
                
                y += step * dy
            except Exception as e:
                print(f"Error in inner loop: {e}")
                break
        
        eta *= tau
        outer_iter += 1
    
    yw_final = y[:k_dim]
    w = c_const + A @ yw_final
    
    return w, f_values



plt.figure(figsize=(8,5))
plt.plot(f_values_pgd, label='projected gradient descent')
#plt.plot(f_values_pgda, label='projected gradient descent armijo') 
plt.plot(f_values_pgdm, label='projected gradient descent momentum') 
plt.plot(f_values_pcd, label='projected coordinate descent')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Objective Function vs Iteration")
plt.grid(True)
plt.legend()
plt.show()





# Plot the objective function over iterations
plt.figure(figsize=(8,5))
plt.plot(f_values_sub, label='subgradiend descent')
plt.plot(f_values_prox, label='proximal gradient descent')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Objective Function vs Iteration")
plt.grid(True)
plt.legend()
plt.show()
