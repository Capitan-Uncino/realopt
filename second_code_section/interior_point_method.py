import numpy as np

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

def newton_step(y, c, A, Sigma, mu, t, lam, wprev, c_l1, obj, eta):
    g = -obj + eta * G_grad(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
    H = eta * G_hess(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
    dy = np.linalg.solve(H, -g)
    delta = np.sqrt(float(g @ (-dy)))
    return dy, delta

if __name__ == "__main__":
    n = 4
    c, A = make_simplex_param(n)
    
    # Initial portfolio weights
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Objective function (portfolio returns)
    obj = np.array([1.5, 0.2, 0.3, 1.7])
    
    # Define problem parameters
    Sigma = np.eye(n) * 0.1  # Covariance matrix (simplified)
    mu_param = np.array([0.1, 0.05, 0.08, 0.12])  # Expected returns
    t = 10.0  # Epigraph variable upper bound
    lam = 1.0  # Risk-return tradeoff parameter
    wprev = x0.copy()  # Previous portfolio (for turnover constraints)
    c_l1 = 0.1  # L1 regularization coefficient
    
    # Barrier method parameters
    eta = 1.0
    tau = 0.6
    epsilon = 1e-5 
    eta_stop = 1e-8
    
    if np.any(x0 <= 0) or not np.isclose(sum(x0), 1):
        raise ValueError("initial value x0 unfeasible")
    
    # Initialize y and z variables
    A_pinv = np.linalg.pinv(A)
    yw = A_pinv @ (x0 - c)
    z = np.ones(n) * 0.5  # Initialize slack variables
    y = np.concatenate([yw, z])
    
    obj_y = A.T @ obj
    obj_full = np.concatenate([obj_y, np.zeros(n)])
    
    for i in range(1000):
        print("______________________________________________")
        print("Iteration number:", i)
        try:
            g_val = G_value(y, c, A, Sigma, mu_param, t, lam, wprev, c_l1)
            print("G(y) =", g_val)
            
            dy, delta = newton_step(y, c, A, Sigma, mu_param, t, lam, wprev, c_l1, obj_full, eta)
            print("Newton step dy norm:", np.linalg.norm(dy))
            print("Delta:", delta)
            print("eta:", eta)
            
            y += 1.0 / (1.0 + delta) * dy 
            
            if delta < epsilon:
                eta *= tau
            if eta < eta_stop:
                break
        except ValueError as e:
            print(f"Error: {e}")
            break
    
    yw_final = y[:A.shape[1]]
    x = c + A @ yw_final
    print("\nFinal value for x is:", x)
    print("Sum of x:", np.sum(x))
    print("All positive:", np.all(x > 0))
