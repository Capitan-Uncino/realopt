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
    g = -t + eta * G_grad(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
    H = eta * G_hess(y, c, A, Sigma, mu, t, lam, wprev, c_l1)
    
    # Add small regularization for numerical stability
    H += np.eye(H.shape[0]) * 1e-8
    
    dy = np.linalg.solve(H, -g)
    lambda_sq = g @ (-dy)  # Newton decrement squared
    
    # Make sure lambda_sq is non-negative
    lambda_sq = max(0, lambda_sq)
    delta = np.sqrt(lambda_sq)
    
    return dy, delta

if __name__ == "__main__":
    n = 4
    c, A = make_simplex_param(n)
    
    # Initial portfolio weights
    x0 = np.array([0.25, 0.25, 0.25, 0.25])  # Start with uniform weights
    
    # Objective function (negative for maximization problem)
    obj = np.array([0.15, 0.02, 0.03, 0.17])  # Scaled down returns
    
    # Define problem parameters (scaled appropriately)
    Sigma = np.eye(n) * 0.01  # Smaller covariance
    mu_param = np.array([0.1, 0.05, 0.08, 0.12])
    t = 1.0  # Smaller epigraph bound
    lam = 0.5  # Risk-return tradeoff
    wprev = x0.copy()
    c_l1 = 0.01  # Smaller L1 penalty
    
    # Barrier method parameters
    eta = 10.0  # Start with larger eta
    tau = 0.5
    epsilon = 1e-4  # Convergence tolerance for Newton method
    eta_stop = 1e-6
    max_newton_iters = 50  # Maximum Newton iterations per eta
    
    if np.any(x0 <= 0) or not np.isclose(sum(x0), 1):
        raise ValueError("initial value x0 unfeasible")
    
    # Initialize y and z variables
    A_pinv = np.linalg.pinv(A)
    yw = A_pinv @ (x0 - c)
    z = np.ones(n) * 0.3  # Initialize slack variables conservatively
    y = np.concatenate([yw, z])
    
    obj_y = A.T @ obj
    obj_full = np.concatenate([obj_y, np.zeros(n)])
    
    outer_iter = 0
    while eta > eta_stop and outer_iter < 100:
        print(f"\n{'='*60}")
        print(f"OUTER ITERATION {outer_iter}, eta = {eta:.6e}")
        print(f"{'='*60}")
        
        # Inner Newton iterations for current eta
        for inner_iter in range(max_newton_iters):
            try:
                g_val = G_value(y, c, A, Sigma, mu_param, t, lam, wprev, c_l1)
                dy, delta = newton_step(y, c, A, Sigma, mu_param, t, lam, wprev, c_l1, obj_full, eta)
                
                if inner_iter % 10 == 0:
                    print(f"  Inner iter {inner_iter}: G(y) = {g_val:.6f}, delta = {delta:.6e}")
                
                # Check convergence
                if delta < epsilon:
                    print(f"  Converged! delta = {delta:.6e} < epsilon = {epsilon}")
                    break
                
                # Backtracking line search
                step = backtracking_line_search(y, dy, c, A, Sigma, mu_param, t, lam, wprev, 
                                                c_l1, obj_full, eta)
                
                # Update
                y += step * dy
                
            except ValueError as e:
                print(f"  Error: {e}")
                break
        
        # Decrease eta for next outer iteration
        eta *= tau
        outer_iter += 1
    
    # Extract final solution
    yw_final = y[:A.shape[1]]
    x = c + A @ yw_final
    
    print("\n" + "="*60)
    print("FINAL SOLUTION")
    print("="*60)
    print(f"Optimal weights: {x}")
    print(f"Sum: {np.sum(x):.10f}")
    print(f"All positive: {np.all(x > 0)}")
    print(f"Portfolio return: {obj @ x:.6f}")
