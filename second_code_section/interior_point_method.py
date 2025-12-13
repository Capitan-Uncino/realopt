import numpy as np

def make_simplex_param(n):
    #this function uses the invariance under affine transforms 
    #to enforce the equality constraint
    c = np.ones(n) / n
    A = np.zeros((n, n-1))
    for i in range(n-1):
        A[i, i] = 1.0
        A[n-1, i] = -1.0
    return c, A


def G_value(y, c, A):
    x = c + A @ y
    if np.any(x <= 0):
        raise ValueError("y out of domain: some x_i <= 0")
    return -np.sum(np.log(x))


def G_grad(y, c, A):
    x = c + A @ y
    if np.any(x <= 0):
        raise ValueError("y out of domain: some x_i <= 0")
    inv_x = 1.0 / x           
    return - A.T @ inv_x      


def G_hess(y, c, A):
    x = c + A @ y
    if np.any(x <= 0):
        raise ValueError("y out of domain: some x_i <= 0")
    D = np.diag(1.0 / (x**2))
    return A.T @ D @ A        


def newton_step(y, c, A, obj, mu):

    g = -obj + mu * G_grad(y, c, A)
    H = mu * G_hess(y, c, A)
    dy = np.linalg.solve(H, -g)

    delta = np.sqrt(float(g @ (-dy)))
    return dy, delta


if __name__ == "__main__":
    n = 4
    c, A = make_simplex_param(n)
    x0 = np.array([0.1, 0.2, 0.3, 0.4])
    obj = np.array([1.5, 0.2, 0.3, 1.7])

    mu = 1 
    tau = 0.6
    epsilon = 1e-5 
    mu_stop = 1e-8

    if np.any(x0<=0) or not np.isclose(sum(x0), 1):
        raise ValueError("initial value x0 unfeasible")

    A_pinv = np.linalg.pinv(A)
    y = A_pinv @ (x0 - c)
    obj_y = A.T @ obj

    for i in range(1000):
        print("______________________________________________")
        print("Iteration number:", i)
        print("G(y0) =", G_value(y, c, A))
        print("grad G(y0) =", G_grad(y, c, A))
        print("hess G(y0) = \n", G_hess(y, c, A))
        dy, delta = newton_step(y, c, A, obj_y, mu)
        print("Newton step dy:", dy)
        print("Delta:", delta)
        print("Mu:", mu)
        y+=1/(1+delta)*dy 
        if delta < epsilon:
            mu *= tau
        if mu < mu_stop:
            break
    x = c + A @ y 
    print("final value for x is ", x)

