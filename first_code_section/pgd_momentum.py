# here i fix gamma = 1/L and for beta i try different values, i plot them and see which one performs better
def projected_gradient_descent_momentum(sigma, lamb, mu, w, beta, gamma, max_iter = 1000):
    m = np.zeros(len(w))
    
    f_values = [0.5 * w.T @ sigma @ w - lamb * w.T @ mu]
    
    for i in range(max_iter):
        grad = compute_gradient(sigma, lamb, mu, w)

        m = beta * m + (1 - beta) * grad
        w = w - gamma * m
        
        w = project_simplex(w)

        f_value = 0.5 * w.T @ sigma @ w - lamb * w.T @ mu
        f_values.append(f_value)
    return w, f_values

betas = [0.5,0.6,0.7,0.8,0.9,0.99]
gamma = 1/L

v = {}

for b in betas:
    w_opt, f_values = projected_gradient_descent_momentum(sigma, lamb, mu, w0, beta=b, gamma=gamma, max_iter=300)
    v[b] = f_values

plt.figure(figsize=(9,6))
for b, hist in v.items():
    plt.plot(hist, label=f"beta={b}")

plt.xlabel("Iterations")
plt.ylabel("Objective value")
plt.title("Convergence with gamma = 1/L")
plt.legend()
plt.grid(True)
plt.show()
