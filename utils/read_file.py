data = pd.read_csv(f"data/all_stocks_5yr.csv")

print(data)

#compute returns
data = data.sort_values(["Name", "date"])
data["prev_close"] = data.groupby("Name")["close"].shift(1)
data["return"] = (data["close"] - data["prev_close"]) / data["prev_close"]

returns = data.pivot(index="date", columns="Name", values="return")   # it contains the returns per day of each asset

#mean return
mu = returns.mean()
print(mu)
mu.shape

#covariance matrix
sigma = returns.cov()
print(sigma)

eigenvalues, eigenvectors = LA.eig(sigma)
print(f"Max: ", np.max(eigenvalues), "min: ", np.min(eigenvalues))

cond_number = np.linalg.cond(sigma)
print(cond_number)

det = np.linalg.det(sigma)
print(det)

# number of iterations
max_iter = 1000

# risk adversion
lamb = 1.0
n = len(mu)
w0 = np.ones(n) / n   # equally distributed portfolio
