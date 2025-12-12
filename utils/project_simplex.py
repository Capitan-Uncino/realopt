def project_simplex(x):
    n = len(x)

    #sort in descending order
    u = np.sort(x)[::-1]

    cumsum_u = np.cumsum(u) - 1
    rho = np.nonzero(u > (cumsum_u / np.arange(1, n+1)))[0][-1]

    #compute the threshold theta
    theta = cumsum_u[rho] / (rho + 1)

    #project onto the simplex
    w = np.maximum(x - theta, 0)
    return w
