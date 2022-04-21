import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def piecewise_linear_loss(params, p):
    [m1, c1, m2, c2] = params
    return np.mean([(p[k] - np.log(max(m1*k + c1, m2*k + c2, 0.01)))**2 for k in range(len(p))])

def exponential_loss(params, p):
    [m0, c0] = params
    return np.mean([(p[k] - (m0*k + c0))**2 for k in range(len(p))])

mu = 0.7/100
sigma = 1.1/100

N = 132
x = 0
p = [x]

for _ in range(N):
    x += np.random.normal(loc=mu, scale=sigma)
    p.append(x)

res = minimize(exponential_loss, np.array([mu, 0]), args=p, method="Powell", tol=1e-6)
g_loss = res.fun
[m0, c0] = res.x

res2 = minimize(piecewise_linear_loss, np.array([mu, 0, mu, 0]), args=p, method="Powell", tol=1e-6)
d_loss = res2.fun
[m1, c1, m2, c2] = res2.x

print("Exponential log L2 loss: \t", g_loss)
print("Piecewise linear log L2 loss: \t", d_loss)

plt.plot(range(1890, 1890+N+1), np.exp(p))
plt.plot(range(1890, 1890+N+1), [max(m1*k + c1, m2*k + c2) for k in range(N+1)])
plt.plot(range(1890, 1890+N+1), [np.exp(m0*k + c0) for k in range(N+1)])

plt.show()
