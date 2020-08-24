import numpy as np
from matplotlib import pyplot as plt

def exp_mud(mu, sigma, M, lam_ref):
	res = mu - lam_ref
	op = np.dot(sigma, np.dot(M.T, M) )
	wt = np.dot(M, np.dot(sigma, M.T) )
	P = op/wt
	return mu - np.dot(P, res)

K = np.array([[2, 1]])
qnum = 6
J = np.array([[0.9*np.sin(theta), np.cos(theta)] for theta in np.linspace(0, np.pi, qnum+1)[0:-1]]).reshape(qnum,2)

skew = 1.2
K = np.array([[1, 0], [np.sqrt(skew**2 - 1), 1]])
sigma = np.array([[1, 0], [0, 1]])*0.5
lam = np.array([0.5, 0.5])

num_repeats = 20
num_initials = 5
color = ["b", "k", "r", "g", "c"]
for i in range(num_initials):
	mu_last = np.random.rand(2)
	mu = np.copy(mu_last)
	for it in range(qnum*num_repeats):
		M = np.array([J[it%qnum, :]])
		#M = np.array([[6.28/(np.random.randint(qnum)%qnum+1), 1]])
# 		M = np.array([K[it%2, :]])
		#if it%2:
		#	M = np.array([[0.7, 0.7]])
		#else:
		#	M = np.array([[0, 1]])


		mu = exp_mud(mu_last, sigma, M, lam)
		print(mu_last, M, mu)
		plt.plot([mu_last[0], mu[0]], [mu_last[1], mu[1]], c=color[i%5])
		mu_last = np.copy(mu)

#plt.plot([0, 1], [0, 1], alpha=0.5, c="k")
#plt.plot([0, 1], [1.5, -0.5], alpha=0.5, c="k")
# plotting window
delta = 0.01
original_domain = True
#plt.axis("equal")
# plt.plot()
angle = np.pi/2
JJ = (np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])@J.T).T
# JJ = (np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])@K.T).T
for i in range(qnum):
    try:
        plt.plot([0.5-JJ[i,0],0.5+JJ[i,0]], [0.5-JJ[i,1], 0.5+JJ[i,1]], c='k', ls=':')
    except:
        pass

if original_domain:
    plt.xlim([0, 1])
    plt.ylim([0, 1])
else:
    plt.xlim([0.5-delta, 0.5+delta])
    plt.ylim([0.5-delta, 0.5+delta])


plt.show()
