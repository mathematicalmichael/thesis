import numpy as np
from mud import make_map_sol, make_mud_sol
from mud import transform_linear_map, mud_sol, map_sol

def wrap_problem(dim_input = 2,
                 dim_output = 1,
                 data_std = .05,
                 cov_11=0.5, cov_01=-0.5,
                 seed = 21, max_num=10):
    assert dim_output <= dim_input, "Dimension mismatch."

    np.random.seed(seed)
    initial_mean = np.random.rand(dim_input).reshape(-1,1)*0
#     initial_cov = np.eye(dim_input) # TODO expand to general diagonals, then arbitrary
#     initial_cov[0:2,0:2] = np.array([[1, cov_01],[cov_01, cov_11]])
    initial_cov = np.diag(np.random.rand(dim_input))
    M = np.random.rand(dim_output, dim_input)
#     M = np.ones((1, dim_input))
    std_of_data = data_std*np.ones(dim_output) # TODO generalize to different vals
#     std_of_data = data_std
    num_obs = np.random.randint(1,max_num+1,dim_output)
#     num_obs = np.ones(dim_output)*max_num
#     print(num_obs)
    data_list = np.array([np.random.rand(int(num_obs[i])) + np.random.randn(int(num_obs[i]))*std_of_data[i] for i in range(dim_output)])
    std_list = np.array([ std_of_data[i]*np.ones(int(num_obs[i])) for i in range(dim_output)]).ravel()
#     print(np.sum(num_obs), dim_output)
#     np.zeros((int(np.sum(num_obs)), dim_input))
    map_array = np.vstack([np.tile(M[i,:],(int(num_obs[i]),1))  for i in range(dim_output) ]).reshape(int(np.sum(num_obs)), -1)
    data = np.hstack(data_list)
    # observed mean should be zero for the repeated observation problem.
#     A, b, observed_mean, predicted_cov, mud = make_mud_sol(initial_mean=initial_mean,
#                                     initial_cov=initial_cov, M=M,
#                                     data_list=data_list, std_of_data=std_of_data)



#     post_mean, post_cov = make_map_sol(prior_mean=initial_mean,
#                                        prior_cov=initial_cov,
#                                        data_std=1,
#                                        A=A, data=observed_mean, b=b)



    A, b = transform_linear_map(M, data_list, std_of_data)
    mud_pt = mud_sol(A, b, initial_mean, initial_cov)
#     map_pt = np.linalg.solve(map_array.T@map_array + np.linalg.inv(initial_cov), map_array.T@data)
#     map_pt = map_sol(A, b, initial_mean, initial_cov,w=1)
    # regularization keeps solution from being accurate in predictions.

    map_pt = np.linalg.pinv(A)@-b
#     map_pt = map_sol(map_array, -data, initial_mean, initial_cov, np.diag(std_list))
#     map_pt = map_sol(map_array, data, initial_mean, initial_cov)
#     data_cov = np.linalg.inv(np.diag(std_of_data)**2)
#     map_pt = np.linalg.inv(map_array.T@map_array + np.linalg.inv(initial_cov))@map_array.T@data
#     print(np.linalg.norm(A@mud_pt + b), np.linalg.norm(A@map_pt + b))
#     print(A@mud_pt + b)

    err_mud = np.linalg.norm(A@mud_pt + b)
#     print("Error MUD:", err_mud)
#     err_map = np.linalg.norm(map_array@map_pt - data)
    err_map = np.linalg.norm(A@map_pt + b)
#     print("Error MAP:", err_map)

    return err_mud, err_map

# if __name__=='__main__':
dim_input = 2
dim_output = 1
print("Demonstrating example: MUD | MAP")
for i in range(5):
#     dim_input = np.random.randint(50,1000)
#     dim_output = np.random.randint(1,10)

    res = wrap_problem(dim_input, dim_output, max_num=i*10+2, data_std=.1, seed=i)
    print("%3d => %3d"%(dim_input, dim_output), "%2.6e, %2.6e"%(res[0], res[1]))




R = {}
# max_num_list = [10, 25, 50, 100, 250, 500, 1000]
max_num_list = 2**np.arange(1,16)
data_std_list = np.linspace(0.01, .1, 20)
# data_std_list = np.array([.1, 1])
for data_std in data_std_list:
    R[data_std] = {}
    for max_num in max_num_list:
        s = np.array([0, 0])
        nt=25
        for t in range(nt):
            s = s+np.array(wrap_problem(dim_input, dim_output,
                                           max_num=max_num,
                                           data_std=data_std, seed=21))
        R[data_std][max_num] = s/nt

import pickle
# with open('save_grid_{}.to.{}-{}.pkl'.format(dim_input, dim_output, nt), 'wb') as f:
#     pickle.dump(R, f)

import matplotlib.pyplot as plt
import pandas as pd
b = pd.DataFrame(R)
c = b.to_numpy()
P = np.ones((len(max_num_list), len(data_std_list)))
U = np.ones((len(max_num_list), len(data_std_list)))
# with open('save_grid_{}.to.{}-{}.pkl'.format(dim_input, dim_output, nt), 'wb') as f:
#     pickle.dump([P, U], f)
for i in range(c.shape[0]):
    print(max_num_list[i])

    mudpt = [cc[0] for cc in c[i]]
    mappt = [cc[1] for cc in c[i]]
    P[i,:] = mappt
    U[i,:] = mudpt

    plt.scatter(data_std_list, mudpt, c='k', alpha = 1/(i+2))
    plt.scatter(data_std_list, mappt, c='r', alpha = 1/(i+2))
#         plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('std')
    plt.ylabel('error')
    plt.ylim(1E-18, 1)
plt.show()

plt.imshow(P)
plt.ylabel('num obs (down = more)')
plt.xlabel('std (right = more)')
plt.title("MAP")
plt.show()
plt.imshow(U)
plt.ylabel('num obs (down = more)')
plt.xlabel('std (right = more)')
plt.title("MUD")
plt.show()


R = {}
input_dims = np.arange(21,40)
output_dims = np.arange(1,20)
for dim_input in input_dims:
    R[dim_input] = {}
    for dim_output in output_dims:
        s = np.array([0, 0])
        nt=25
        for t in range(nt):
            s = s+np.array(wrap_problem(dim_input, dim_output,
                                           max_num=100,
                                           data_std=0.1, seed=21))
        R[dim_input][dim_output] = s/nt

b = pd.DataFrame(R)
c = b.to_numpy()
insz, outsz = len(input_dims), len(output_dims)

P = np.empty((outsz,insz))
U = np.empty((outsz,insz))
# with open('save_grid_{}.to.{}-{}.pkl'.format(dim_input, dim_output, nt), 'wb') as f:
#     pickle.dump([P, U], f)
for i in range(c.shape[0]):
    mudpt = [cc[0] for cc in c[i]]
    mappt = [cc[1] for cc in c[i]]
    P[i,:] = mappt
    U[i,:] = mudpt


with open('save_grid_dimensions.pkl'.format(dim_input, dim_output, nt), 'wb') as f:
    pickle.dump([input_dims, output_dims, P, U], f)

plt.contourf(input_dims, output_dims, P, vmin=1E-16, vmax=1E-11)
plt.show()

plt.contourf(input_dims, output_dims, U, vmin=1E-16, vmax=1E-11)
plt.show()

for i in range(outsz):
    plt.plot(input_dims, P[:,i], c='xkcd:forest green')
    plt.plot(input_dims, U[:,i], c='xkcd:black')
    plt.yscale('log')

for i in range(outsz):
    plt.plot(output_dims, P[i,:], c='xkcd:forest green')
    plt.yscale('log')

for i in range(outsz):
    plt.plot(output_dims, U[i,:], c='xkcd:black')
    plt.yscale('log')
