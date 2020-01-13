from models import make1DHeatModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# choose temperature locations along rod, in interval (0, 1)
num_locations = 100
temperature_locations = np.linspace(0, 1, 2+num_locations)[1:-1]

# kappa (thermal conductivity) values (2D), in interval [0.01, 0.2]

kappa_lin = np.linspace(0.01, 0.2, 5)[1:-1]
kappa_ref_locs = np.array([[k1, k2] for k1 in kappa_lin for k2 in kappa_lin])
ref_input_num = 6 # 4 is middle. choose 0-8,
# 2 and 6 are most disparate. 0, 4, 8 lead to symmetric heat curves
ref_input = np.array([kappa_ref_locs[ref_input_num]])
print(ref_input)

num_frames = 30
for t in np.linspace(0,1,num_frames):
    model = make1DHeatModel(temperature_locations, end_time=t)
    Qref = model(ref_input)[0]
    print(Qref)
    plt.cla()
    plt.plot(temperature_locations, Qref)
    plt.ylim(0,7)
    plt.savefig('heatrod_time/test-plot-t%04d.png'%(100*t))
