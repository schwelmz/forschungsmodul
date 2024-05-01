import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np
import settings
import birthDeathModel
import visualization

#read parameters
tend, nsteps, lam, sigma, mu, NoT, init_val, Omega, delta, n1, n2, t_decay = settings.read_parameters()

#load results
print('open file:',f"out/out_tend{'%.1e'%nsteps}_NoT{NoT}.npy")
results_array = np.load(f"out/out_tend{'%.1e'%nsteps}_NoT{NoT}.npy")

#plot each trajectory over time
if False:
    increment = 1000
    plt.rcParams.update({'font.size': 18})
    fig1, ax1 = plt.subplots(figsize=(12,8))
    visualization.plot(ax1,results_array, nsteps, NoT, n1, n2, t_decay, increment)
    ax1.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}")
    # plt.savefig(f"../results/plot_tend{'%.1e'%nsteps}_NoT{NoT}.png")

#histogram of the last value of each trajectory
if False:
    fig2, ax2 = plt.subplots(figsize=(12,8))
    bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    visualization.histogram(ax2,results_array, NoT, bins)
    ax2.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}")
    #plt.savefig(f"../results/hist_tend{'%.1e'%nsteps}_NoT{NoT}.png")

#visualize all trajectories via a heatmap
if True:
    fig3, ax3 = plt.subplots(figsize=(12,8))
    visualization.heatmap(fig3,ax3,results_array,t_decay,1)
    ax3.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}, tau = {'%.1e'%t_decay}")
    #plt.savefig(f"../results/heatmap_tend{'%.1e'%nsteps}_NoT{NoT}.png")

plt.show()