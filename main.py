import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gillespy2 import Species, Reaction, Parameter, Model
import scipy as sp
import numpy as np
import settings
import birthDeathModel
import visualization

#read parameters
tend, nsteps, lam, sigma, mu, NoT, init_val, Omega, delta, n1, n2, t_decay = settings.read_parameters()

plt.rcParams.update({'font.size': 18})

#load results
print('open file:',f"out/out_tend{'%.1e'%tend}_NoT{NoT}_lam{lam}_sigma{sigma}_mu{mu}.npy")
results_array = np.load(f"out/out_tend{'%.1e'%tend}_NoT{NoT}_lam{lam}_sigma_{sigma}_mu{mu}.npy")

#plot each trajectory over time
if False:
    increment = 1
    plt.rcParams.update({'font.size': 18})
    fig1, ax1 = plt.subplots(figsize=(12,8))
    visualization.plot(ax1,results_array, nsteps, NoT, n1, n2, t_decay, increment)
    #ax1.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}")
    plt.savefig(f"../results/plot_tend{'%.1e'%nsteps}_NoT{NoT}.png")

#histogram of the last value of each trajectory
if False:
    fig2, ax2 = plt.subplots(figsize=(12,8))
    bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    visualization.histogram(ax2,results_array, NoT, bins)
    ax2.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}")
    #plt.savefig(f"../results/hist_tend{'%.1e'%nsteps}_NoT{NoT}.png")

#visualize all trajectories via a heatmap
if False:
    fig3, ax3 = plt.subplots(figsize=(12,8))
    increment = 1
    visualization.heatmap(fig3,ax3,results_array,t_decay,increment)
    # ax3.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}, tau = {'%.1e'%t_decay}")
    # plt.savefig(f"../results/heatmap_tend{'%.1e'%nsteps}_NoT{NoT}.png")


#histogram time point of escape to infinity
if False:
    fig4, ax4 = plt.subplots(figsize=(12,8))
    bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    visualization.histogram(ax4,results_array, NoT, bins)
    # plt.savefig(f"../results/hist_tend{'%.1e'%nsteps}_NoT{NoT}.png")

#histogram of time points when escape happens
if False:
    fig5, ax5 = plt.subplots(figsize=(12,8))
    escape_points = np.argmax(results_array>20, axis=1)
    print(escape_points)
    escape_points = escape_points[escape_points != 0]
    skips = tend/nsteps
    escape_points = escape_points*skips
    binsize = 100000
    style = {'facecolor': 'C1', 'edgecolor':'black', 'linewidth': 2}
    ax5.hist(escape_points,bins=np.arange(0,tend,binsize),**style)
    ax5.set_xlabel('timepoint of escape')
    ax5.set_ylabel('number of trajectories')
    # plt.savefig(f"../results/hist2_tend{'%.1e'%tend}_NoT{NoT}_lam{lam}_sigma{sigma}_mu{mu}.png")

#plot the right hand side of the rate equation
if False:
    fig6, ax6 = plt.subplots(figsize=(12,8))

    #define right hand side as function
    flux = lambda n: lam/2 * n**2 + mu - sigma*n

    A_space = np.arange(0,15)
    A_space_fine = np.linspace(0,15,1000)
    ax6.plot(A_space_fine, flux(A_space_fine),color='blue', label=r'$f(\bar n) = \mu - \sigma \bar n + \frac{\lambda}{2}\bar n^2$')
    ax6.axhline(y=0,color='black')
    ax6.scatter(n1,0, color='magenta',label='steady state 1')
    ax6.scatter(n2,0, color='darkblue', label="steady state 2")
    ax6.set_xticks(A_space)
    ax6.set_xlim(0,14)
    ax6.legend()
    ax6.set_xlabel(r'avg. population size $\bar n$')
    ax6.grid()

plt.show()