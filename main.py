import matplotlib.pyplot as plt
import numpy as np
import birthDeathModel
import settings

#settings
tend, nsteps, lam, sigma, mu, Omega, delta, n1, n2, tau = settings.read_parameters()
tspace = np.linspace(0,tend,nsteps)
settings.print_settings()
initial_values = [n1+0.1, n1+1, n1+2, n1+10]

fig1, ax1 = plt.subplots(figsize=(8,6))
fig2, axs2 = plt.subplots(4,1,figsize=(8,12),layout="constrained")

# Run the simulation
pdx = 0
for i in initial_values: #np.linspace(np.maximum(0,int(n1-(n2-n1)/2)), int(n2+(n2-n1)/2), 7):
    print("running with initial value A0=",i)
    model = birthDeathModel.birth_death(lam, sigma, mu, i, tend, nsteps)
    results = model.run(number_of_trajectories=1, algorithm = "ODE")
    #plot 1
    ax1.plot(results['time'], abs(results['A']), label=fr"$\bar n(0)$={round(i,3)}")
    #plot 2
    offset = i - n1
    f_compare = offset*np.exp(-1/tau*tspace)
    axs2[pdx].plot(results['time'], results['A']-n1, label=rf"$\bar n(0)$ ={round(i,3)}")
    axs2[pdx].plot(results['time'], f_compare, label=r"offset $\cdot \exp(-1\slash\tau_r\cdot t)$", linestyle="--")
    axs2[pdx].set_title(f"offset = {round(offset,3)}")
    axs2[pdx].legend()
    pdx += 1

# Plot 1: solution over time
ax1.axhline(y=n1, linestyle='--', label=r"$\bar n_1^s$", color="red")
ax1.axhline(y=n2, linestyle='--', label=r"$\bar n_2^s$", color="blue")
ax1.set_xlabel(r'Time $t$')
ax1.set_ylabel(r'avg. population Size $\bar n$')
ax1.set_ylim(np.maximum(0,int(n1-(n2-n1)/4)-5)-1, int(n2+(n2-n1)/4)+2)
ax1.set_xlim(0,tend)
ax1.legend()

# Plot 2: visualization of the relaxation time
axs2[3].set_xlabel(r'time $t$')
axs2[2].set_ylabel(r'avg. population size $\bar n$')

#fig1.savefig("../results/birthDeath_ODE_2.png")
# fig2.savefig("../results/relaxationTime.png")
plt.show()