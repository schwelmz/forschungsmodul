import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np

# Define parameters
def birth_death(lam, sigma, mu, init_val, tend, nsteps):
    # Define the model
    model = Model(name="BirthDeathImmigrationProcess")

    # Define species
    A = Species(name="A", initial_value=init_val, mode='continuous')

    #reaction rates
    birth_rate = Parameter(name="birth_rate", expression=lam)
    death_rate = Parameter(name="death_rate", expression=sigma)
    immigration_rate = Parameter(name="immigration_rate", expression=mu)


    # Add species and parameters to the model
    model.add_species([A])
    model.add_parameter([birth_rate, death_rate, immigration_rate])

    # Define reactions
    birth_reaction = Reaction(name="birth", reactants={A: 2}, products={A: 3}, propensity_function='birth_rate/2*pow(A,2)')
    death_reaction = Reaction(name="death", reactants={A: 1}, products={}, propensity_function= 'death_rate * A')
    immigration_reaction = Reaction(name="immigration", reactants={}, products={A: 1}, propensity_function= 'immigration_rate')

    # Add reactions to the model
    model.add_reaction([birth_reaction, death_reaction, immigration_reaction])

    # Set up the solver
    timepoints = np.linspace(0, tend, nsteps)
    model.timespan(timepoints)

    return model

#settings
tend = 600
nsteps = 200
tspace = np.linspace(0,tend,nsteps)
print("tend=",tend,"nsteps=",nsteps)
init_population = 84.36e6
lam = 0.5*738819/init_population  #birth rate (0.01)
sigma = 2*1.066e6/init_population   #death_rate (1)
mu = 0.5*17.3e-3    #immigration rate (0.5)
print("lambda=",lam,"sigma=",sigma,"mu=",mu)
Omega = sigma/lam
delta_squared = 1- 2*mu*lam/(sigma**2)
print("Omega=",Omega," delta^2=",delta_squared)
delta = np.sqrt(delta_squared)
n1 = Omega*(1-delta)
n2 = Omega*(1+delta)
print("n1=",n1,"n2=",n2)

#calc characteristic relaxation time
tau = 1/(sigma*delta)

# Run the simulation
fig1, ax1 = plt.subplots(figsize=(12,8))
fig2, axs2 = plt.subplots(6,1,figsize=(8,12),layout="constrained")

pdx = 0
for i in [0, n1+1, n1+4, n1+7, n1+(n2-n1-0.1), 12]: #np.linspace(np.maximum(0,int(n1-(n2-n1)/2)), int(n2+(n2-n1)/2), 7):
    print("running with initial value A0=",i)
    model = birth_death(lam, sigma, mu, i, tend, nsteps)
    results = model.run(number_of_trajectories=1, algorithm = "ODE")
    #plot 1
    ax1.plot(results['time'], abs(results['A']), label=f"model A0={round(i,3)}")
    #plot 2
    offset = i - n1
    f_compare = offset*np.exp(-1/tau*tspace)
    axs2[pdx].plot(results['time'], results['A']-n1, label=f"model A0={round(i,3)}")
    axs2[pdx].plot(results['time'], f_compare, label="comparison", linestyle="--")
    # axs2[pdx].axhline(y=n1, linestyle='--', label="n1", color="red")
    axs2[pdx].set_title(f"offset = {round(offset,3)}")
    axs2[pdx].legend()
    pdx += 1

# Plot the results
ax1.axhline(y=n1, linestyle='--', label="n1", color="red")
ax1.axhline(y=n2, linestyle='--', label="n2", color="blue")
ax1.set_xlabel('Time')
ax1.set_ylabel('Population Size')
ax1.set_ylim(np.maximum(0,int(n1-(n2-n1)/4)-5)-1, int(n2+(n2-n1)/4)+2)
ax1.set_xlim(0,tend)
ax1.legend()

# Plot 2
# axs2.set_xlabel('Time')
# axs2.set_ylabel('Population Size')
# axs2.set_xlim(0,tend)

plt.show()
