import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np

# Define parameters
def birth_death(lam, sigma, mu, init_val, tend, nsteps):
    # Define the model
    model = Model(name="BirthDeathImmigrationProcess")

    # Define species
    A = Species(name="A", initial_value=init_val)

    #reaction rates
    birth_rate = Parameter(name="birth_rate", expression=lam)
    death_rate = Parameter(name="death_rate", expression=sigma)
    immigration_rate = Parameter(name="immigration_rate", expression=mu)


    # Add species and parameters to the model
    model.add_species([A])
    model.add_parameter([birth_rate, death_rate, immigration_rate])

    # Define reactions
    birth_reaction = Reaction(name="birth", reactants={A: 2}, products={A: 3}, rate=birth_rate)
    death_reaction = Reaction(name="death", reactants={A: 1}, products={}, rate=death_rate)
    immigration_reaction = Reaction(name="immigration", reactants={}, products={A: 1}, rate=immigration_rate)

    # Add reactions to the model
    model.add_reaction([birth_reaction, death_reaction, immigration_reaction])

    # Set up the solver
    timepoints = np.linspace(0, tend, nsteps)
    model.timespan(timepoints)

    return model

#print settings
tend = 5
nsteps = 100
print("tend=",tend,"nsteps=",nsteps)
lam = 0.01
sigma = 1
mu = 0.5
print("lambda=",lam,"sigma=",sigma,"mu=",mu)
Omega = mu/lam
delta_squared = 1- 2*sigma*lam/(mu**2)
delta = np.sqrt(delta_squared)
print("Omega=",Omega," delta^2=",delta_squared)
n1 = Omega*(1-delta)
n2 = Omega*(1+delta)
print("n1=",n1,"n2=",n2)
relaxation_time = 1/(mu*delta)
print("tau_r=",relaxation_time)

# Run the simulation
for i in np.linspace(n1,n2+3,8):
    print("running with initial value A0=",int(i))
    model = birth_death(lam, sigma, mu, int(i), tend, nsteps)
    results = model.run(number_of_trajectories=1, algorithm = "ODE")
    plt.plot(results['time'], results['A'],color="magenta")

# Plot the results
plt.axhline(y=n1, linestyle='--', label="n1", color="red")
plt.axhline(y=n2, linestyle='--', label="n2", color="blue")
plt.axvline(x=relaxation_time, linestyle='--', color='black')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.legend()
plt.show()