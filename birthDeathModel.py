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
    birth_reaction = Reaction(name="birth", reactants={A: 2}, products={A: 3}, propensity_function='birth_rate/2 * A * (A-1)')
    death_reaction = Reaction(name="death", reactants={A: 1}, products={}, propensity_function= 'death_rate * A')
    immigration_reaction = Reaction(name="immigration", reactants={}, products={A: 1}, propensity_function= 'immigration_rate')

    # Add reactions to the model
    model.add_reaction([birth_reaction, death_reaction, immigration_reaction])

    # Set up the solver
    timepoints = np.linspace(0, tend, nsteps)
    model.timespan(timepoints)

    return model

#settings
tend = 1000
nsteps = 2001
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

# Run the simulation
for i in np.linspace(np.maximum(0,int(n1-(n2-n1)/2)), int(n2+(n2-n1)/2), 15):
    print("running with initial value A0=",int(i))
    model = birth_death(lam, sigma, mu, int(i), tend, nsteps)
    results = model.run(number_of_trajectories=1, algorithm = "ODE")
    plt.plot(results['time'], results['A'])

# Plot the results
plt.axhline(y=n1, linestyle='--', label="n1", color="red")
plt.axhline(y=n2, linestyle='--', label="n2", color="blue")
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.ylim(np.maximum(0,int(n1-(n2-n1)/4)-5), int(n2+(n2-n1)/4)+5)
plt.legend()
plt.title("Deterministic")
plt.show()