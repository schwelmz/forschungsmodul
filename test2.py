import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Model, Species, Parameter, Reaction, RateRule, FunctionDefinition

# Define the model
class SimpleModel(Model):
    def __init__(self, init_val, lam_val, mu_val, sigma_val, parameter_values=None):
        Model.__init__(self, name="SimpleModel")
        
        # Species
        A = Species(name='A', initial_value=init_val, mode='continuous')
        
        self.add_species([A])
        
        # Parameters
        rate_lambda = Parameter(name='rate_lambda', expression=lam_val)
        mu = Parameter(name='mu', expression=mu_val)
        sigma = Parameter(name='sigma', expression=sigma_val)
        
        self.add_parameter([rate_lambda, mu, sigma])
        
        # Reactions
        r1 = Reaction(name="production", reactants={A:2}, products={A: 3}, ode_propensity_function = "rate_lambda/2*pow(A,2)")
        
        r2 = Reaction(name="immigration", reactants={}, products={A: 1}, ode_propensity_function = "mu")
        
        r3 = Reaction(name="decay", reactants={A: 1}, products={}, ode_propensity_function = "sigma*A") 
        
        self.add_reaction([r1, r2, r3])
        
        # Set the timespan for the simulation
        self.timespan(np.linspace(0, 1000, 10000))
        

# Run the model
init_population = 84.36e6
lam_val = 0.5*738819/init_population  #birth rate (0.01)
sigma_val = 2*1.066e6/init_population   #death_rate (1)
mu_val = 0.5*17.3e-3    #immigration rate (0.5)
Omega = sigma_val/lam_val
delta_squared = 1- 2*mu_val*lam_val/(sigma_val**2)
delta = np.sqrt(delta_squared)
n1 = Omega*(1-delta)
n2 = Omega*(1+delta)
print(n1,n2)

fig1, ax1 = plt.subplots()
for A0 in [n2-0.1,n2,n2+0.1]:
    print(A0)
    model = SimpleModel(A0, lam_val, mu_val, sigma_val)
    results = model.run(algorithm = "ODE")

    # Plot the results
    ax1.plot(results['time'], results['A'], label='A')
ax1.axhline(y=n1,color='black',linestyle='--')
ax1.axhline(y=n2,color='black',linestyle='--')
ax1.set_ylim(0,20)
plt.show()

fig2, ax2 = plt.subplots()
flow = lambda n: mu_val - sigma_val*n + lam_val/2*n**2
ns = np.linspace(0,15,1000)
fs = flow(ns)
ax2.plot(ns,fs)
ax2.grid()
ax2.axhline(y=0)
ax2.axvline(x=n2)
plt.show()
