import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np
import settings

# Define parameters
def create(lam, sigma, mu, init_val, tend, nsteps):
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