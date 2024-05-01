import numpy as np
import matplotlib.pyplot as plt
import settings

tend, nsteps, lam, sigma, mu, NoT, init_val, Omega, delta, n1, n2, t_decay = settings.read_parameters()
print(n2)

birth_rate = lam
death_rate = sigma
immigration_rate = mu

birth_propensity_function = lambda A: birth_rate/2 * A * (A)
death_propensity_function= lambda A: death_rate * A
immigration_propensity_function= lambda A: immigration_rate

flux = lambda n: birth_rate/2 * n**2 + immigration_rate - death_rate*n
flux_2 = lambda n: birth_rate/2 * n*(n-1) + immigration_rate - death_rate*n
A_space = np.arange(0,15)
A_space_fine = np.linspace(0,15,1000)

# plt.scatter(A_space, birth_propensity_function(A_space), label='birth')
# plt.scatter(A_space, death_propensity_function(A_space), label='death')
# plt.scatter(A_space, np.ones(A_space.shape[0])*immigration_rate, label='immigration')
plt.scatter(A_space, flux(A_space), label="lam/2*A^2 + mu - sigma*A")
plt.plot(A_space_fine, flux(A_space_fine),linestyle='--')
plt.scatter(A_space, flux_2(A_space), label="lam/2*A*(A-1) + mu - sigma*A")
plt.plot(A_space_fine, flux_2(A_space_fine),linestyle='--')
plt.axhline(y=0,color='black')
plt.axvline(x=n1,linestyle='--', color='green',label='Fixpunkt 1')
plt.axvline(x=n2,linestyle='--', color='red', label="Fixpunkt 2")
plt.xticks(A_space)
plt.legend()
plt.xlabel('Anzahl Moleküle A')
plt.grid()
plt.show()
