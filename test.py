import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np
import settings
import birthDeathModel
import visualization

nsteps =1e6
NoT = 50
n1=0
n2=12
results_array = np.load(f"out/out_tend1e6_NoT{NoT}.npy")

fig1, ax1 = plt.subplots()
visualization.plot(ax1,results_array, nsteps, NoT, n1, n2,100)

fig2, ax2 = plt.subplots(figsize=(12,8))
bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
visualization.histogram(ax2,results_array, NoT, bins)
ax2.set_title(f"tend = {nsteps}, trajectories = {NoT}")

plt.show()