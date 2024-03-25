import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np

# Plot the results
def plot(ax,results, nsteps, NoT, n1, n2,increment):
    tspace = np.arange(nsteps)
    for idx in range(0, NoT):
        ax.plot(tspace[::increment], results[idx,::increment])
    ax.set_title('SSA')
    ax.axhline(y=n1, linestyle='--', label="n1", color="red")
    ax.axhline(y=n2, linestyle='--', label="n2", color="blue")
    ax.set_xlabel('Time')
    ax.set_ylabel('Population Size')
    ax.set_ylim(np.maximum(-1,int(n1-(n2-n1)/4)-5), int(n2+(n2-n1)/4)+5)
    ax.legend()

def histogram(ax,results,NoT,bins):
    end_values = [] 
    for idx in range(0, NoT):
        end_values.append(results[idx,-1])
    values, bins, bars = ax.hist(np.clip(end_values, bins[0], bins[-1]), bins=bins)

    #xlabels
    xlabels = [str(arr) for arr in bins[:-1]]
    xlabels[-1] = f'{xlabels[-1]}+'
    Nlabels = len(xlabels)
    ax.set_xticks(np.arange(Nlabels)+0.5)
    ax.set_xticklabels(xlabels)

    #barlabels
    ax.bar_label(bars)

    ax.set_ylabel('#Trajectories')
    ax.set_xlabel('Value at last Iteration')