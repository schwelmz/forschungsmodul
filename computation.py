import numpy as np
import matplotlib.pyplot as plt
from gillespy2 import Species, Reaction, Parameter, Model
import numpy as np
import settings
import birthDeathModel
import visualization
import time

#settings
tend, nsteps, lam, sigma, mu, NoT, init_val, Omega, delta, n1, n2, t_decay = settings.read_parameters()
settings.print_settings()

# Run the simulation
model = birthDeathModel.create(lam, sigma, mu, init_val, tend, nsteps)
threshold = 50
results = np.zeros((NoT, nsteps))
for tdx in range(0, NoT):
    print(f'trajectory:{tdx}/{NoT}')
    st = time.time()
    trajectory = model.run(number_of_trajectories=1, algorithm = "SSA", timeout=30)     #live_output='progress',live_output_options={"interval":1})
    et = time.time()
    print(f"comp. time: {et-st}s")
    results[tdx,:] = trajectory['A']

#save output
np.save(f"out/out_tend{'%.1e'%tend}_NoT{NoT}.npy",results)

#plot results
fig1, ax1 = plt.subplots(figsize=(12,8))
visualization.plot(ax1,results, nsteps, NoT, n1, n2, t_decay, 100)
ax1.set_title(f"tend = {'%.1e'%nsteps}, trajectories = {NoT}")
plt.show()

fig,ax=plt.subplots()
bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
visualization.histogram(ax,results, NoT, bins)
plt.show()
