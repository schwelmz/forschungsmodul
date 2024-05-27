# Simulating noise-driven unlimited population growth (stochastic)

For the deterministic simulations switch to the branch "deterministic".

## Requirements
The following python packages need to be installed:
- matplotlib
- numpy
- gillespy2
- configparser

## Usage
- The model parameters $\lambda$, $\sigma$ and $\mu$ as well as the simulation time, time steps and number of trajectories can be modified in the parameter.ini file.
- Run the following command to generate the data for the figures (first create the "out" directory when running for the first time):

```bash
python3 computation.py parameter.ini
```
- Enable the figures you want to produce in "main.py" by setting them to "True". Then run:

```bash
python3 main.py parameter.ini
```