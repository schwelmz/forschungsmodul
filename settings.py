import configparser
import sys
import numpy as np

def read_parameters():
    file_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(file_path)

    parameters = {}
    for section in config.sections():
        for key, value in config.items(section):
            parameters[key] = value

    tend = float(parameters.get('tend'))
    nsteps = int(parameters.get('nsteps'))
    lam = float(parameters.get('lam'))
    sigma = float(parameters.get('sigma'))
    mu = float(parameters.get('mu'))
    numberOfTrajectories = int(parameters.get('numberoftrajectories'))

    #calculations
    Omega = sigma/lam
    delta_squared = 1- 2*mu*lam/(sigma**2)
    delta = np.sqrt(delta_squared)
    n1 = Omega*(1-delta)
    n2 = Omega*(1+delta)
    init_val = int(n1)

    #decay time
    f = lambda x: x - np.sqrt(8*mu/lam) * np.arctan(x*np.sqrt(lam/(2*mu)))
    S0 = f(n2)-f(n1)
    t_decay = 1/sigma*np.exp(S0)

    return tend, nsteps, lam, sigma, mu, numberOfTrajectories, init_val, Omega, delta, n1, n2, t_decay

def print_settings():
    tend, nsteps, lam, sigma, mu, numberOfTrajectories, init_val, Omega, delta, n1, n2, t_decay = read_parameters()
    print('tend =',tend)
    print('nsteps =',nsteps)
    print('birth-rate lambda =',lam)
    print('death-rate sigma =',sigma)
    print('immigration-rate mu =', mu)
    print('Number of trajectories =', numberOfTrajectories)
    print('Initial value: ', init_val)
    print('Omega =', Omega)
    print('delta =', delta)
    print('n1 =', n1)
    print('n2 =', n2)
    print("t_decay", t_decay)