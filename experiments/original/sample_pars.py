import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-n', help = 'number of parameters to simulate', nargs='?', default=1, type=int)
parser.add_argument('-palpha', help='prior alpha (min, max)', nargs=2, metavar= ('min', 'max'), type=float)
parser.add_argument('-psigma', help='prior sigma (min, max)', nargs=2, metavar= ('min', 'max'), type=float)
parser.add_argument('-o', help = 'output folder', nargs='?', default='', type=str)
args = parser.parse_args()

# parse arguments 
alpha_min = args.palpha[0]
alpha_max = args.palpha[1]
sigma_min = args.psigma[0]
sigma_max = args.psigma[1]
outpath = args.o
n= args.n

# simulate parameters 
alphas = np.random.uniform(alpha_min, alpha_max, n)
sigmas = np.random.uniform(sigma_min, sigma_max, n)
pars = np.array([alphas, sigmas]).T
np.savetxt(f'{outpath}alpha:{alpha_min}-{alpha_max}_sigma:{sigma_min}-{sigma_max}.csv',pars, delimiter = ' ')