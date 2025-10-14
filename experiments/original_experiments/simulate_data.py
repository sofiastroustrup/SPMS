#%%
import os
import numpy as np
from ete3 import Tree
import jax
import jax.numpy as jnp
import argparse
import sys 
import subprocess

from bridge_sampling.setup_SDEs import Stratonovich_to_Ito, dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import get_flat_values_sim
from bridge_sampling.simulate import simulate_tree


#%%
parser = argparse.ArgumentParser(
    description='Arguments for script')
parser.add_argument('-ds', help = 'seed for the simulation of data', nargs='?', type=int, default = 0)
parser.add_argument('-dt', help = 'stepsize for simulation', nargs='?', default=0.01, type=float)
parser.add_argument('-s', help='sigma_sim', nargs='?', type=float)
parser.add_argument('-a', help='alpha_sim', nargs='?', type=float)
parser.add_argument('-root', help='path to data file, which will be used in the root', nargs='?', default = 'shapes/hercules_forewing_n=20.csv', type=str )
parser.add_argument('-o', help='path to output folder', nargs='?', default = 'simdata', type=str )
parser.add_argument('-simtree', help='path to newick tree for simulation', nargs='?', default = 'chazot_subtree.nw', type=str )
parser.add_argument('-sti', help = 'Use stratonovich-ito correction', nargs='?', 
                    default=1, type=int)
parser.add_argument('-ov', help = 'obs_var also denoted gamma', nargs='?', default=0.01, type=float)
parser.add_argument('-rb', help = 'length of root branch, if 0 no root branch', nargs='?', default=0, type=float)


#%%
args = parser.parse_args()
seed_sim_data = args.ds
dt = args.dt
sigma_sim = args.s
alpha_sim = args.a
rootdata = args.root
obs_var = args.ov
simtree = args.simtree
outputfolder = args.o
sti = args.sti
rb = args.rb

#%%
# parameter settings
d=2
root = np.genfromtxt(rootdata, delimiter=',')
n = root.shape[0]//d

if seed_sim_data==0:
    print('!! no data seed given')
    seed_sim_data = np.random.randint(100000000000000000, size=1)[0]
    # seed = int(os.urandom(16).hex(), 16)
print(f'simulation seed: {seed_sim_data}')

outputpath = outputfolder+'/'+str(seed_sim_data)+'/'
cur_dir = os.getcwd()
path = cur_dir +'/'+ outputpath
if not os.path.isdir(path): 
    os.makedirs(path)
sys.stdout = open(f'{outputpath}log', 'w')

print('Simulating data')
print(f'Simulated data is saved in {outputpath}')
print(f'Use stratonovich-to-Ito correction: {args.sti}')
print('***')
print('Simulating data with the following settings:')
print(f'Sim seed = {seed_sim_data}')
print(f'Root = {rootdata}')
print(f'Stepsize = {dt}')
print(f'Seed simdata: {seed_sim_data}')
print(f'gtheta = {sigma_sim}')
print(f'kalpha = {alpha_sim}')
print(f'Observation noise = {obs_var}')
print(f'Root branch = {rb}')

#%%

theta_true = {
    'k_alpha': alpha_sim, # kernel amplitud
    'inv_k_sigma': 1./(sigma_sim)*jnp.eye(d), # kernel width, gets squared in kQ12
    'd':d,
    'n':n, 
}

# define drift and diffusion for process of interest 
if sti ==1:
    b,sigma,_ = Stratonovich_to_Ito(lambda t,x,theta: jnp.zeros(n*d),
                               lambda t,x,theta: Q12(x,theta))
else:
    b = lambda t,x,theta: jnp.zeros(n*d)
    sigma = lambda t,x,theta: Q12(x,theta)


bphylogeny_sim = Tree(simtree)
leafidx = []
inneridx = []
i = 0
for node in bphylogeny_sim.traverse('levelorder'):
    if node.is_leaf():
        print(node.name)
        leafidx.append(i)
    else:
        inneridx.append(i)
    i+=1
print(leafidx)
print(inneridx)

#%%
key = jax.random.PRNGKey(seed_sim_data)
key, subkey = jax.random.split(key)
bphylogeny_sim.dist = rb
for node in bphylogeny_sim.traverse("levelorder"): 
    node.add_feature('T', round(node.dist,1)) # this is a choice for simulation, could be different
    node.add_feature('n_steps', round(node.T/dt))
    node.add_feature('message', None)
    node.dist = node.T
    
if rb==0:
    key, *subkeys = jax.random.split(key, len(bphylogeny_sim.children)+1)
    _dts = jnp.array([0]); _dWs = jnp.array([0]); Xscirc = root.reshape(1,-1) # set variables for root 
    children = [bphylogeny_sim.children[0],bphylogeny_sim.children[1]]
    dWs_children = [dtsdWsT(bphylogeny_sim.children[i],subkeys[i], lambda ckey, _dts: dWs(n*d,ckey, _dts)) for i in range(len(bphylogeny_sim.children))]
    stree = [_dts, _dWs, Xscirc, [simulate_tree(Xscirc[-1], b, sigma, theta_true, dtsdWs_child) for dtsdWs_child in dWs_children]]
else:
    stree = simulate_tree(root, b, sigma, theta_true, dtsdWsT(bphylogeny_sim,subkey, lambda ckey, _dts: dWs(n*d,ckey, _dts)))
flat_true_tree = np.array(get_flat_values_sim(stree)) 
np.savetxt(outputpath+'flat_true_tree.csv', flat_true_tree, delimiter=",")
print(bphylogeny_sim)
print(f'super root branch length for simulation: {bphylogeny_sim.dist}')
bphylogeny_sim.write(format=1, outfile=outputpath+"phylogeny.nw")

#%%
leaves = flat_true_tree[leafidx,:]
if obs_var!=0:
    obs_noise=np.random.normal(loc=0.0, scale=np.sqrt(obs_var), size=(len(leaves),leaves[0].shape[0]))
    obs_leaves = np.array(leaves) + obs_noise 
else: 
    obs_leaves = np.array(leaves)
np.savetxt(outputpath+'leaves.csv', obs_leaves, delimiter=",")
np.savetxt(outputpath+'leaves_no_noise.csv', leaves, delimiter=",")

# %%
# SAVE FOR MCMC
np.savetxt(outputpath+'gtheta_sim.csv', np.array([sigma_sim]))
np.savetxt(outputpath+'kalpha_sim.csv', np.array([alpha_sim]))
np.savetxt(outputpath+'root.csv',root, delimiter=',')

# GET VCV by running 
path = outputpath + 'phylogeny'
print(path)
subprocess.call('/usr/local/bin/Rscript get_vcv.R ' + path, shell=True)


