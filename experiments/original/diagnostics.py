#ls _root-exp/e1/runs | while read seed; do python diagnostics.py -MCMC_iter 3000 -burnin 1000 -nnodes 15 -simtree _root-exp/simulation-setup/symmetric_sim.nw -folder_runs _root-exp/e1/runs/$seed -folder_simdata _root-exp/e1/simdata/$seed; done


import os
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.pyplot as plt
import arviz
import seaborn as sns
from ete3 import Tree
import argparse
import arviz as az
import gc

from plotting import *

parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-folder_runs', help = 'folder for plots', nargs='?', type=str)
parser.add_argument('-folder_simdata', help = 'folder for plots', nargs='?', type=str)
parser.add_argument('-MCMC_iter', help = '', nargs='?', type=int)
parser.add_argument('-burnin', help = '', nargs='?', type=int)
parser.add_argument('-nnodes', help = '', nargs='?', type=int, default=9)
parser.add_argument('-simtree', help = '', nargs='?', type=str, default = 'chazot_subtree_levelorder.nw')
parser.add_argument('-nxd', help = '', nargs='?', type=int, default=40)
args = parser.parse_args()

#%%
MCMC_iter = args.MCMC_iter
burnin = args.burnin
nthin = 1 # see from script/running conditions, not used for plotting
folder_runs = args.folder_runs +'/' 
folder_simdata = args.folder_simdata +'/' 
nnodes = args.nnodes
simtree = args.simtree
nxd = args.nxd
#thresh_kdeplot_2D = 0.05 # probability mass below last contour line
pars_name = ['kalpha', 'gtheta']
rep_path = len(pars_name)+1
chains = os.listdir(folder_runs) # use all chains in data seed folder 
chains = [c for c in chains if c[0] not in ['_', '.']] # remove files starting with underscore
print(chains)
temp_name = ['' for i in range(len(chains))]



# create output folder
outputfolder = folder_runs
path = outputfolder+'_*'+'-'.join(chains)
if not os.path.isdir(path): 
    os.mkdir(path)
if not os.path.isdir(path+'/stats'): 
    os.mkdir(path+'/stats')

tree = Tree(simtree)
leafidx = []
inneridx = []
i = 0
for node in tree.traverse('levelorder'):
    if node.is_leaf():
        print(node.name)
        leafidx.append(i)
    else:
        inneridx.append(i)
    i+=1
print(leafidx)
print(inneridx)

# read in data and MCMC chains
raw_trees = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"tree_nodes.csv", delimiter = ",") for i in range(len(chains))]
tree_counters = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"tree_counter.csv", delimiter = ",").astype(int) for i in range(len(chains))]
flat_trees_raw = [raw_trees[i].reshape(len(tree_counters[i]),nnodes,nxd) for i in range(len(raw_trees))]
flat_true_tree = np.genfromtxt(folder_simdata+"flat_true_tree.csv", delimiter = ",")
super_root = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"inference_root_start.csv", delimiter = ",") for i in range(len(chains))]
_super_root = [np.concatenate((super_root[i], super_root[i][0:2])) for i in range(len(chains))]
_super_root = np.unique(np.array(_super_root), axis=0)
flat_trees = np.array([np.repeat(flat_trees_raw[i], tree_counters[i], axis=0)[burnin*rep_path:(MCMC_iter//nthin)*rep_path] for i in range(len(flat_trees_raw))])
flat_trees.shape


# get rhat and ESS for all nodes and dimensions
rhats = []
esss = []
for idx in range(flat_trees.shape[2]):  # calculate for all nodes 
    innernodes = flat_trees[:,:,idx, :]
    keys = list(range(innernodes.shape[2]))
    MCMCres = arviz.convert_to_dataset({k:innernodes[:,:,i] for i,k in enumerate(keys)})
    rhats.append(arviz.rhat(MCMCres).to_array().to_numpy())
    esss.append(arviz.ess(MCMCres).to_array().to_numpy())

# save rhat for plotting
np.savetxt(path+'/stats/' +"rhats_paths.csv",np.array(rhats), delimiter=",")

# plot traces
plot_traces_tree(flat_trees, inneridx, esss, rhats, flat_true_tree, true_innernode=True, outpath = path + f'/trace-innernodes_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')

# plot summary of Rhat
summary_rhat(rhats, inneridx, outpath = path + f'/summary_rhat_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')

# Plot all Rhat in one figure 
sns.violinplot(np.array(rhats).flatten())
plt.hlines(y=1.1, xmin=-0.5, xmax=0.5, color='red', linestyle='--')
plt.savefig(path+ f'/summary_rhat_allnodes_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')

# Plot full posterior 
plot_posterior(flat_trees, inneridx, outpath = path + f'/posterior_samples_burnin={burnin}_MCMCiter={MCMC_iter}.png', flat_true_tree=flat_true_tree, sample_n=50)

# Plot data 
plot_leaves(flat_true_tree, leafidx, outpath = path + f'/leaves_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')
