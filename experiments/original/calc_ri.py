#%%
# script for evaluating results from benchmark
import os
import numpy as np
import argparse
#from ete3 import Tree

parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-N', help = 'MCMC iter', nargs='?', default=10, type=int)
parser.add_argument('-burnin', help='burnin', nargs='?', default = 10, type=int )
parser.add_argument('-nnodes', help='number of nodes in tree', nargs='?', default = 10, type=int )
parser.add_argument('-nxd', help='num_landmarks x dimensions', nargs='?', default = 40, type=int )
parser.add_argument('-folder_simdata', help='path to folder with simdata', nargs='?', default = 40, type=str )
parser.add_argument('-folder_runs', help='path to folder with runs', nargs='?', default = 40, type=str )
args = parser.parse_args()

# %%
# general settings 
MCMC_iter = args.N #5000
burnin = args.burnin #1500
nthin = 1 
nnodes = args.nnodes #9
nxd = args.nxd #40
rep_path=1

#proj = 'BM14'
_folder = args.folder_runs #proj + '/resub'


_folder_simdata = args.folder_simdata #proj + '/simdata'
bfolder = os.listdir(_folder)
bfolder_ignore = ['.DS_Store','figures', 'bias_paths.pdf','bias_pars.pdf','mse_pars.pdf', 'mse_paths.pdf', 'sampled_parameters.pdf','all_ranks.pdf', 'ranks.pdf','wandb', 'rhat-pars.pdf', 'rhat-paths.pdf', 'ranks_kalpha.pdf', 'ranks_gtheta.pdf', '_']
[bfolder.remove(bi) for bi in bfolder_ignore if bi in bfolder] # ignore specific files
print(len(bfolder))
# %%
for subfolder in bfolder: 
    #if '_ri.csv' in os.listdir(_folder + '/' + subfolder): 
    #    print('ri')
    #    continue
    folder = _folder+'/'+subfolder+"/"
    simfolder = _folder_simdata +'/'+subfolder+"/"
    print(folder)
    chains = os.listdir(folder)
    chains[:] = [x for x in chains if not x.startswith('_')]

    raw_trees = [np.genfromtxt(folder + chains[i]+'/'+"tree_nodes.csv", delimiter = ",") for i in range(len(chains))]
    tree_counters = [np.genfromtxt(folder + chains[i]+'/'+"tree_counter.csv", delimiter = ",").astype(int) for i in range(len(chains))]
    flat_trees_raw = [raw_trees[i].reshape(len(tree_counters[i]),nnodes,nxd) for i in range(len(raw_trees))]
    flat_true_tree = np.genfromtxt(simfolder+"flat_true_tree.csv", delimiter = ",")
    flat_trees = np.array([np.repeat(flat_trees_raw[i], tree_counters[i], axis=0)[burnin*rep_path:(MCMC_iter//nthin)*rep_path] for i in range(len(flat_trees_raw))])

    gthetas = [np.genfromtxt(folder + chains[i]+'/'+"gthetas.csv", delimiter = ",") for i in range(len(chains))]
    gtheta_true = np.unique([np.genfromtxt(folder + chains[i]+'/'+"true_gtheta.csv", delimiter = ",") for i in range(len(chains))])
    kalphas = [np.genfromtxt(folder + chains[i]+'/'+"kalphas.csv", delimiter = ",") for i in range(len(chains))]
    kalpha_true = np.unique([np.genfromtxt(folder + chains[i]+'/'+"true_kalpha.csv", delimiter = ",") for i in range(len(chains))])

    rank_kalpha = np.mean(np.array(kalphas).flatten()<kalpha_true)
    rank_gtheta = np.mean(np.array(gthetas).flatten()<gtheta_true)

    ranks = []
    for ni in range(flat_true_tree.shape[0]):
        ri = np.mean(flat_trees[:,:,ni,:].reshape(-1, nxd)<flat_true_tree[ni], axis=0)
        ranks.append(ri)
    ris = np.array(ranks)
    np.savetxt(folder+'/_ri.csv', ris)
    np.savetxt(folder+'/_ri_kalpha.csv', np.array([rank_kalpha]))
    np.savetxt(folder+'/_ri_gtheta.csv', np.array([rank_gtheta]))
# %%
