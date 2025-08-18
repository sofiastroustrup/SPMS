# make a script that reads all the rank statistics from the different folders 
# and plots the distribution of the ranks for each dimension in each node in the tree

#%%
import os
import numpy as np
import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.pyplot as plt
from ete3 import Tree
import scipy
import argparse

#%%
parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-folder_runs', help='path to folder with runs', nargs='?',  type=str )
parser.add_argument('-tree', help='path to treefile', nargs='?', type=str )
args = parser.parse_args()

tree = Tree(args.tree)
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


_folder = args.folder_runs #'BM14/runs'
bfolder = os.listdir(_folder)
bfolder_ignore = ['.DS_Store', '_', 'figures', 'sampled_parameters.pdf','mse_pars.pdf', 'mse_paths.pdf', 'bias_pars.pdf','bias_paths.pdf', 'all_ranks.pdf', 'ranks.pdf','wandb', 'rhat-pars.pdf', 'rhat-paths.pdf', 'ranks_kalpha.pdf', 'ranks_gtheta.pdf', 'ranks_gtheta_sqrt.pdf', '_dataseeds5_.csv']#['.DS_Store', 'wandb', 'not_converged', 'bamf-211024', 'all_ranks.pdf', 'ranks.pdf',  '_dataseeds0_.csv', 'ranks.pdf',  '_dataseeds1_.csv', '_dataseeds2_.csv', '_dataseeds3_.csv', '_dataseeds4_.csv', '_dataseeds5_.csv']
[bfolder.remove(bi) for bi in bfolder_ignore if bi in bfolder] # ignore specific files

#_all_ris_gtheta_sqrt = []
_all_ris_gtheta = []
_all_ris_kalpha = []
_all_ris = []
for subfolder in bfolder: 
    folder = _folder+'/'+subfolder+"/"
    cri = np.genfromtxt(folder+'_ri.csv')
    cra = np.genfromtxt(folder+'_ri_kalpha.csv')
    crt = np.genfromtxt(folder+'_ri_gtheta.csv')
    _all_ris.append(cri)
    _all_ris_gtheta.append(crt)
    _all_ris_kalpha.append(cra)

# %%
#all_ris_gtheta_sqrt = np.array(_all_ris_gtheta_sqrt)
all_ris_gtheta = np.array(_all_ris_gtheta)
all_ris_kalpha = np.array(_all_ris_kalpha)
all_ris = np.array(_all_ris)
n = all_ris.shape[0]
nbins=40 #int(round(all_ris_gtheta.shape[0]/3,1))
q005 = scipy.stats.binom.ppf(0.005,n,1/nbins)
q995 = scipy.stats.binom.ppf(0.995,n,1/nbins)

#%%
plt.hist(all_ris_gtheta, bins=nbins, color='steelblue', zorder=2)
plt.hlines(n/nbins, 0,1, color='olive', linestyles='dashed', lw=1, zorder=3)
plt.fill_between(x=np.arange(0, 1,0.001), y1=q005, y2=q995, color='grey', alpha=0.2, zorder=1)
plt.xlim(0,1)
plt.title(r'All ranks $\sigma$'+f' (n={n})')
plt.savefig(_folder + f'/figures/ranks_gtheta.pdf')
plt.close()


plt.hist(all_ris_kalpha, bins=nbins, color='steelblue', zorder=2)
plt.hlines(n/nbins, 0,1, color='olive', linestyles='dashed', lw=1, zorder=3)
plt.fill_between(x=np.arange(0, 1,0.001), y1=q005, y2=q995, color='grey', alpha=0.2, zorder=1)
plt.xlim(0,1)
plt.title(r'All ranks $\alpha$'+f' (n={n})')
plt.savefig(_folder + f'/figures/ranks_kalpha.pdf')
plt.close()


# %%
#colors = sns.color_palette('pastel', len(chains))
pdf = backend_pdf.PdfPages(_folder + f'/figures/ranks.pdf')
plt.figure(1)
for idx in list(inneridx): # loop over innernodes #range(all_ris.shape[1]): # loop over nodes
    fig, axes = plt.subplots(nrows=7, ncols=6, figsize=(25,15), sharex=True)
    innernode = all_ris[:,idx,:] #flat_trees[j,:,idx, :]
    for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
        ax.hist(innernode[:,i], bins=nbins, color='steelblue', zorder=2)
        ax.hlines(n/nbins, 0,1, color='olive', linestyles='dashed', lw=1, zorder=3)
        ax.fill_between(x=np.arange(0, 1,0.001), y1=q005, y2=q995, color='grey', alpha=0.2, zorder=1)
    fig.suptitle(f'Node {idx} (n={n})', size=40)
    pdf.savefig()
    plt.clf()
pdf.close();
# %%
# plot for paper 

idx = 0 # plot for root node 
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(25,15), sharex=True, sharey=True)
innernode = all_ris[:,idx,:] #flat_trees[j,:,idx, :]
for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
    ax.hist(innernode[:,i], bins=nbins, color='steelblue', zorder=2)
    ax.hlines(n/nbins, 0,1, color='olive', linestyles='dashed', lw=1, zorder=3)
    ax.fill_between(x=np.arange(0, 1,0.001), y1=q005, y2=q995, color='grey', alpha=0.2, zorder=1)
plt.savefig(_folder + '/figures/landmarks12_ranks_root.pdf')


