import numpy as np
import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import gc

def plot_traces_tree(flat_trees, inneridx, ess, rhats, flat_true_tree, true_innernode= False, outpath = 'traces.pdf'): 
    colors = sns.color_palette('pastel', flat_trees.shape[0])
    pdf = backend_pdf.PdfPages(outpath)
    plt.figure(1)
    for idx in inneridx: # loop over innernodes
        print(idx)
        fig, axes = plt.subplots(nrows=7, ncols=6, figsize=(25,15), sharex=True)
        for j in range(flat_trees.shape[0]): # loop over chains
            innernode = flat_trees[j,:,idx, :]
            if true_innernode is not False: 
                true_innernode = flat_true_tree[idx,:]
            curcol = colors[j]
            for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
                ax.plot(innernode[:,i], color = curcol, alpha=0.5)
                if true_innernode is not False:
                    ax.hlines(y=true_innernode[i], xmin=0, xmax=innernode.shape[0], color='skyblue')
                cur_ess = ess[idx][i]
                cur_rhat = rhats[idx][i]
                ax.set_title(f'{i}, Rhat={round(cur_rhat,2)}, ESS: {round(cur_ess,2)}')
        fig.suptitle(f'Node {idx}', size=40)
        pdf.savefig()
        plt.clf()
    pdf.close();

def summary_rhat(rhats, inneridx, outpath):
    # plot summary of Rhat values 
    n_nodes = len(inneridx)
    #print(n_nodes)
    # Determine the grid size
    grid_size = int(np.ceil(np.sqrt(n_nodes)))

    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle(r'$\hat{R}$ for all innernodes')

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create a scatter plot for each k
    for i in range(n_nodes):
        sns.violinplot(rhats[inneridx[i]], ax=axes[i])
        axes[i].hlines(y=1.1, xmin=-0.5, xmax=0.5, color='red', linestyle='--')
        axes[i].set_title(f'Node={list(inneridx)[i]}')

    # Hide any unused subplots
    for j in range(n_nodes, grid_size * grid_size):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(outpath)
    fig.subplots_adjust(top=0.95)
    #plt.show()
    plt.close()
    gc.collect()


def plot_posterior(flat_trees, inneridx, outpath, flat_true_tree=False, sample_n=50, nxd=40):
    # plot summary of Rhat values 
    n_nodes = len(inneridx)
    # Determine the grid size
    grid_size = int(np.ceil(np.sqrt(n_nodes)))
    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), sharex=True, sharey=True)
    #fig.suptitle(f'Samples from posterior (every {sample_n}) for all innernodes', size=20)
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create a scatter plot for each k
    for i in range(n_nodes):
        idx = inneridx[i]
        innernodes = flat_trees[:,:,idx,:].reshape(-1, nxd)[::sample_n,:]
        inode = innernodes #np.append(innernodes, innernodes[:,0:2],1)
        for j in range(inode.shape[0]):
            axes[i].plot(inode[j,::2], inode[j,1::2], '--.', color='steelblue', alpha=0.3)
        if flat_true_tree is not False:
            true_innernode = flat_true_tree[idx,:]
            tinode = np.concatenate((true_innernode, true_innernode[0:2]))  
            axes[i].plot(tinode[::2], tinode[1::2], '--.', color='black', label='True shape')
        axes[i].set_title(f'Node {idx}', size=20);

    # Hide any unused subplots
    for j in range(n_nodes, grid_size * grid_size):
        fig.delaxes(axes[j])

   # plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight') #
    #fig.subplots_adjust(top=0.95)
    #plt.show()
    plt.close()
    gc.collect()


def plot_leaves(flat_true_tree, leafidx, outpath):
    # plot summary of Rhat values 
    n_nodes = len(leafidx)
    # Determine the grid size
    grid_size = int(np.ceil(np.sqrt(n_nodes)))
    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), sharex=True, sharey=True)
    fig.suptitle(f'Observed data (i.e. leaves)', size=20)
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    root = flat_true_tree[0,:]
    # Create a scatter plot for each k
    for i in range(n_nodes):
        idx = leafidx[i]
        leaf = flat_true_tree[idx,:]
        leafp = np.append(leaf, leaf[0:2],0)
        axes[i].plot(leafp[::2], leafp[1::2], '--.', color='orange', lw=2)
        #rootp = np.concatenate((root, root[0:2]))  
        #axes[i].plot(rootp[::2], rootp[1::2], '--.', color='black', label='Root')
        #axes[i].set_title(f'Node {idx}', size=10);

    # Hide any unused subplots
    for j in range(n_nodes, grid_size * grid_size):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(outpath)
    fig.subplots_adjust(top=0.95)
    #plt.show()
    plt.close()
    gc.collect()


def get_mode(all_chains, bw='silverman'): 
    '''Function for marginal mode estimation '''
    mode_est = []
    for i in range(all_chains.shape[1]):
        kdes = az.kde(all_chains[:,i].flatten(), bw=bw)
        mest = kdes[0][np.where(kdes[1] == max(kdes[1]))[0][0]]
        mode_est.append(mest)
    mode_est = np.array(mode_est)
    return(mode_est)