import matplotlib.pyplot as plt
import numpy as np
import arviz 
import seaborn as sns
import matplotlib.backends.backend_pdf as backend_pdf


def compute_diagnostics(chain_results, burnin_percent):
    trees = np.array([chain_results[i]['trees'] for i in range(len(chain_results))])
    trees = trees[:, int(burnin_percent*trees.shape[1]):, :, :]  # discard burnin
    rhats = []
    esss = []
    for idx in range(trees.shape[2]):  # calculate for all nodes 
        innernodes = trees[:,:,idx, :]
        keys = list(range(innernodes.shape[2]))
        MCMCres = arviz.convert_to_dataset({k:innernodes[:,:,i] for i,k in enumerate(keys)})
        rhats.append(arviz.rhat(MCMCres).to_array().to_numpy())
        esss.append(arviz.ess(MCMCres).to_array().to_numpy())
    return {'Rhat': np.array(rhats), 'ESS': np.array(esss)}


def plot_traces(results, burnin_percent, node_idx, save_path, diagnostics=None, true_values=None): 
    colors = sns.color_palette('pastel', len(results))
    pdf = backend_pdf.PdfPages(save_path + f'/trace_burnin_percent={burnin_percent}.pdf')
    burnin_end = int(results[0]['trees'].shape[0] * burnin_percent)
    plt.figure(1)

    for idx in node_idx: 
        fig, axes = plt.subplots(nrows=7, ncols=6, figsize=(25,15), sharex=True)
        fig.subplots_adjust(top=0.9)  # Adjust this value between 0 and 1
        if true_values is not None: 
            true_innernode = true_values[idx,:]
        for j in range(len(results)): # loop over chains
            innernode = results[j]['trees'][:,idx, :]
            curcol = colors[j]
            for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
                ax.plot(innernode[burnin_end::,i], color = curcol, alpha=0.5)
                if diagnostics:
                    cur_ess = round(diagnostics['ESS'][idx][i], 2)
                    cur_rhat = round(diagnostics['Rhat'][idx][i], 2)
                if true_values is not None:
                    ax.hlines(y=true_innernode[i], xmin=0, xmax=innernode.shape[0]-burnin_end, color='skyblue')
                ax.set_title(f'{i}, Rhat={cur_rhat}, ESS: {cur_ess}')
        fig.suptitle(f'Node {idx}', size=25)
        pdf.savefig()
        plt.clf()
    pdf.close();
    
    
    
# define function for plotting parameter traces
def plot_parameter_traces(chain_results, param_names, burnin_percent, savepath=None):
    """
    Plot the traces of MCMC parameters.
    
    Args:
        chain_results: List of MCMC result dictionaries
        param_names: List of parameter names to plot
        burnin_percent: Percentage of samples to discard as burn-in
    """
    burnin_index = int(burnin_percent * len(chain_results[0][param_names[0]]))
    num_params = len(param_names)
    plt.figure(figsize=(15, 5 * num_params))
    
    # compute convergence diagnostics
    sigmas = [chain_results[i]['sigma'] for i in range(len(chain_results)) if chain_results[i] is not None]
    alphas = [chain_results[i]['alpha'] for i in range(len(chain_results)) if chain_results[i] is not None]
    MCMC_result = dict(zip(["alpha", "sigma"], [alphas, sigmas])) 
    parsres = arviz.convert_to_dataset(MCMC_result)
    rhat = arviz.rhat(parsres)
    ess = arviz.ess(parsres)
    
    for i, param in enumerate(param_names):
        plt.subplot(num_params, 1, i + 1)
        for j in range(len(chain_results)):
            if chain_results[j] is not None:
                plt.plot(chain_results[j][param][burnin_index:], label=f'Chain {j}')
        plt.xlabel('Iteration')
        plt.ylabel(param)
        plt.title(f'MCMC Trace for {param} (R-hat: {rhat[param]:.2f}, ESS: {ess[param]:.1f})')
        plt.legend()
    
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.show()

def plot_log_posterior(result, burnin_percent, save_path=None): 
    """
    Plot the posterior distribution of the MCMC results.
    
    Args:
        result: MCMC result dictionary
        burnin_percent: Percentage of samples to discard as burn-in
    """
    # Compute burn-in index
    burnin_index = int(burnin_percent * len(result[0]['log_posterior']))
    
    # Plot log posterior
    plt.figure(figsize=(10, 5))
    [plt.plot(result[i]['log_posterior'][burnin_index:], label=f'Chain {i}')
     for i in range(len(result))]
    plt.xlabel('Iteration')
    plt.ylabel('Log Posterior')
    plt.title('MCMC Log Posterior Trace')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_samples_from_posterior(chain_results, burnin_percent, node_idx, sample_every, savepath, true_values=None):
    pdf = backend_pdf.PdfPages(savepath + f'/samples-posterior-sample_n={sample_every}_burnin_percent={burnin_percent}.pdf')
    
    burnin_end = int(chain_results[0]['trees'].shape[0] * burnin_percent)
    print(f"Burnin ends at index: {burnin_end}")
    for idx in node_idx: # loop over innernodes
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        # Get all samples from all chains
        all_samples = []
        for j in range(len(chain_results)):
            # Make sure we have data after burnin
            post_burnin_data = chain_results[j]['trees'][burnin_end:, idx, :]
            #print(post_burnin_data)
            
            if len(post_burnin_data) > 0:
                thinned_data = post_burnin_data[0::sample_every, :]
                all_samples.append(thinned_data)
        
        # Combine all chains
        if all_samples:
            innernodes = np.vstack(all_samples)
            
            # Debug info
            print(f"Node {idx}: Found {len(innernodes)} samples, shape={innernodes.shape}")
            
            # Close the shape by appending first landmarks
            inode = np.append(innernodes, innernodes[:, 0:2], axis=1)
            
            # Plot each shape sample
            for i in range(inode.shape[0]):
                x_coords = inode[i, 0::2]  # Every other element, starting at 0 (x coordinates)
                y_coords = inode[i, 1::2]  # Every other element, starting at 1 (y coordinates)
                axes.plot(x_coords, y_coords, '--.', color='steelblue', alpha=0.3)
            
            # Add true values if provided
            if true_values is not None:
                true_innernode = true_values[idx, :]
                tinode = np.concatenate((true_innernode, true_innernode[0:2]))  
                axes.plot(tinode[::2], tinode[1::2], '--.', color='black', linewidth=2, label='true shape')
            
            # Add title and format
            fig.suptitle(f'Node {idx}', size=25)
            fig.tight_layout()
            axes.set_aspect('equal')  # Equal aspect ratio
            axes.grid(True, alpha=0.3)
            
            # Save the figure
            pdf.savefig(fig)
            
        plt.close(fig)  # Close the figure to free memory
    
    pdf.close()  # Close the PDF
    print(f"Saved plots to {savepath}/samples-posterior-sample_n={sample_every}_burnin_percent={burnin_percent}.pdf")




    