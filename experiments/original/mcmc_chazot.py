import os
import numpy as np
from ete3 import Tree
import jax
import jax.numpy as jnp
from tqdm import tqdm 
import pandas as pd
import wandb
import argparse
import scipy
import subprocess

from bridge_sampling.BFFG import backward_filter, forward_guide, forward_guide_edge, get_logpsi
from bridge_sampling.setup_SDEs import Stratonovich_to_Ito, dtsdWsT, dWs
from bridge_sampling.noise_kernel import Q12
from bridge_sampling.helper_functions import *


## PARSE ARGUMENTS
parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-N', help = 'MCMC iter', nargs='?', default=10, type=int)
parser.add_argument('-l', help = 'Crank-nicholson lambda', nargs='?', default=0, type=float)
parser.add_argument('-dt', help = 'dt for MCMC', nargs='?', default=0.1, type=float)
parser.add_argument('-ov', help = 'obs_var', nargs='?', default=0.01, type=float)
parser.add_argument('-sti', help = 'Use stratonovich-ito correction', nargs='?', 
                    default=1, type=int)
parser.add_argument('-ms', help = 'seed for MCMC', nargs='?', type = int, default = 0)
parser.add_argument('-datapath', help='path to data i.e. observations at leaves', nargs='?', default = 'simdata/leaves.csv', type=str )
#parser.add_argument('-treepath', help='path to phylogeny', nargs='?', default = 'simdata/phylogeny.nw', type=str )
parser.add_argument('-nxd', help='number of landmarks and dimension of landmarks', nargs=2, default = [20, 2], type=int )
parser.add_argument('-wandb', help='wandb project', nargs='?', default = '', type=str )
parser.add_argument('-tau_sigma', help='proposal sd gtheta', nargs='?', default=0.02, type=float)
parser.add_argument('-tau_alpha', help='proposal sd kalpha', nargs='?', default=0.02, type=float)
parser.add_argument('-o', help='path to output folder', nargs='?', default = 'runs', type=str )
parser.add_argument('-palpha', help='prior kalpha (uniform, loc scale)', nargs=2, default=[0.005, 0.025], metavar= ('loc', 'scale'), type=float)
parser.add_argument('-psigma', help='prior gtheta (uniform, loc scale)', nargs=2, default=[0.8, 0.4], metavar= ('loc', 'scale'), type=float)
parser.add_argument('-super_root', help='path to super root', nargs='?', default='shapes/circle_rotated_n=20.csv')
parser.add_argument('-rb', help='dist from super root to root', nargs='?', default = 10, type=float )

args = parser.parse_args()
seed_mcmc = args.ms
N = args.N
lambd = args.l
dt = args.dt
obs_var = args.ov
length_root_branch=args.rb
proposal_sd_kalpha = args.tau_alpha
proposal_sd_gtheta = args.tau_sigma
datapath = args.datapath
wandb_project = args.wandb
d = args.nxd[1]
n = args.nxd[0]
#treefile = args.treepath
datapath = args.datapath


### MCMC setting for prior ###
kalpha_loc = args.palpha[0]
kalpha_scale = args.palpha[1]
gtheta_loc = args.psigma[0]
gtheta_scale = args.psigma[1]

# INITIATE WANDB PROJECT
wandb.init(project=wandb_project, settings=wandb.Settings(_service_wait=1000))

# SETUP DIRECTORY FOR OUTPUT
outputpath = args.o + '/' + wandb.run.name + '/'
cur_dir = os.getcwd()
path = cur_dir +'/'+ outputpath
if not os.path.isdir(path): 
    os.makedirs(path)

# PRINT MCMC SETTINGS TO STDOUT
if seed_mcmc==0: 
    seed_mcmc = np.random.randint(100000000000000000, size=1)[0]
    print('!! no MCMC seed given')

print('Running MCMC on simulated data')
print(f'All data is saved in {outputpath}')
print(f'Run name: {wandb.run.name}')
print(f'Run ID: {wandb.run.id}')
print(f'Use stratonovich-to-Ito correction: {args.sti}')
print('***')
print('Runnings MCMC with the following settings:')
print(f'MCMC iter = {N}')
print(f'Crank-Nicholson lambda = {lambd}')
print(f'Proposal sd kalpha = {proposal_sd_kalpha}')
print(f'Proposal sd gtheta = {proposal_sd_gtheta}')
print(f'Stepsize = {dt}')
print(f'Noise on endpoint = {obs_var}')
print(f'seed MCMC = {seed_mcmc}')
print(f'length root branch: {length_root_branch}')
print(f'uniform prior on kalpha, loc={kalpha_loc}, scale={kalpha_scale}')
print(f'uniform prior on gtheta, loc={gtheta_loc}, scale={gtheta_scale}')

# SET UP STOCHASTIC PROCESS
# define drift and diffusion for process of interest 
if args.sti ==1:
    b,sigma,_ = Stratonovich_to_Ito(lambda t,x,theta: jnp.zeros(n*d),
                               lambda t,x,theta: Q12(x,theta))
else:
    b = lambda t,x,theta: jnp.zeros(n*d)
    sigma = lambda t,x,theta: Q12(x,theta)

# READ IN DATA (RIGHT NOW SETUP FOR SIMULATED DATA)
treefile = datapath + '/chazot_full_tree.nw'
with open(treefile, 'r') as file: 
        newick_tree = file.read()
bphylogeny = Tree(treefile)

# read data + metadata
leaves = pd.read_csv(datapath+'/forewing_data_male.csv', delimiter=',')

# prep tree inference tree
bphylogeny.dist = length_root_branch # we need to add the super root branch length because it is not saved in the newick file... 
for node in bphylogeny.traverse("levelorder"): 
    node.add_feature('T', round(node.dist,1)) 
    node.add_feature('message', None)
    node.add_feature('theta', False)
    node.add_feature('n_steps', int(node.T/dt)) # obs the rounded node should be divisable with the stepsize


for leaf in bphylogeny: 
    print(leaf.name)
    leaf.add_feature('v', jnp.array(leaves[leaf.name]))
    leaf.add_feature('obs_var', obs_var)

# RUN MCMC
#### Initiate MCMC chain ####

# initiate parameters and root 
key = jax.random.PRNGKey(seed_mcmc)
key, *subkeys = jax.random.split(key,3)
kalpha_cur = jax.random.uniform(subkeys[0], (1,), minval=kalpha_loc, maxval=kalpha_loc+kalpha_scale)[0]
gtheta_cur = jax.random.uniform(subkeys[1], (1,), minval=gtheta_loc, maxval=gtheta_loc+gtheta_scale)[0]
#leaves = bphylogeny.get_leaves()

if args.super_root == 'mean':
    print('super_root: euclidean mean')
    super_root = np.mean(leaves, axis=0)
elif args.super_root == 'phylomean':
    print('super_root: phylogenetic mean')
    _path = datapath + '/chazot_full_tree'
    print(_path)
    subprocess.call('Rscript get_vcv.R ' + _path, shell=True)
    vcv = np.genfromtxt(datapath + '/chazot_full_tree_vcv.csv', delimiter=' ')
    leaves = jnp.array(leaves).T
    super_root = 1/(np.ones(leaves.shape[0]).T@np.linalg.inv(vcv)@np.ones(leaves.shape[0]))*np.ones(leaves.shape[0]).T@np.linalg.inv(vcv)@leaves # update this for more dynamic code
else:
    print(f'super root: {args.super_root}')
    super_root = np.genfromtxt(args.super_root, delimiter=',')

print(f'Inference super root: {super_root}')
print(f'kalpha start: {kalpha_cur}')
print(f'gtheta start: {gtheta_cur}')


np.savetxt(outputpath+'inference_root_start.csv', super_root, delimiter=",")

# backwards filter
# set theta for inference
theta_cur = {
    'k_alpha': kalpha_cur, # kernel amplitude
    'inv_k_sigma': 1./(gtheta_cur)*jnp.eye(d),
    'd':d,
    'n':n, 
}

# backwards filter 
data_tree_bf = backward_filter(bphylogeny, theta_cur, sigma)

# get Wiener process and steps on entire tree
key, subkey = jax.random.split(key,2) 
_dtsdWsT = dtsdWsT(bphylogeny, subkey, lambda ckey,_dts: dWs(n*d,ckey,_dts))

# Initiate tree
fge = jax.jit(lambda *x: forward_guide_edge(*x, b, sigma, theta_cur))
initialized_tree = forward_guide(super_root,data_tree_bf,_dtsdWsT, fge) 
logpsicur = get_logpsi(initialized_tree)
logrhotildecur = -data_tree_bf.message['c']-0.5*super_root.T@data_tree_bf.message['H'][0]@super_root+data_tree_bf.message['F'][0].T@super_root


# results 
guided_tree = get_flat_values(initialized_tree) #used to be get_flat_values_root_branch
trees = np.expand_dims(guided_tree, axis=0)
tree_counter = [1]

kalphas = [kalpha_cur]
gthetas = [gtheta_cur]


wandb.config.update({
    "dt": dt,
    'obs_var': obs_var,
    'proposal_sd_kalpha': proposal_sd_kalpha,
    'proposal_sd_gtheta': proposal_sd_gtheta,
    'kalpha uniform prior loc': kalpha_loc, 
    'kalpha uniform prior scale': kalpha_scale,
    'gtheta uniform prior loc': gtheta_loc,
    'gtheta uniform prior scale': gtheta_scale, 
    'cranknicholson_lambda':lambd, 
    'seed_mcmc': str(seed_mcmc), 
    'k_alpha_start': kalpha_cur,
    'gtheta_start': gtheta_cur,
    'MCMC_iter': N, 
    'length root branch': length_root_branch,
    'comments': f'stratonovich-ito correction = {args.sti}, inference_root_start = {args.super_root} '
    })


wandb.save(outputpath+'flat_true_tree.csv')
wandb.save(outputpath+'true_gtheta.csv')
wandb.save(outputpath+'true_kalpha.csv')
wandb.save(outputpath+'simulated_tree.pdf')
wandb.save(outputpath+'guided_tree.pdf')
wandb.save(outputpath+'cur_tree.nw')
wandb.save(outputpath+'inference_root_start.csv')

#gtheta = theta['gtheta'] # set starting parameter equal to true parameter
acceptpath = np.zeros(N+1)
acceptgtheta = np.zeros(N+1)
acceptkalpha = np.zeros(N+1)
acceptpathall = []

for j in tqdm(range(N)):
    #######################
    ## propose path/tree ##
    #######################
    key, subkey = jax.random.split(key, 2)

    # take a step
    _dtsdWsTcirc = crank_nicholson_step(subkey, _dtsdWsT, lambd)
    guidedcirc = forward_guide(super_root, data_tree_bf,_dtsdWsTcirc, fge)
    logpsicirc = get_logpsi(guidedcirc)
    
    # calculate acceptance probability
    log_r = logpsicirc - logpsicur
    A = min(1, np.exp(log_r))
    print(f'path acceptance probability {A}')

    key, subkey = jax.random.split(key, 2)
    if jax.random.uniform(subkey)<A:
        # update driving noise 
        _dtsdWsT = _dtsdWsTcirc

        # update probabilities
        logpsicur = logpsicirc

        # update statistics
        acceptpath[j+1] = 1
        acceptpathall.append(1)
        
        # save new paths 
        guided_tree = get_flat_values(guidedcirc) #used to be get_flat_values_root_branch
        trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
        tree_counter.append(1)

    else: 
        acceptpathall.append(0)
        tree_counter[-1]+=1    
    # log
    inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
    tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
    tolog.update(inner)
    wandb.log(tolog)


    #######################
    ##   propose gtheta  ##
    #######################
    
    # propose parameter, proposal is mirrored gaussian with sd
    key, subkey = jax.random.split(key, 2)
    gthetacirc = mirrored_gaussian(subkey, gtheta_cur, proposal_sd_gtheta, 0, 10)  
    thetacirc = theta_cur.copy()
    thetacirc['inv_k_sigma']= 1./(gthetacirc)*jnp.eye(d) # update kernel width

    # do backwards filter using new parameter
    tree_bf_circ = backward_filter(bphylogeny, thetacirc, sigma)
    # get paths for new parameter same wiener process 
    fgecirc = jax.jit(lambda *x: forward_guide_edge(*x, b, sigma, thetacirc))
    guidedcirc = forward_guide(super_root, tree_bf_circ,_dtsdWsT, fgecirc)  
    logpsicirc = get_logpsi(guidedcirc)
    logrhotildecirc = -tree_bf_circ.message['c']-0.5*super_root.T@tree_bf_circ.message['H'][0]@super_root+tree_bf_circ.message['F'][0].T@super_root
    
    # get acceptance probability
    log_r = logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(gthetacirc, loc=gtheta_loc, scale=gtheta_scale) - scipy.stats.uniform.logpdf(gtheta_cur, loc=gtheta_loc, scale=gtheta_scale) #+ q_gt_gtcirc - q_gtcirc_gt
    A = min(1, np.exp(log_r))
    print(f'gtheta acceptance probability {A}')

    key, subkey = jax.random.split(key, 2)
    if jax.random.uniform(subkey)<A: 
        # update variables
        gtheta_cur = gthetacirc
        data_tree_bf = tree_bf_circ
        theta_cur = thetacirc
        fge = fgecirc

        # update probabilities
        logrhotildecur = logrhotildecirc 
        logpsicur = logpsicirc

        # update statistics 
        acceptgtheta[j+1] = 1

        # save new paths
        guided_tree = get_flat_values(guidedcirc) #used to be get_flat_values_root_branch
        trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
        tree_counter.append(1)

    else: 
        tree_counter[-1]+=1  

    # store values 
    acceptpathall.append(0) # store in order to have path updates and innernode match
    gthetas.append(gtheta_cur)
    inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
    tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
    tolog.update(inner)
    tolog.update({"gtheta": gtheta_cur})
    wandb.log(tolog) 


    #######################
    ##   propose kalpha  ##
    #######################
    # propose parameter, proposal is mirrored gaussian with sd
    key, subkey = jax.random.split(key, 2)
    kalphacirc = mirrored_gaussian(subkey, kalpha_cur, proposal_sd_kalpha, 0, 10) 
    thetacirc['k_alpha']= kalphacirc # propose rate 

    # do backwards filter using new parameter
    tree_bf_circ = backward_filter(bphylogeny, thetacirc, sigma)
    
    # get paths for new parameter same wiener process 
    fgecirc = jax.jit(lambda *x: forward_guide_edge(*x, b, sigma, thetacirc))
    guidedcirc = forward_guide(super_root, tree_bf_circ,_dtsdWsT, fgecirc)
    logpsicirc = get_logpsi(guidedcirc)
    logrhotildecirc = -tree_bf_circ.message['c']-0.5*super_root.T@tree_bf_circ.message['H'][0]@super_root+tree_bf_circ.message['F'][0].T@super_root
    
    # get acceptance probability
    log_r = logpsicirc - logpsicur + logrhotildecirc - logrhotildecur + scipy.stats.uniform.logpdf(kalphacirc, loc=kalpha_loc, scale=kalpha_scale) - scipy.stats.uniform.logpdf(kalpha_cur, loc=kalpha_loc, scale=kalpha_scale) #+ q_ka_kacirc - q_kacirc_ka
    A = min(1, np.exp(log_r))
    print(f'kalpha acceptance probability {A}')

    key, subkey = jax.random.split(key, 2)
    if jax.random.uniform(subkey)<A: 
        # update variables 
        kalpha_cur = kalphacirc
        theta_cur = thetacirc
        data_tree_bf = tree_bf_circ
        fge = fgecirc
        
        # update probabilities
        logrhotildecur = logrhotildecirc
        logpsicur = logpsicirc 

        # update statistics
        acceptkalpha[j+1] = 1

        # save new paths
        guided_tree = get_flat_values(guidedcirc) #used to be get_flat_values_root_branch
        trees = np.concatenate((trees, np.expand_dims(guided_tree, axis=0)), axis=0)
        tree_counter.append(1)
    else: 
        tree_counter[-1]+=1

    # store values 
    acceptpathall.append(0) # store in order to have path updates and innernode match
    kalphas.append(kalpha_cur)
    inner = dict([(str(i),guided_tree[2][i]) for i in range(2)])
    tolog = dict([('root-'+str(l),guided_tree[0][l]) for l in range(2)])
    tolog.update(inner)
    tolog.update({"kalpha": kalpha_cur})
    wandb.log(tolog) 

    if j%20==0 or j==N-1:
        np.savetxt(outputpath+"kalphas.csv", kalphas, delimiter=",")
        np.savetxt(outputpath+"acceptkalpha.csv", acceptkalpha, delimiter=",")
        np.savetxt(outputpath+"acceptgtheta.csv", acceptgtheta, delimiter=",")
        np.savetxt(outputpath+"acceptpath.csv", acceptpath, delimiter=",") # for plotting
        np.savetxt(outputpath+"tree_nodes.csv", trees.reshape(trees.shape[0],-1), delimiter=",") # use reshape(number of trees,59,40) to get back
        np.savetxt(outputpath+"tree_counter.csv", tree_counter, delimiter=",")
        np.savetxt(outputpath+"gthetas.csv", gthetas, delimiter=",")

        wandb.save(outputpath+"kalphas.csv")
        wandb.save(outputpath+"gthetas.csv")
        wandb.save(outputpath+'acceptkalpha.csv')
        wandb.save(outputpath+'acceptgtheta.csv')
        wandb.save(outputpath+'acceptpath.csv')
        wandb.save(outputpath+'tree_nodes.csv')
        wandb.save(outputpath+'tree_counter.csv')
    wandb.config.update({'acceptance rate path': np.mean(acceptpath[:j+1]), 'acceptance rate gtheta': np.mean(acceptgtheta[:j+1]), 'acceptance rate kalpha': np.mean(acceptkalpha[:j+1])}, allow_val_change=True)
wandb.finish()
 