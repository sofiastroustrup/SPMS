import jax
import jax.numpy as jnp
from jax import debug

from .setup_SDEs import *
from .noise_kernel import *

    
def backward_filter(node, theta, sigma):
    #node = dict(node) # make copy to mimic functional approach (no modifiable state)
    node = node.copy()
    _ts = jnp.cumsum(jnp.concatenate((jnp.array([0.]), dts(T=node.T, n_steps = node.n_steps))))[:-1]
    # recursive call downwards in tree
    children = [backward_filter(child,theta, sigma) for child in node.children]
    
    # all values below are at the node
    T = node.T
    n = theta['n']
    d = theta['d']

    def a(t,x,theta):
        sigmax = sigma(t,x,theta)
        return jnp.einsum('ij,kj->ik',sigmax,sigmax)

    if len(children) == 0: # leaf node
        v = node.v
        tildea = a(T,v,theta) 
        subtree_var = node.obs_var
        invC = H_T = np.eye(n*d)*(1/subtree_var) # efficient way to get inverse 
        F_T = invC@v
        #C =  np.eye(n*d)*(subtree_var)
        #c_T = -jax.scipy.stats.multivariate_normal.logpdf(v,jnp.zeros_like(v),C) 

    else: # use same approach as previously to get v for the inner node 
        F_T = jnp.sum(jnp.array([child.message['F'][0] for child in children]),0)
        H_T = jnp.sum(jnp.array([child.message['H'][0] for child in children]),0)
        #c_T = jnp.sum(jnp.array([child.message['c'](0.,theta) for child in children]),0) 
        #Mdagger = C =  jnp.linalg.inv(H_T) # use solve instead?
        Mdagger = jnp.linalg.inv(H_T) # should be more numerically stabel with solve
        v = jnp.dot(Mdagger,F_T) # should be more numerically stable with solve
        subtree_var = 1./jnp.sum(jnp.array([1./(child.T+child.message['subtree_var']) for child in children]),0)
        tildea = Mdagger/subtree_var #a(T, v, theta) #Mdagger/subtree_var # could be done better i.e. using solve ror a(T, v, theta)

        
    # update node and return
    message = {
        'theta': theta,
        'subtree_var': subtree_var
    }; 
    
    # We assume B=0 and beta= 0
#     message['tildebeta'] =  lambda t,theta: 0
#     message['tildeB'] = lambda t, theta: 0
    
    # t-dependent values 
    message['tildea'] = tildea
    Phi_inv = lambda t:  np.eye(n*d)+H_T@message['tildea']*(T-t)
    message['H'] = jax.vmap(lambda ti: jnp.linalg.solve(Phi_inv(ti), H_T))(_ts)
    message['F'] = jax.vmap(lambda ti: jnp.linalg.solve(Phi_inv(ti), F_T))(_ts)

    # calculations for c, we use that M=H and Mdagger = H^(-1)
    # use results from page 35 in Continous-discreete smoothing of diffusions
    message['H_logdet'] = jnp.linalg.slogdet(message['H'][0])
    if message['H_logdet'][0]<=0:
        print("Warning: non positive determinant of H in BFFG")
    message['c2'] = (v.shape[0]/2)*jnp.log((2*jnp.pi))-0.5*message['H_logdet'][1] # det(A^-1)=1/det(A)
    message['mu'] = -jnp.linalg.solve(message['H'][0], message['F'][0])+v
    message['c1'] = (v-message['mu']).T@message['H'][0]@(v-message['mu']) #M=H
    message['c'] = 0.5*message['c1'] + message['c2']

    node.message = message
    node.children = children
    return node

# forward guide along entire tree
def forward_guide(v,node,dtsdWsT,forward_guide_edge):
    children = node.children
    (dts,dWs),dWs_children = dtsdWsT
    Xscirc,logPsi = forward_guide_edge(v,node.message,dts,dWs) # sample edge
    return dts,dWs,Xscirc,logPsi,[forward_guide(Xscirc[-1],child,dtsdWs_child, forward_guide_edge) 
                                  for child,dtsdWs_child in zip(children,dWs_children)]

# forward sampling along edge, assumes already backward filtered
def forward_guide_edge(x, message, dts, dWs, b, sigma, theta):
    tildea = message['tildea']
    tildebeta = lambda t,theta: 0 # message['tildebeta']
    #tilder = message['tilder']
    H = message['H']
    F = message['F']
    
    tildeb = lambda t,x,theta: tildebeta(t,theta) #+jnp.dot(tildeB,x) #tildeB is zero for now

    def a(t,x,theta):
        sigmax = sigma(t,x,theta)
        return jnp.einsum('ij,kj->ik',sigmax,sigmax)

    def bridge_MS(res, el):
        t, X = res
        dt, dW, H, F = el
        Xcur = X + b(t,X, theta)*dt + jnp.dot(a(t,X,theta), F-jnp.dot(H,X))*dt + jnp.dot(sigma(t,X, theta),dW)
        tcur = t + dt
        return((tcur,Xcur), (t, X))
    
    def logpsi(res, el):
        """
        - `res`: The result from the previous loop.
        - `el`: The current array element.
        """
        G, t = res
        dt, X, H, F = el 
        tilderx = F - jnp.dot(H, X)
        b_diff = b(t, X, theta) - tildeb(t, X, theta)
        a_diff = a(t, X, theta) - tildea
        H_minus_outer = H - jnp.outer(tilderx, tilderx)
        einsum_val = jnp.einsum('ij,ji->', a_diff, H_minus_outer)
        dot_val = jnp.dot(b_diff, tilderx)
        resG = G + (dot_val - 0.5 * einsum_val) * dt
        rest = t + dt

        return ((resG, rest), (dt, X))
    

    # sample
    (T, X), (ts, Xs)=jax.lax.scan(bridge_MS,(0., x),(dts, dWs, H, F))
    Xscirc = jnp.vstack((Xs, X))
    final , _ = jax.lax.scan(logpsi, (0, 0), (dts, Xscirc[:-1,:], H, F))
    return Xscirc,final[0]


#@jax.jit
def get_logpsi(tree):
    LogPhi = 0
    _,_,_,logphi, children = tree
    LogPhi+=jnp.array(logphi, float)
    for child in children:
        LogPhi += get_logpsi(child)
    return(LogPhi)







#############################################################

