
from .setup_SDEs import ito_integrator 

def simulate_tree(x, b, sigma, theta, dtsdWsT):
    (dts,dWs),dWs_children = dtsdWsT
    Xscirc = ito_integrator(x,b,sigma,dts,dWs,theta) # forward sample
    return dts, dWs, Xscirc, [simulate_tree(Xscirc[-1],b, sigma,theta, dtsdWs_child) 
                                  for dtsdWs_child in dWs_children]
