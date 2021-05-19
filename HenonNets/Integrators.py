import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time
import sys
sys.path.append('../')
import pickle

def rk_pmap(z,eps,n_rk_steps = 100):
    dphi = 2*np.pi/n_rk_steps
    phi_current = 0.0
    z_current = 1.0*z
    for i in range(n_rk_steps):
        k1 = zdot(z_current,phi_current,eps)
        k2 = zdot(z_current + .5*dphi*k1, phi_current + .5*dphi,eps)
        k3 = zdot(z_current + .5*dphi*k2, phi_current + .5*dphi,eps)
        k4 = zdot(z_current + dphi * k3, phi_current + dphi,eps)
        z_current = z_current + (1.0/6.0) * dphi * (k1 + 2*k2 + 2*k3 + k4)
        phi_current = phi_current + dphi
    return z_current

def leapfrog(z,eps,n_lf_steps):
    dphi = 2*np.pi/n_rk_steps
    phi_current = 0.0
    z0   = 1.0*z
    # midpoint for update
    z_arg1    = 1.0*z0
    z_arg2    = 1.0*z0
    z_current = 1.0*z0
    z_new     = 1.0*z1
    for i in range(n_lf_steps):
        z_up        = zdot(z_current,phi_current,eps)
        z_new       = z_current + dphi*zdot(z_current + .5*dphi*z_up, phi_current + .5*dphi,eps)
        y_mid     = (z_current[:,1] + z_new[:,1])/2.0
        z_arg1[:,0] = z_current[:,0]
        z_arg1[:,1] = y_mid[:,1]
        z_arg2[:,0] = z_new[:,0] 
        z_arg2[:,1] = y_mid[:,1]
        x_new      = z_current[:,0] + 0.5*dphi*(zdot(z_arg1)+zdot(z_arg2)) 
        y_mid      = z_current[:,1] - 0.5*dphi*zdot(z_arg1)
        z_arg2[:,0] = x_new[:,0] 
        z_arg2[:,1] = y_mid[:,1]
        y_new      = z_current[:,1] - 0.5*dphi*zdot(z_arg2)
        z_current[:,0]  = x_new
        z_current[:,1]  = y_new
    return z_current 

#H = p^2/2 - cos(q)                                                                                                                                                                     
def zdot(z,t,eps):
    X = z[:,0:1]
    Y = z[:,1:2]
    return np.hstack([-np.sin(Y),X])

x       = np.random.uniform(0.0,1.0,1000)
y       = np.random.uniform(-1.0,1.0,1000)
samples = np.hstack([x.reshape(1000,1),y.reshape(1000,1)])
rk4   = rk_pmap(samples ,0.25,2000)

plt.scatter(rk4[:,0],rk4[:,1])
plt.show()
