import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time
import sys
sys.path.append('../')
import pickle

def rk4(z,eps,n_rk_steps = 100):
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

def leapfrog2(z,DU,eps,n_lf_steps):
    z_current = 1.0*z
    q_new     = 1.0*z[:,0]
    dphi = 2*np.pi/n_lf_steps
    phi_current = 0.0
    for i in range(n_lf_steps):
        q_new = z_current[:,0] + dphi * z_current[:,1] - 0.5*dphi**2 * DU(z_current[:,0])
        z_current[:,1] = z_current[:,1] - 0.5*dphi*(DU(z_current[:,0]) + DU(q_new))
        z_current[:,0] = q_new
        phi_current = phi_current + dphi
    return z_current


"""
# H(q,p) = p^2/2 - cos(q)
def zdot(z,t,eps):
    Q = z[:,0:1]
    P = z[:,1:2]
    return np.hstack([-P,np.sin(Q)])

ydata=np.array([[0.5, 1.0, 1.5]])
xic = np.zeros([3,1])
yic = ydata.transpose()
zic = np.hstack([xic,yic])

rk_state_model=1.0*zic
rk_model = np.zeros([3,2,1000+1])
lf_state_model=1.0*zic
lf_model = np.zeros([3,2,1000+1])

for i in range(1000+1):
    lf_model[:,:, i] = lf_state_model
    rk_model[:,:, i] = rk_state_model
    rk_state_model = rk4(rk_state_model,.1,20)
    lf_state_model = leapfrog2(lf_state_model,np.sin,.1,20)

xplot1 = np.ravel(rk_model[:,0,:])
yplot1 = np.ravel(rk_model[:,1,:])
xplot2 = np.ravel(lf_model[:,0,:])
yplot2 = np.ravel(lf_model[:,1,:])

plt.scatter(xplot2,yplot2)
#plt.scatter(lf[:,0],lf[:,1])
plt.show()
"""
