import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
sys.path.append('../')
import pickle
from datetime import datetime

"""
For generating the baseline RK4 and LF maps
"""
 
filename = 'LF_int_data_' + str(datetime.now()) +'.pickle' 
d = {}

def rk4(z,eps,n_rk_steps = 100):
#    dphi = 2*np.pi/n_rk_steps
    dphi =0.1
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

def get_eps():
    return 0.5

def zdot(z,phi,e):
    x = z[:,0:1]
    y = z[:,1:]
    xdot=e*(0.3*x*np.sin(2*phi) + 0.7*x*np.sin(3*phi))+y
    ydot=-e*(0.3*y*np.sin(2*phi) + 0.7*y*np.sin(3*phi)) - 0.25*np.sin(x)
    return np.hstack([xdot,ydot])

def gen_samples_circle(origin,radius,n_samples):
# (s,theta) -> (r(s)cos theta, r(s) sin theta) maps                                                                                                                                     
# ds dtheta into (ds/dr) dr dtheta = (1/r)(ds/dr) dx dy                                                                                                                                 
# to get uniform in x, y from uniform in s, theta                                                                                                                                       
# we need ds/dr = r, or s(r) = .5 r^2, or r(s) = sqrt(2s)                                                                                                                              
    s_radius = .5*radius**2
    #s = np.random.uniform(0.0,s_radius,n_samples)                                                                                                                                     
    s = s_radius*np.cos(np.pi * np.arange(n_samples)/(n_samples -1))
    theta = np.random.uniform(0.0,2*np.pi,n_samples)
    x = origin[0]+np.sqrt(2*s)*np.cos(theta)
    y = origin[1]+np.sqrt(2*s)*np.sin(theta)
    return np.hstack([x.reshape(n_samples,1),y.reshape(n_samples,1)])

def gen_samples_pmap(origin,r1,nics,n_iterations,eps):
    rkstep=1500
    latent_samples = gen_samples_circle(origin, r1,nics)
    sample = latent_samples
    n_samples=(n_iterations)*nics
    out = np.zeros([n_samples,2])
    for i in range(n_iterations):
        sample = rk_pmap(sample,eps,rkstep)
        out[(i)*nics:(i+1)*nics,:] = sample[:,:]
    return [out,latent_samples]


# prediction
eps=get_eps()
n_samp  = 20
xic = np.linspace(0.05,0.7,n_samp).reshape([n_samp,1])
yic = 0.0*np.ones([n_samp, 1])
zic = np.hstack([xic,yic])  
current_state_rk = 1.0*zic
rk_map  = rk4(current_state_rk,0.1,lf_steps)

"""
current_state_rk = 1.0*zic
n_steps=4000
history_model = np.zeros([n_samp,2,n_steps+1])
history_rk = 1.0*history_model
for i in range(n_steps+1):
    history_rk[:,:,i] = current_state_rk[:,:]
    current_state_rk = rk4(current_state_rk,0.1,500)
"""

d['zic']        = zic
d['history_rk'] = history_rk
d['rk_map']     = rk_map
d['n_samp']     = n_samp
pickle.dump(d,open(filename,"wb"))
print("dumped pickle to file " + filename)

