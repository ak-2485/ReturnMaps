import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from HenonNet import HenonNet

def get_eps():
    return 0.5

def zdot(z,phi,e):
    x = z[:,0:1]
    y = z[:,1:]
    xdot=e*(0.3*x*np.sin(2*phi) + 0.7*x*np.sin(3*phi))+y
    ydot=-e*(0.3*y*np.sin(2*phi) + 0.7*y*np.sin(3*phi)) - 0.25*np.sin(x)
    return np.hstack([xdot,ydot])

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

def gen_samples_circle(origin,radius,n_samples):
# (s,theta) -> (r(s)cos theta, r(s) sin theta) maps
# ds dtheta into (ds/dr) dr dtheta = (1/r)(ds/dr) dx dy
# to get uniform in x, y from uniform in s, theta
# we need ds/dr = r, or s(r) = .5 r^2, or r(s) = sqrt(2s)
    s_radius = .5*radius**2
    s = np.random.uniform(0.0,s_radius,n_samples)
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

radius=0.9
eps=get_eps()
[labels_raw, data_raw] = gen_samples_pmap([0,0], radius, 300000, 1,eps)
rr=labels_raw[:,0]**2+labels_raw[:,1]**2
ind=np.argwhere(rr<=radius**2)
n_data=len(ind)
data=np.zeros([n_data,2])
labels=np.zeros([n_data,2])
data[:,0:1]=data_raw[ind,0]
data[:,1:2]=data_raw[ind,1]
labels[:,0:1]=labels_raw[ind,0]
labels[:,1:2]=labels_raw[ind,1]
#visualize training data
fig, ax = plt.subplots()
plt.plot(labels[:,0],labels[:,1],'.',markersize=1)
ax.set_title('Poincare map output')
fig, ax = plt.subplots()
plt.plot(data[:,0],data[:,1],'r.',markersize=1)
ax.set_title('Poincare map input')
plt.show()
