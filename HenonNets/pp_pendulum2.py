import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from HenonNet import HenonNet
import time

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
data[:,0:1]=data_raw[ind,0]
data[:,1:2]=data_raw[ind,1]

labels=np.zeros([n_data,2])
labels[:,0:1]=labels_raw[ind,0]
labels[:,1:2]=labels_raw[ind,1]

def scheduler(epoch):
    if epoch < 20:
        return 5e-2
    elif epoch < 80:
        return 2e-2
    elif epoch < 200:
        return 6e-3
    elif epoch < 300:
        return 4e-3
    elif epoch < 400:
        return 2e-3
    elif epoch < 600:
        return 1e-3
    elif epoch < 1000:
        return 8e-4
    elif epoch < 1500:
        return 7e-4
    elif epoch < 2500:
        return 5e-4
    elif epoch < 3500:
        return 2e-4
    elif epoch < 4500:
        return 5e-5
    else:
        return 1e-5

ymean_tf = tf.constant(0., dtype = tf.float64)
ydiam_tf = tf.constant(2., dtype = tf.float64)
callback = keras.callbacks.LearningRateScheduler(scheduler)
loss_fun = keras.losses.MeanSquaredError()
l = []
for i in range(10):
    l.append(10)
unit_schedule = l

n_data=220000

test_model = HenonNet()
test_model.setvals(unit_schedule, ymean_tf, ydiam_tf)
Adamoptimizer = keras.optimizers.Adam()
test_model.compile(optimizer = Adamoptimizer, loss = loss_fun)

h = test_model.fit(tf.convert_to_tensor(data[:n_data], dtype = tf.float64)
    ,tf.convert_to_tensor(labels[:n_data], dtype = tf.float64)
    , batch_size = 1000, epochs = 5000, verbose=0, callbacks = [callback])


nics = 20
n_steps = 1000
xic = np.linspace(0.05,.7,nics).reshape([nics,1])
yic = 0.0*np.ones([nics, 1])
yic2 = np.linspace(0.3,.6,nics//2).reshape([nics//2,1])
xic2 = 0.0*np.ones([nics//2, 1])
zic = np.hstack([np.vstack([xic,xic2]),np.vstack([yic,yic2])])
nics=nics+nics//2
current_state_model = tf.convert_to_tensor(zic, dtype = tf.float64)
current_state_rk = 1.0*zic
history_model = np.zeros([nics,2,n_steps+1])
history_rk = 1.0*history_model
start=time.time()
print('Using model and rk4 to generate poincare sections...')

for i in range(n_steps+1):
    history_model[:,:, i] = current_state_model.numpy()[:,:]
    current_state_model = test_model2(current_state_model)

end=time.time()
history_rk = np.zeros([nics,2,n_steps+1])

for i in range(n_steps+1):
    history_rk[:,:,i] = current_state_rk[:,:]
    current_state_rk = rk_pmap(current_state_rk,eps,500)

end2=time.time()
print('Poincare sections generated. Uses {} seconds'.format(end-start))
print('RK sections generated. Uses {} seconds'.format(end2-end))
# plot both sections
fig1, ax1 = plt.subplots()
xplot1 = np.ravel(history_model[:,0,:])
yplot1 = np.ravel(history_model[:,1,:])
ax1.plot(xplot1,yplot1,'b.', markersize = .2)
ax1.plot(zic[:,0],zic[:,1],'r.')
ax1.set_title('Poincare plot by HenonNet')

plt.savefig('pp_pendulum.png')
