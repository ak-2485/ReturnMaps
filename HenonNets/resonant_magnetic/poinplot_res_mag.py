import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import sys
sys.path.append('../')
from HenonNet import HenonNet
import time

tf.keras.backend.set_floatx('float64')


def get_eps():
    return 0.25

def zdot(z,phi,e):
    x      = z[:,0:1]
    y      = z[:,1:2]
    arg_a  = (x-y) * x**2 * np.cos(3*phi)
    arg_b  = (x-y) * x**2 * np.sin(3*phi)
    xdot_a = (1 - (np.tanh(arg_a))**2) * (3*x**2 - 2*x*y)*np.cos(3*phi) 
    xdot_b = (1 - (np.tanh(arg_b))**2) * (3*x**2 - 2*x*y)*np.sin(3*phi)
    ydot_a = (1 - (np.tanh(arg_a))**2) * -(x**2 * np.cos(3*phi))
    ydot_b = (1 - (np.tanh(arg_b))**2) * -(x**2 * np.sin(3*phi))
    xdot   = 0.5*x + 0.25*e*(xdot_a + 0.2*xdot_b) 
    ydot   = y + 0.25*e*(ydot_a + 0.2*ydot_b)
    return np.hstack([ydot,-xdot])

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

def gen_samples_ellipse(origin,r_major,r_minor,n_samples):
    """
    Generate random points inside a circle of radius 1, ensuring
    uniformity in ellipse by taking sqrt of resulting radius.
    """
    r = np.random.uniform(0.0,1.0,n_samples)
    theta = np.random.uniform(0.0,2*np.pi,n_samples)
    x = origin[0]+np.sqrt(r)*np.cos(theta)
    y = origin[1]+np.sqrt(r)*np.sin(theta)
    x = x * r_major
    y = y * r_minor
    return np.hstack([x.reshape(n_samples,1),y.reshape(n_samples,1)])

def gen_samples_pmap(origin,r1,r2,nics,n_iterations,eps):
    rkstep   = 1500
    interval = 10
    latent_samples = gen_samples_ellipse(origin, r1,r2,nics)
    n_samples=(n_iterations//interval)*nics
    samples  = np.zeros([n_samples,2])
    samples[0:nics,:] = latent_samples[:,:]
    sample = latent_samples[:,:]
    for i in range(n_iterations):
        sample = rk_pmap(sample,eps,rkstep)
        if (i % interval == 0 and i != 0):
            ind = i//interval
            samples[ind*nics:(ind+1)*nics,:] = sample[:,:]
    return samples

def gen_labels(samples, nics, n_iterations, eps):
    rkstep=1500
    n_samples=nics
    out = np.zeros([n_samples,2])
    for i in range(n_iterations):
        sample = rk_pmap(samples,eps,rkstep)
        out[(i)*nics:(i+1)*nics,:] = sample[:,:]
    return out


r1=1.75
r2=1.0
eps=get_eps()
data_raw = gen_samples_pmap([0,0], r1,r2, 10000, 400,eps)
sys.exit()
#labels_raw = gen_labels(data_raw, 400000, 1,eps)
print("done generating labels and data")
print(np.shape(data_raw))

#data=np.zeros([n_data,2])
#data[:,0:1]=data_raw[:,0]
#data[:,1:2]=data_raw[:,1]
data = data_raw
#labels=np.zeros([n_data,2])
#labels = labels_raw
#labels[:,0:1]=labels_raw[:,0]
#labels[:,1:2]=labels_raw[:,1]

print("starting viz data")
#visualize training data
#fig, ax = plt.subplots()
#plt.plot(labels[:,0],labels[:,1],'.',markersize=1)
#ax.set_title('Poincare map output')
#plt.savefig('res_mag_out_data.png')

fig, ax = plt.subplots()
plt.plot(data[:,0],data[:,1],'r.',markersize=1)
ax.set_title('Poincare map input')
plt.savefig('res_mag_in_data-low_res.png')
print("finished plotting")
sys.exit()
rate_init = 0.1
def scheduler(epoch):
    if epoch < 20:
        return rate_init
    elif epoch < 80:
        return rate_init*0.5**2
    elif epoch < 200:
        return rate_init*0.5**3
    elif epoch < 300:
        return rate_init*0.5**4
    elif epoch < 400:
        return rate_init*0.5**5
    elif epoch < 600:
        return rate_init*0.5**6
    elif epoch < 1000:
        return rate_init*0.5**7
    elif epoch < 1500:
        return rate_init*0.5**8
    elif epoch < 2500:
        return rate_init*0.5**9
    elif epoch < 3500:
        return rate_init*0.5**10
    elif epoch < 4500:
        return rate_init*0.5**11
    elif epoch < 5500:
        return rate_init*0.5**12
    else:
        return rate_init*0.5**13
    
ymean_tf = tf.constant(0., dtype = tf.float64)
ydiam_tf = tf.constant(2., dtype = tf.float64)
callback = keras.callbacks.LearningRateScheduler(scheduler)
loss_fun = keras.losses.MeanSquaredError()
l = []
for i in range(50):
    l.append(5)
unit_schedule = l

print("starting optimization")
test_model = HenonNet()
test_model.setvals(unit_schedule, ymean_tf, ydiam_tf)
Adamoptimizer = keras.optimizers.Adam()
test_model.compile(optimizer = Adamoptimizer, loss = loss_fun)

h = test_model.fit(tf.convert_to_tensor(data, dtype = tf.float64)
    ,tf.convert_to_tensor(labels, dtype = tf.float64)
    , batch_size = 400, epochs = 7000, verbose=0, callbacks = [callback])


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
    current_state_model = test_model(current_state_model)

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
plt.savefig('ppH_res_mag.png')

fig2, ax2 = plt.subpots()
xplot2 = np.ravel(history_rk[:,0,:])
yplot2 = np.ravel(history_rk[:,1,:])
ax2.plot(xplot2,yplot2,'b.', markersize = .2)
ax2.plot(zic[:,0],zic[:,1],'r.')
ax2.set_title('Poincare plot by Runge-Kutta')
plt.savefig('ppR_res_mag.png')
