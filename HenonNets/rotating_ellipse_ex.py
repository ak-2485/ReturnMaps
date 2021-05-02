import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from HenonNet import HenonNet
import time
import random
from coilpy import *
import sys


ref = FOCUSHDF5('focus_ellipse_10k.h5')
# get the poincare plot points from FOCUS data
r = ref.ppr - ref.pp_raxis
z = ref.ppz - ref.pp_zaxis
# starting points are raw data
n_samples = len(r[0])
data = np.hstack([r[0].reshape(n_samples,1),z[0].reshape(n_samples,1)])
# labels are final integration points from FOCUS
lf = len(r)-1
labels = np.hstack([r[lf].reshape(n_samples,1),z[lf].reshape(n_samples,1)])


tf.keras.backend.set_floatx('float64')

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

n_data=n_samples

test_model = HenonNet()
test_model.setvals(unit_schedule, ymean_tf, ydiam_tf)
Adamoptimizer = keras.optimizers.Adam()
test_model.compile(optimizer = Adamoptimizer, loss = loss_fun)

h = test_model.fit(tf.convert_to_tensor(data[:n_data], dtype = tf.float64)
    ,tf.convert_to_tensor(labels[:n_data], dtype = tf.float64)
    , batch_size = 1000, epochs = 5000, verbose=0, callbacks = [callback])


"""
Verify
"""
nics = 20
n_steps = 1000

# get the poincare plot points from FOCUS data
r_init = r[0]
z_init = z[0]
ind_rng = range(0,len(r_init),len(r_init)//nics)
rp = [r_init[i] for i in ind_rng]
zp = [z_init[i] for i in ind_rng]
xic = np.array(rp).reshape([nics,1])
yic = np.array(zp).reshape([nics,1])
zic = np.hstack([xic,yic])
current_state_model = tf.convert_to_tensor(zic, dtype = tf.float64)
history_model = np.zeros([nics,2,n_steps+1])

for i in range(n_steps+1):
    history_model[:,:, i] = current_state_model.numpy()[:,:]
    current_state_model = test_model(current_state_model)
    # poincare plot

def poincare_plot(ppr, ppz, pp_ns_range, color=None, **kwargs):
    """Poincare plot
    Args:
    color (matplotlib color, or None): dot colors; default None (rainbow).
    kwargs : matplotlib scatter keyword arguments
    Returns:
    None
    """
    from matplotlib import cm

    pp_ns = len(pp_ns_range)
    
    # get figure and ax data
    if plt.get_fignums():
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig, ax = plt.subplots()
        
    # get colors
    if color == None:
        colors = cm.rainbow(np.linspace(1, 0, pp_ns))
    else:
        colors = [color] * pp_ns
    kwargs["s"] = kwargs.get("s", 0.1)  # dotsize
    # scatter plot
    for i in pp_ns_range:
        ax.scatter(ppr[:, i], ppz[:, i], color=colors[i], **kwargs)
        plt.axis("equal")
        plt.xlabel("R [m]", fontsize=20)
        plt.ylabel("Z [m]", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    return
    
print('Starting plots')
# plot both sections
fig1, ax1 = plt.subplots()
xplot1 = np.ravel(history_model[:,0,:])
yplot1 = np.ravel(history_model[:,1,:])
ax1.plot(xplot1,yplot1,'b.', markersize = .2)
ax1.plot(zic[:,0],zic[:,1],'r.')
ax1.set_title('Poincare plot by HenonNet')
plt.savefig('ppH_ellipse.png')

fig2, ax2 = plt.subplots()
ref.ppr = ref.ppr - ref.pp_raxis
ref.ppz = ref.ppz - ref.pp_zaxis
poincare_plot(ref.ppr,ref.ppz,ind_rng)
ax2.plot(zic[:,0],zic[:,1],'r.')
ax2.set_title('Poincare plot by FOCUS')
plt.savefig('ppF_ellipse.png')
