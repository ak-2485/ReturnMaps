import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from HenonNet import HenonNet
import time
import random

import sys
sys.path.append('~/Documents/research/FOCUS/python')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus

from coilpy import *
ref = FOCUSHDF5('../FOCUSdata/focus_ellipse.h5')

# MPI_INIT
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# run FOCUS; pp_ns training points requires a large number of CPUs
if rank==0:
    print("##### Begin FOCUS run with {:d} CPUs. #####".format(size))
    master = True

test = FOCUSpy(comm=comm, extension='ellipse', verbose=True)
focus.globals.cg_maxiter = 1 # bunk
focus.globals.pp_ns      = 100000 # number of fieldlines
focus.globals.pp_maxiter = 10 # number of periods to integrate
test.run(verbose=True)

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

def gen_samples(x,y,n_samples):
    inds = np.random.choice(range(len(x)),n_samples,replace=False)
    xs = np.array([x[i] for i in inds])
    ys = np.array([y[i] for i in inds])
    latent_samples = np.hstack([xs.reshape(n_samples,1),ys.reshape(n_samples,1)])
    sample = latent_samples
    out = sample[:,:]
    return [out,latent_samples]

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

test = FOCUSpy(comm=comm, extension='ellipse', verbose=True)
focus.globals.cg_maxiter = 1 # bunk
focus.globals.pp_ns      = nics # number of fieldlines
focus.globals.pp_maxiter = n_steps # number of periods to integrate
test.run(verbose=True)

# get the poincare plot points from FOCUS data
r = ref.ppr - ref.pp_raxis
z = ref.ppz - ref.pp_zaxis
xic = r[0]
xic = np.array(xic).reshape([nics,1])
yic = z[0]
yic = np.array(yic).reshape([nics,1])
zic = np.hstack([xic,yic])
current_state_model = tf.convert_to_tensor(zic, dtype = tf.float64)
history_model = np.zeros([nics,2,n_steps+1])
start=time.time()

for i in range(n_steps+1):
    history_model[:,:, i] = current_state_model.numpy()[:,:]
    current_state_model = test_model(current_state_model)

end=time.time()

print('Poincare sections generated. Uses {} seconds'.format(end-start))

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
ref.poincare_plot()
ax2.plot(zic[:,0],zic[:,1],'r.')
ax2.set_title('Poincare plot by FOCUS')

plt.savefig('ppF_ellipse.png')
