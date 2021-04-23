"""
Code from
https://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-20-24873
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

"""
Define a Henon map
"""
@tf.function
def HenonMap(X,Y,Win,Wout,bin,eta):
    with tf.GradientTape() as tape:
        tape.watch(Y)
        Ylast = (Y - ymean_tf) / ydiam_tf
        hidden = tf.math.tanh(tf.linalg.matmul(Ylast, Win) + bin)
        V = tf.linalg.matmul(hidden,Wout)
    Xout= Y+eta
    Yout=-X+tape.gradient(V,Y)
    return Xout, Yout

'''Define a Henon layer'''
class HenonLayer(layers.Layer):
    def __init__(self,ni):
        super(HenonLayer, self).__init__()
        init = tf.initializers.GlorotNormal()
        init_zero = tf.zeros_initializer()
        W_in_init = init(shape=[1,ni], dtype = tf.float64)
        W_out_init = init(shape=[ni,1], dtype = tf.float64)
        b_in_init = init(shape=[1,ni], dtype = tf.float64)
        eta_init = init(shape=[1,1], dtype = tf.float64)

        self.Win = tf.Variable(W_in_init, dtype = tf.float64)
        self.Wout = tf.Variable(W_out_init, dtype = tf.float64)
        self.bin = tf.Variable(b_in_init, dtype = tf.float64)
        self.eta = tf.Variable(eta_init, dtype = tf.float64)

    @tf.function
    def call(self,z):
        xnext,ynext=HenonMap(z[:,0:1],z[:,1:],self.Win,self.Wout,self.bin,
            self.eta)
        xnext,ynext=HenonMap(xnext,ynext,self.Win,self.Wout,self.bin,self.eta)
        xnext,ynext=HenonMap(xnext,ynext,self.Win,self.Wout,self.bin,self.eta)
        xnext,ynext=HenonMap(xnext,ynext,self.Win,self.Wout,self.bin,self.eta)
        return tf.concat([xnext,ynext], axis =1)

'''Define a HenonNet'''
class HenonNet(Model):
    def __init__(self,unit_list):#
        super(HenonNet, self).__init__()
        self.N = len(unit_list)
        self.hlayers=[]

        for i in range(self.N):
            ni = unit_list[i]
            hl = HenonLayer(ni)
            self.hlayers.append(hl)

    def call(self, r):
        rout = r
        for i in range(self.N):
            rout = self.hlayers[i](rout)
        return rout

tf.keras.backend.set_floatx('float64')

#H = p^2/2 - cos(q)
def zdot(z,t):
    X = z[:,0:1]
    Y = z[:,1:2]
    return np.hstack([-np.sin(Y),X])

def rk4(z,h,n_rk_steps = 100):
    dh = h/n_rk_steps
    z_current = 1.0*z
    t=0
    for i in range(n_rk_steps):
        k1 = zdot(z_current,t)
        k2 = zdot(z_current + .5*dh*k1, t + .5*dh)
        k3 = zdot(z_current + .5*dh*k2, t + .5*dh)
        k4 = zdot(z_current + dh * k3, t + dh)
        z_current = z_current + (1.0/6.0) * dh * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dh
    return z_current

n_data=10000
xdata=np.random.uniform(-np.sqrt(2), np.sqrt(2), n_data)
ydata=np.random.uniform(-np.pi/2, np.pi/2, n_data)

data = np.hstack([xdata.reshape(n_data,1),ydata.reshape(n_data,1)])
labels = rk4(data,h=0.1,n_rk_steps=10)

def schedulerHenon(epoch):
    if epoch < 100:
        return 1e-1
    elif epoch < 150:
        return 6e-2
    elif epoch < 200:
        return 2e-2
    elif epoch < 300:
        return 5e-3
    elif epoch < 1000:
        return 1e-3
    elif epoch < 3000:
        return 4e-4
    else:
        return 1e-4

ymean_tf = tf.constant(0., dtype = tf.float64)
ydiam_tf = tf.constant(np.pi, dtype = tf.float64)

loss_fun = tf.keras.losses.MeanSquaredError()
test_model = HenonNet([5,5,5])
test_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
    loss = loss_fun)

callback = tf.keras.callbacks.LearningRateScheduler(schedulerHenon)
h = test_model.fit(data, labels, batch_size = 1000, epochs = 5000, verbose=0
    ,callbacks=[callback])

# Get training loss histories
training_loss = h.history['loss']
# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)
# Visualize loss history
fig0, ax0 = plt.subplots()
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss'])
ax0.set_title('Loss histroy')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')

#predict using the model
nics = 3
n_steps = 1000
ydata=np.array([[0.5, 1.0, 1.5]])
xic = np.zeros([3,1])
yic = ydata.transpose()
zic = np.hstack([xic,yic])
current_state_model = tf.convert_to_tensor(zic, dtype = tf.float64)
rk_state_model=zic
history_model = np.zeros([nics,2,n_steps+1])
rk_model = np.zeros([nics,2,n_steps+1])
print('using model to generate figures')

for i in range(n_steps+1):
    history_model[:,:, i] = current_state_model.numpy()[:,:]
    rk_model[:,:, i] = rk_state_model
    current_state_model = test_model(current_state_model)
    rk_state_model = rk4(rk_state_model,.1,20)

# plot both sections
fig1, ax1 = plt.subplots()

for i in range(3):
    line2,=ax1.plot(rk_model[i,0,:],rk_model[i,1,:],'b')

xplot1 = np.ravel(history_model[:,0,:])
yplot1 = np.ravel(history_model[:,1,:])
line1,=ax1.plot(xplot1,yplot1,'r.', markersize = 2)
ax1.legend((line1, line2), ('HenonNet', 'Reference'))
ax1.set_title('Learned flow for pendulum')
