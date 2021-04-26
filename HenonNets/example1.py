import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from HenonNet import HenonNet

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
test_model = HenonNet()
test_model.setvals([5,5,5], ymean_tf, ydiam_tf)
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


plt.show()
