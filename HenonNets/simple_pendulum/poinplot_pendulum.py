import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import sys
sys.path.append('../')
from HenonNet import HenonNet

tf.keras.backend.set_floatx('float64')

#H = p^2/2 - cos(q)
def zdot(z,t):
    X = z[:,0:1]
    Y = z[:,1:2]
    return np.hstack([-np.sin(Y),X])

def rk4(z,h,n_rk_steps = 100):
    dh = 2*np.pi/n_rk_steps
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

def gen_samples_pmap(origin,r1,nics,n_iterations):
    rkstep=2000
    latent_samples = gen_samples_circle(origin, r1,nics)
    sample = latent_samples
    n_samples=(n_iterations)*nics
    out = np.zeros([n_samples,2])
    for i in range(n_iterations):
        sample = rk4(sample,rkstep)
        out[(i)*nics:(i+1)*nics,:] = sample[:,:]
    return [out,latent_samples]

r1=0.3
r2=1.5
[labels_raw1, data_raw1] = gen_samples_pmap([0,0], r1, 110000, 1)
[labels_raw2, data_raw2] = gen_samples_pmap([0,0], r2, 100000, 1)
print(labels_raw1)
labels_raw = np.vstack((labels_raw1, labels_raw2))
print(labels_raw)
data_raw   = np.vstack((data_raw1, data_raw2))
rr=labels_raw[:,0]**2+labels_raw[:,1]**2
ind=np.argwhere(rr<=r2**2)
n_data=len(ind)
print(n_data)

data=np.zeros([n_data,2])
data[:,0:1]=data_raw[ind,0]
data[:,1:2]=data_raw[ind,1]

labels=np.zeros([n_data,2])
labels[:,0:1]=labels_raw[ind,0]
labels[:,1:2]=labels_raw[ind,1]

print("finished generating samples")

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

l = []
for i in range(10):
    l.append(10)
unit_schedule = l

loss_fun = tf.keras.losses.MeanSquaredError()
test_model = HenonNet()
test_model.setvals(unit_schedule, ymean_tf, ydiam_tf)
test_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
    loss = loss_fun)

callback = tf.keras.callbacks.LearningRateScheduler(schedulerHenon)
h = test_model.fit(data, labels, batch_size = 1000, epochs = 20000, verbose=0
    ,callbacks=[callback])

print("finished training")

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

plt.savefig('lossplot_simple_pendulum.png')




#predict using the model
nics = 20
n_steps = 2000

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

plt.savefig('poinplot_simple_pendulum.png')
