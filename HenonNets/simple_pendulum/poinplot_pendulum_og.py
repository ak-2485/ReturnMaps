import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
sys.path.append('../')
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import pickle
from datetime import datetime

filename = 'sp_data_' + str(datetime.now()) +'.pickle' 

'''Define a Henon map'''
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
        xnext,ynext=HenonMap(z[:,0:1],z[:,1:],self.Win,self.Wout,self.bin,self.eta)
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

"""END OF HENON NET ARCH """
"""    START EXAMPLE     """
tf.keras.backend.set_floatx('float64')
d = {}

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

def leapfrog(z,DU,eps,n_lf_steps):
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

# H(q,p) = p^2/2 - cos(q)
def zdot(z,t,eps):
    Q = z[:,0:1]
    P = z[:,1:2]
    return np.hstack([-P,np.sin(Q)])

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
    lfstep=200
    latent_samples = gen_samples_circle(origin, r1,nics)
    sample = latent_samples
    n_samples=(n_iterations)*nics
    out = np.zeros([n_samples,2])
    for i in range(n_iterations):
        sample = rk4(sample,0.1,rkstep)
        d['integrator'] = 'rk4'
        #d['integrator'] = 'leapfrog'
        d['lfstep']     = 'lfstep'
        d['rkstep']     = 'rkstep'
        #sample = leapfrog(sample,np.sin,0.1,lfstep)
        out[(i)*nics:(i+1)*nics,:] = sample[:,:]
    return [out,latent_samples]

r1=0.3
r2=1.5
n_samp_r1 = 1000
n_samp_r2 = 1000
[labels_raw1, data_raw1] = gen_samples_pmap([0,0], r1, n_samp_r1, 1)
[labels_raw2, data_raw2] = gen_samples_pmap([0,0], r2, n_samp_r2, 1)
labels_raw = np.vstack((labels_raw1, labels_raw2))
data_raw   = np.vstack((data_raw1, data_raw2))
rr=labels_raw[:,0]**2+labels_raw[:,1]**2
ind=np.argwhere(rr <= r2**2)
n_data=len(ind)

data=np.zeros([n_data,2])
data[:,0:1]=data_raw[ind,0]
data[:,1:2]=data_raw[ind,1]

labels=np.zeros([n_data,2])
labels[:,0:1]=labels_raw[ind,0]
labels[:,1:2]=labels_raw[ind,1]

d['data']   = data
d['lables'] = labels
pickle.dump(d,open(filename,"wb"))

print("dumped pickle with data and labels")
print("finished generating samples")

def scheduler(epoch):
    if epoch < 20:
        return 2e-2
    elif epoch < 80:
        return 1e-2
    elif epoch < 200:
        return 4e-3
    elif epoch < 300:
        return 2e-3
    elif epoch < 400:
        return 1e-3
    elif epoch < 600:
        return 8e-4
    elif epoch < 1000:
        return 7e-4
    elif epoch < 1500:
        return 5e-4
    elif epoch < 2500:
        return 2e-4
    else:
        return 5e-5

ymean_tf = tf.constant(0., dtype = tf.float64)
ydiam_tf = tf.constant(np.pi, dtype = tf.float64)

l = []
for i in range(10):
    l.append(10)
unit_schedule = l

loss_fun = tf.keras.losses.MeanSquaredError()
test_model = HenonNet(unit_schedule)
optimizer = keras.optimizers.Adam()
test_model.compile(optimizer = optimizer, loss = loss_fun)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
h = test_model.fit(tf.convert_to_tensor(data[:n_data], dtype = tf.float64)
                   ,tf.convert_to_tensor(labels[:n_data], dtype = tf.float64), batch_size = 1000, epochs = 2000, verbose=0
                   ,callbacks=[callback])

print("finished training")

#Get training loss histories                                                                                                                                                            
training_loss = h.history['loss']
# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)
# Save loss to pickle for plotting                                                                                                                                                      
d['epoch_count']   = epoch_count
d['training_loss'] = training_loss

#predict using the model                                                                                                                                                                
nics = 20
n_steps = 2000
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

print('Using model and rk4 to generate poincare sections...')

start=time.time()
for i in range(n_steps+1):
    history_model[:,:, i] = current_state_model.numpy()[:,:]
    current_state_model = test_model(current_state_model)
end=time.time()

history_rk = np.zeros([nics,2,n_steps+1])
for i in range(n_steps+1):
    history_rk[:,:,i] = current_state_rk[:,:]
    current_state_rk = rk4(current_state_rk,0.1,500)
    #current_state_lf  = leapfrog(current_state_rk,np.sin,.1,20)
end2=time.time()

d['history_model'] = history_model
d['history_rk']    = history_rk
d['zic']           = zic
d['integ_time']    = end2-end
d['p_time']        = end-start

print('Poincare sections generated. Uses {} seconds'.format(end-start))
print('RK sections generated. Uses {} seconds'.format(end2-end))

pickle.dump(d,open(filename,"wb"))
print("dumped pickle to file " + filename)


fig1, ax1 = plt.subplots()
xplot1 = np.ravel(history_model[:,0,:])
yplot1 = np.ravel(history_model[:,1,:])
ax1.plot(xplot1,yplot1,'b.', markersize = .2)
ax1.plot(zic[:,0],zic[:,1],'r.')
ax1.set_title('Poincare plot by HenonNet')
plt.savefig('HNET_plot.png')


fig2, ax2 = plt.subplots()
xplot2 = np.ravel(history_rk[:,0,:])
yplot2 = np.ravel(history_rk[:,1,:])
ax2.plot(xplot2,yplot2,'b.', markersize = .2)
ax2.plot(zic[:,0],zic[:,1],'r.')
ax2.set_title('Poincare plot by Runge-Kutta')
plt.savefig('int_plot.png')


fig3, ax3 = plt.subplots()
line1,=ax3.plot(xplot1,yplot1,'b.', markersize = .5)
line2,=ax3.plot(xplot2,yplot2,'r.', markersize = .5)
ax3.set_title('HenonNet vs Reference')
plt.legend((line1,line2),('Learned','Reference'), loc='upper right',fontsize='medium')
plt.savefig('HNETvsRef.png')
