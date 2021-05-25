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
import pickle
from datetime import datetime

filename = 'p_pendulum_data' + str(datetime.now()) +'.pickle'
d        = {}

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


def get_eps():
    return 0.25

def zdot(z,phi,e):
    x      = z[:,0:1]
    y      = z[:,1:2]
    t_a1   = (3*x**2 - 2*x*y)*np.cos(3*phi)
    t_b1   = (3*x**2 - 2*x*y)*np.sin(3*phi)
    dHdx_1 = (1 - (np.tanh((x-y) * x**2 * np.cos(3*phi)))**2) * t_a1
    dHdx_2 = 0.2*( (1 - (np.tanh((x-y) * x**2 * np.sin(3*phi)))**2) * t_b1)
    dHdx   = 0.5*x + 0.25*e*(dHdx_1 + dHdx_2)
    t_a2   = -1.0 * (x**2 * np.cos(3*phi))
    t_b2   = -1.0 * (x**2 * np.sin(3*phi))
    dHdy_1 = (1 - (np.tanh((x-y) * x**2 * np.cos(3*phi)))**2) * t_a2
    dHdy_2 = 0.2*((1 - (np.tanh((x-y) * x**2 * np.sin(3*phi)))**2) * t_b2)
    dHdy   = y + 0.25*e*(dHdy_1 + dHdy_2)
    xdot   = -1.0*dHdy
    ydot   = 1.0*dHdx
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
    interval = 10
    latent_samples = gen_samples_ellipse(origin, r1,r2,nics)
    samples = latent_samples[:,:]
    for i in range(n_iterations):
        print("generating samples step " + str(i))
        step = rk_pmap(samples,eps,500)
        #save every "interval" pmaps
        if (i % interval == 0):
            samples = np.vstack([samples,step])
    return samples

def gen_labels(samples, nics, n_iterations, eps):
    rkstep=1500
    n_samples=nics
    out = np.zeros([n_samples,2])
    for i in range(n_iterations):
        sample = rk_pmap(samples,eps,rkstep)
        out[(i)*nics:(i+1)*nics,:] = sample[:,:]
    return out

def resolve_samples_pmap(in_samples,n_iterations,eps):
    interval = 10
    samples = in_samples[:,:]
    for i in range(n_iterations):
        print("resolving step " + str(i))
        step = rk_pmap(samples,eps,500)
        #save every "interval" pmaps
        if (i % interval == 0):
            samples = np.vstack([samples,step])
    return samples


r1=1.75
r2=1.0
eps=get_eps()
#data = gen_samples_pmap([0,0], r1,r2, 10000, 60,eps)
infile1 = 'p_pendulum_data2021-05-24 21:45:24.734526.pickle'
###                                                                                                                                                                                     
d1 = pickle.load(open(infile1,"rb"))
in_data          = d1['data']
####     
data   = resolve_samples_pmap(in_data,400,eps)
labels = gen_labels(data,len(data),1,eps)
d['data'] = data
pickle.dump(d,open(filename,"wb"))
print(np.shape(data))
print("finished labels and data")

sys.exit()
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
n_data=len(data)

test_model2 = HenonNet(unit_schedule)
Adamoptimizer = keras.optimizers.Adam()
test_model2.compile(optimizer = Adamoptimizer, loss = loss_fun)

h = test_model2.fit(tf.convert_to_tensor(data[:n_data], dtype = tf.float64),
                    tf.convert_to_tensor(labels[:n_data], dtype = tf.float64),
                    batch_size = 1000, epochs = 5000, verbose=0, callbacks =[callback])

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

