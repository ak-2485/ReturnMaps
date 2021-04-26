"""
Code from
https://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-20-24873
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense


"""
Define a Henon map
"""
@tf.function
def HenonMap(X,Y,Win,Wout,bin,eta,ymean,ydiam):
    with tf.GradientTape() as tape:
        tape.watch(Y)
        Ylast = (Y - ymean) / ydiam
        hidden = tf.math.tanh(tf.linalg.matmul(Ylast, Win) + bin)
        V = tf.linalg.matmul(hidden,Wout)
    Xout= Y+eta
    Yout=-X+tape.gradient(V,Y)
    return Xout, Yout

"""
Define a Henon layer
"""
class HenonLayer(layers.Layer):
    def __init__(self,ni,ymean,ydiam):
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
    def call(self,z,ymean,ydiam):
        xnext,ynext=HenonMap(z[:,0:1],z[:,1:],self.Win,self.Wout,self.bin, self.eta,ymean,ydiam)
        xnext,ynext=HenonMap(xnext,ynext,self.Win,self.Wout,self.bin,
        self.eta,ymean,ydiam)
        xnext,ynext=HenonMap(xnext,ynext,self.Win,self.Wout,self.bin,
        self.eta,ymean,ydiam)
        xnext,ynext=HenonMap(xnext,ynext,self.Win,self.Wout,self.bin,
        self.eta,ymean,ydiam)
        return tf.concat([xnext,ynext], axis =1)

"""
Define a HenonNet
"""
class HenonNet(Model):
    def __init__(self):#
        pass

    def setvals(self, unit_list, ymean_tf, ydiam_tf):
        self.ymean_tf = ymean_tf
        self.ydiam_tf = ydiam_tf
        super(HenonNet, self).__init__()
        self.N = len(unit_list)
        self.hlayers=[]

        for i in range(self.N):
            ni = unit_list[i]
            hl = HenonLayer(ni,self.ymean_tf,self.ydiam_tf)
            self.hlayers.append(hl)

    def call(self,r):
        rout = r
        for i in range(self.N):
            rout = self.hlayers[i](rout,self.ymean_tf,self.ydiam_tf)
        return rout
