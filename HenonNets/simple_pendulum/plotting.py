import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn.metrics

infile1 = 'sp_data_2021-05-21 01:41:38.887895.pickle'              # RK 200K rand 10K epochs, single circle
infile2 = 'sp_data_2021-05-20 19:05:44.866877.pickle'              # RK 200K cheb 10K epochs
infile3 = 'sp_data_2021-05-21 07:34:45.758217.pickle'              # RK 200K rand 10K epochs 
infile4 = 'sp_data_2021-05-21 09:25:53.089670.pickle'              # RK 20K  rand 20K epochs
infile5 = 'sp_data_2021-05-21 07:35:49.582556.pickle'              # LF 200k rand 10k epochs
infile6 = 'sp_data_2021-05-20 03:10:31.818704.pickle'              # LF 20K  rand 20K epochs
mapfile = 'LF_int_data_2021-05-23 05:56:35.346899.pickle'  
check_plot = 'LF_int_data_2021-05-23 05:49:39.174256.pickle'
###
d1 = pickle.load(open(infile1,"rb"))
history_model1 = d1['history_model']
history_rk1    = d1['history_rk']
data1          = d1['data']
####
d2 = pickle.load(open(infile2,"rb"))
history_model2 = d2['history_model']
history_rk2    = d2['history_rk']
data2          = d2['data']
####
d3 = pickle.load(open(infile3,"rb"))
history_model3 = d3['history_model']
history_rk3    = d3['history_rk']
samples3       = d3['data']
####
d4 = pickle.load(open(infile4,"rb"))
samples4       = d4['data']
history_model4 = d4['history_model']
history_rk4    = d4['history_rk']
####
d5 = pickle.load(open(infile5,"rb"))
samples5       = d5['data']
history_model5 = d5['history_model']
history_lf1    = d5['history_rk']
####
d6 = pickle.load(open(infile6,"rb"))
history_model6 = d6['history_model']
history_lf2    = d6['history_rk']
samples6       = d6['data']
####
dmap =  pickle.load(open(mapfile,"rb"))
dcheck = pickle.load(open(check_plot,"rb"))
history_rk_check = dcheck['history_rk']
history_lf_check = dcheck['history_lf']

dzic = dcheck['zic']
zic = d1['zic']


"""
# plot sections
fig1, ax1 = plt.subplots()
xplot1 = np.ravel(history_model1[:,0,:])
yplot1 = np.ravel(history_model1[:,1,:])
ax1.plot(xplot1,yplot1,'b.', markersize = .2)
ax1.plot(zic[:,0],zic[:,1],'r.')
ax1.set_title('Poincare plot by RK HenonNet')
plt.savefig('HNET_full_res_RK_200k.png')


fig2, ax2 = plt.subplots()
xplot2 = np.ravel(history_lf_check[:,0,:]) 
yplot2 =  np.ravel(history_lf_check[:,1,:]) 
ax2.plot(xplot2,yplot2,'b.', markersize = .2)
ax2.plot(dzic[:,0],dzic[:,1],'r.')
ax2.set_title('Poincare plot by Symplectic Integrator')
plt.savefig('LF_ref.png')

xplot3 = np.ravel(history_lf1[:,0,:])
yplot3 = np.ravel(history_lf1[:,1,:])

fig3, ax3 = plt.subplots()
line1,=ax3.plot(xplot2,yplot2,'b.', markersize = .5)
line2,=ax3.plot(xplot3,yplot3,'r.', markersize = .5)
ax3.set_title('Runge-Kutta vs Symplectic')
plt.legend((line1,line2),('RK','LF'), loc='upper right',fontsize='medium')
plt.savefig('RKvsLF_full_res_200k.png')
"""

"""
training_loss1 = d1['training_loss']
epoch_count1   = d1['epoch_count']
training_loss2 = d2['training_loss']
epoch_count2   = d2['epoch_count']
training_loss3 = d3['training_loss']
epoch_count3   = d3['epoch_count']
training_loss4 = d4['training_loss']
epoch_count4   = d4['epoch_count']


# Visualize loss history
fig0, ax0 = plt.subplots()
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss'])
ax0.set_title('Loss history')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Loss_history_full_res_200k.png')


infile2 = 'full_res_20k/sp_data_2021-05-20 03:10:31.818704.pickle'
d2 = pickle.load(open(infile2,"rb"))
training_loss2 = d2['training_loss']
epoch_count2   = d2['epoch_count']

fig0b, ax0b = plt.subplots()

plt.plot(epoch_count1, training_loss1, 'r--', label='20e3 samples, ref. LF')
plt.plot(epoch_count2, training_loss2, 'b--', label='200e3 samples, ref. RK')
plt.plot(epoch_count3, training_loss3, 'g--', label='200e3 samples, Cheb RK')
plt.legend()
ax0b.set_title('Loss Histories')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('log Loss')
plt.savefig('Loss_history_compare.png')
"""

n_samps  = 20
#lf_map   = dmap['lf_map']
#rk_map   = dmap['rk_map']

check1 = history_model1[:,:,:]
base1  = history_rk1[:,:,:]
check2 = history_model2[:,:,:]
base2  = history_rk2[:,:,:]
check3 = history_model3[:,:,:]
base3  = history_rk3[:,:,:]
check4 = history_model4[:,:,:]
base4  = history_rk4[:,:,:]
check5 = history_model5[:,:,:]
base5  = history_lf1[:,:,:]
check6 = history_model6[:,:,:]
base6  = history_lf2[:,:,:]

ax1 = [r[1] for r in check1[:,0,:]]
ay1 = [r[1] for r in check1[:,1,:]]
a1  = np.column_stack((ax1,ay1))
bx1  = [r[1] for r in base1[:,0,:]]
by1  = [r[1] for r in base1[:,1,:]]
b1   = np.column_stack((bx1,by1))

ax2 = [r[1] for r in check2[:,0,:]]
ay2 = [r[1] for r in check2[:,1,:]]
a2  = np.column_stack((ax2,ay2))
bx2  = [r[1] for r in base2[:,0,:]]
by2  = [r[1] for r in base2[:,1,:]]
b2   = np.column_stack((bx2,by2))

ax3 = [r[1] for r in check3[:,0,:]]
ay3 = [r[1] for r in check3[:,1,:]]
a3  = np.column_stack((ax3,ay3))
bx3  = [r[1] for r in base3[:,0,:]]
by3  = [r[1] for r in base3[:,1,:]]
b3   = np.column_stack((bx3,by3))

ax4 = [r[1] for r in check4[:,0,:]]
ay4 = [r[1] for r in check4[:,1,:]]
a4  = np.column_stack((ax4,ay4))
bx4  = [r[1] for r in base4[:,0,:]]
by4  = [r[1] for r in base4[:,1,:]]
b4   = np.column_stack((bx4,by4))

ax5 = [r[1] for r in check5[:,0,:]]
ay5 = [r[1] for r in check5[:,1,:]]
a5  = np.column_stack((ax5,ay5))
bx5  = [r[1] for r in base5[:,0,:]]
by5  = [r[1] for r in base5[:,1,:]]
b5   = np.column_stack((bx5,by5))

ax6 = [r[1] for r in check6[:,0,:]]
ay6 = [r[1] for r in check6[:,1,:]]
a6  = np.column_stack((ax6,ay6))
bx6  = [r[1] for r in base6[:,0,:]]
by6  = [r[1] for r in base6[:,1,:]]
b6   = np.column_stack((bx6,by6))


#print(b)
#print(lf_map)
#print(a[:20])
#mse = [sklearn.metrics.mean_squared_error(b[n_samps*i:n_samps*(i+1)],rk_map) for i in range(2001)]
#mse1 = sklearn.metrics.mean_squared_error(lf_map,a1[:n_samps])
#mse2 = sklearn.metrics.mean_squared_error(lf_map,a2[:n_samps])
#mse3 = sklearn.metrics.mean_squared_error(lf_map,a3[:n_samps])
#mse4 = sklearn.metrics.mean_squared_error(lf_map,a4[:n_samps])
#mse5 = sklearn.metrics.mean_squared_error(lf_map,a5[:n_samps])
#mse6 = sklearn.metrics.mean_squared_error(lf_map,a6[:n_samps])
#mse7 = sklearn.metrics.mean_squared_error(lf_map,rk_map)

#print(mse1)
#print(mse2)
#print(mse3)
#print(mse4)
#print(mse5)
#print(mse6)
#print(mse7)
#r = np.array((1,1))
#d = np.array((2,2))
#s = np.array((3,3))
#y = np.array((4,4))
#rd = np.column_stack((r,d))
#sy = np.column_stack((s,y))

#mse = [np.sqrt(np.mean(np.linalg.norm(a[i*n_samps:n_samps*(i+1)]-b[i*n_samps:n_samps*(i+1)]))) for i in range(1001) ]

#print(rk_map)
#print(b[:n_samps])

"""
mse.reverse()
fig0b, ax0b = plt.subplots()
plt.plot(range(len(mse)),mse, 'r--', label='20e3 samples, ref. LF')
plt.legend()
ax0b.set_title('Loss Histories')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('log Loss')
plt.savefig('Loss_history_plot.png')
"""


#visualize training data
#fig, ax = plt.subplots()
#plt.plot(labels[:,0],labels[:,1],'.',markersize=1)
#ax.set_title('Poincare map output')
#plt.savefig('pmapout_og.png')
fig, ax = plt.subplots()
print(np.shape(data1))
plt.scatter(samples5[:,0],samples5[:,1],s=0.01)
ax.set_title('Poincare map input - Chebyshev grid')
plt.savefig('pmapin1_cheb_sp.png')
