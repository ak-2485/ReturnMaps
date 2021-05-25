import matplotlib.pyplot as plt
import pickle
import numpy as np
import sklearn.metrics

infile1 = 'p_pendulum_data2021-05-20 03:19:22.288070.pickle'       # cheb 1
infile2 = 'p_pendulum_data2021-05-20 03:22:41.456724.pickle'       # circ 1
infile3 = 'p_pendulum_data2021-05-20 11:19:40.593915.pickle'      # cheb 2 
infile4 = 'p_pendulum_data2021-05-20 15:48:37.958089.pickle'       # circ 2
infile5 = 'p_pendulum_data2021-05-21 10:59:31.712890.pickle'       # scipy
infile6 = 'p_pendulum_data2021-05-20 23:43:58.912362.pickle'       # cheb 200K samples

###
d1 = pickle.load(open(infile1,"rb"))
history_model1 = d1['history_model']
history_rk1    = d1['history_rk']
samples1       = d1['data']
####
d2 = pickle.load(open(infile2,"rb"))
history_model2 = d2['history_model']
history_rk2    = d2['history_rk']
samples2       = d2['data']
print(d2['train_time'])
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
print(d5['train_time'])
####
d6 = pickle.load(open(infile6,"rb"))
history_model6 = d6['history_model']
history_lf2    = d6['history_rk']
samples6       = d6['data']
####
#zic = d1['zic']


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


training_loss1 = d1['training_loss']
epoch_count1   = d1['epoch_count']
training_loss2 = d2['training_loss']
epoch_count2   = d2['epoch_count']
training_loss3 = d3['training_loss']
epoch_count3   = d3['epoch_count']
training_loss4 = d4['training_loss']
epoch_count4   = d4['epoch_count']
training_loss5 = d5['training_loss']
epoch_count5   = d5['epoch_count']
training_loss6 = d6['training_loss']
epoch_count6   = d6['epoch_count']


# Visualize loss history
fig0, ax0 = plt.subplots()
plt.plot(epoch_count1, training_loss2, 'k--', label="Reference sampling, 3e05",markersize=0.1)
plt.plot(epoch_count1, training_loss3, 'b--', label='Chebyshev sampling, 3e05',markersize=0.1)
plt.plot(epoch_count1, training_loss6, 'c--', label='Chebyshev sampling, 2e05',markersize=0.1)
plt.legend()
ax0.set_title('Loss history')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Loss_history_full.png')


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
ax1_full = [ [r[i] for r in check1[:,0,:]] for i in range(1001) ]
ay1_full = [ [r[i] for r in check1[:,1,:]] for i in range(1001) ]
ay1 = [r[1] for r in check1[:,1,:]]
a1  = np.column_stack((ax1,ay1))
a1_full   = [np.column_stack((ax1_full[i],ay1_full[i])) for i in range(1001)] 
bx1  = [r[1] for r in base1[:,0,:]]
by1  = [r[1] for r in base1[:,1,:]]
bx1_full = [[r[i] for r in base1[:,0,:]] for i in range(1001) ]
by1_full = [[r[i] for r in base1[:,1,:]] for i in range(1001) ]
b1   = np.column_stack((bx1,by1))
b1_full   = [np.column_stack((bx1_full[i],by1_full[i])) for i in range(1001)] 


ax2 = [r[1] for r in check2[:,0,:]]
ay2 = [r[1] for r in check2[:,1,:]]
a2  = np.column_stack((ax2,ay2))
ax2_full = [ [r[i] for r in check2[:,0,:]] for i in range(1001) ]
ay2_full = [ [r[i] for r in check2[:,1,:]] for i in range(1001) ]
a2_full   = [np.column_stack((ax2_full[i],ay2_full[i])) for i in range(1001)]
bx2  = [r[1] for r in base2[:,0,:]]
by2  = [r[1] for r in base2[:,1,:]]
b2   = np.column_stack((bx2,by2))
bx2_full = [[r[i] for r in base2[:,0,:]] for i in range(1001) ]
by2_full = [[r[i] for r in base2[:,1,:]] for i in range(1001) ]
b2_full   = [np.column_stack((bx2_full[i],by2_full[i])) for i in range(1001)]


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
mse1 = sklearn.metrics.mean_squared_error(b1[:n_samps],a1[:n_samps])
mse2 = sklearn.metrics.mean_squared_error(b1[:n_samps],a2[:n_samps])
mse3 = sklearn.metrics.mean_squared_error(b1[:n_samps],a3[:n_samps])
mse4 = sklearn.metrics.mean_squared_error(b1[:n_samps],a4[:n_samps])
mse5 = sklearn.metrics.mean_squared_error(b1[:n_samps],a5[:n_samps])
mse6 = sklearn.metrics.mean_squared_error(b1[:n_samps],a6[:n_samps])
#mse7 = sklearn.metrics.mean_squared_error(lf_map,rk_map)

mse2_loss = [sklearn.metrics.mean_squared_error(b2_full[i],a2_full[i]) for i in range(1001) ]
print(mse2_loss)
# Visualize loss history
fig0, ax0 = plt.subplots()
plt.plot(epoch_count1, training_loss1, 'r--')
plt.plot(epoch_count1[0:len(mse1_loss)], mse1_loss)
plt.legend(['Training Loss'])
ax0.set_title('Loss history')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Loss_histories.png') 


print(mse1)
print(mse2)
print(mse3)
print(mse4)
print(mse5)
print(mse6)
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
plt.scatter(samples6[:,0],samples6[:,1],s=0.01)
ax.set_title('Poincare map input - Chebyshev grid')
plt.savefig('pmapin_cheb_og.png')

