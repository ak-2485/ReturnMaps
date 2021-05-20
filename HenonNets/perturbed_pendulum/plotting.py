import matplotlib.pyplot as plt
import pickle
import numpy as np

infile_circ = 'p_pendulum_data2021-05-20 03:22:41.456724.pickle'
d = pickle.load(open(infile_circ,"rb"))
history_model = d['history_model']
history_rk = d['history_rk']

nics = 20
n_steps = 1000
xic = np.linspace(0.05,.7,nics).reshape([nics,1])
yic = 0.0*np.ones([nics, 1])
yic2 = np.linspace(0.3,.6,nics//2).reshape([nics//2,1])
xic2 = 0.0*np.ones([nics//2, 1])
zic = np.hstack([np.vstack([xic,xic2]),np.vstack([yic,yic2])])


# plot both sections
fig1, ax1 = plt.subplots()
xplot1 = np.ravel(history_model[:,0,:])
yplot1 = np.ravel(history_model[:,1,:])
ax1.plot(xplot1,yplot1,'b.', markersize = .2)
ax1.plot(zic[:,0],zic[:,1],'r.')
ax1.set_title('Poincare plot by HenonNet')
plt.savefig('HNET_pp_circ.png')

fig2, ax2 = plt.subplots()
xplot2 = np.ravel(history_rk[:,0,:])
yplot2 = np.ravel(history_rk[:,1,:])
ax2.plot(xplot2,yplot2,'b.', markersize = .2)
ax2.plot(zic[:,0],zic[:,1],'r.')
ax2.set_title('Poincare plot by Runge-Kutta')
plt.savefig('RK_pp_circ.png')

fig3, ax3 = plt.subplots()
line1,=ax3.plot(xplot1,yplot1,'b.', markersize = .5)
line2,=ax3.plot(xplot2,yplot2,'r.', markersize = .5)
ax3.set_title('HenonNet vs Reference')
plt.legend((line1,line2),('Learned','Reference'), loc='upper right',fontsize='medium')
plt.savefig('HNETvsRK_pp_circ.png')

training_loss = d['training_loss']
epoch_count   = d['epoch_count']


infile_cheb = 'p_pendulum_data2021-05-20 03:19:22.288070.pickle'
d2 = pickle.load(open(infile_cheb,"rb"))
history_model2 = d2['history_model']
history_rk2 = d2['history_rk']
training_loss2 = d2['training_loss']
epoch_count2   = d2['epoch_count']

fig0b, ax0b = plt.subplots()
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count2, training_loss2, 'b--')
plt.legend(['Training Loss'])
ax0b.set_title('Loss Histories')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('log Loss')
plt.savefig('Loss_history_compare_circ_cheb.png')


# plot both sections
fig1, ax1 = plt.subplots()
xplot1ch = np.ravel(history_model2[:,0,:])
yplot1ch = np.ravel(history_model2[:,1,:])
ax1.plot(xplot1ch,yplot1ch,'b.', markersize = .2)
ax1.plot(zic[:,0],zic[:,1],'r.')
ax1.set_title('Poincare plot by HenonNet')
plt.savefig('HNET_pp_cheb.png')

fig2, ax2 = plt.subplots()
xplot2ch = np.ravel(history_rk2[:,0,:])
yplot2ch = np.ravel(history_rk2[:,1,:])
ax2.plot(xplot2ch,yplot2ch,'b.', markersize = .2)
ax2.plot(zic[:,0],zic[:,1],'r.')
ax2.set_title('Poincare plot by Runge-Kutta')
plt.savefig('RK_pp_cheb.png')

fig3, ax3 = plt.subplots()
line1,=ax3.plot(xplot1ch,yplot1ch,'b.', markersize = .5)
line2,=ax3.plot(xplot2ch,yplot2ch,'r.', markersize = .5)
ax3.set_title('HenonNet vs Reference')
plt.legend((line1,line2),('Learned','Reference'), loc='upper right',fontsize='medium')
plt.savefig('HNETvsRK_pp_cheb.png')

fig4, ax4 = plt.subplots()
line1,=ax4.plot(xplot2,yplot2,'b.', markersize = .5)
line2,=ax4.plot(xplot2ch,yplot2ch,'r.', markersize = .5)
ax3.set_title('Reference Circ vs Reference Cheb')
plt.legend((line1,line2),('Reference Circ','Reference Cheb'), loc='upper right',fontsize='medium')
plt.savefig('RKcirc-vs-RKcheb_pp.png')

fig5, ax5 = plt.subplots()
line1,=ax5.plot(xplot1,yplot1,'b.', markersize = .5)
line2,=ax5.plot(xplot1ch,yplot1ch,'r.', markersize = .5)
ax3.set_title('Learned Circ vs Learned Cheb')
plt.legend((line1,line2),('Learned Circ','Learned Cheb'), loc='upper right',fontsize='medium')
plt.savefig('HNETcirc-vs-HNETcheb_pp.png')
