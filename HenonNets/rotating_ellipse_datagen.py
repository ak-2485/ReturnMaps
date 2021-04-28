
import numpy as np
import sys
sys.path.append('../../FOCUS/python')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus
from coilpy import *

def datagen(opt_iter, num_pts, pp_iter, extension):

    # MPI_INIT
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # run FOCUS
    if rank==0:
        print("##### Begin FOCUS run with {:d} CPUs. #####".format(size))
        master = True

    test = FOCUSpy(comm=comm, extension=extension, verbose=True)
    focus.globals.cg_maxiter = opt_iter # bunk
    focus.globals.pp_ns      = num_pts # number of fieldlines
    focus.globals.pp_maxiter = pp_iter # number of periods to integrate
    test.run(verbose=True)

    ref = FOCUSHDF5('focus_ellipse.h5')
    # get the poincare plot points from FOCUS data
    r = ref.ppr - ref.pp_raxis
    z = ref.ppz - ref.pp_zaxis
    # starting points are raw data
    n_samples = len(r[0])
    data = np.hstack([r[0].reshape(n_samples,1),z[0].reshape(n_samples,1)])
    # labels are final integration points from FOCUS
    lf = len(r)-1
    labels = np.hstack([r[lf].reshape(n_samples,1),z[lf].reshape(n_samples,1)])

    return (labels, data)
