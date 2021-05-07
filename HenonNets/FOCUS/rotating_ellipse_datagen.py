import sys
sys.path.append('../../FOCUS/python')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus
import numpy as np
from coilpy import *

# MPI_INIT
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# run FOCUS
if rank==0:
    print("##### Begin FOCUS run with {:d} CPUs. #####".format(size))
    master = True

test = FOCUSpy(comm=comm, extension='ellipse', verbose=True)
focus.globals.cg_maxiter = 1 # bunk
focus.globals.pp_ns      = 10 # number of fieldlines
focus.globals.pp_maxiter = 10 # number of periods to integrate
test.run(verbose=True)
sys.exit()
