import sys
#sys.path.append('../../FOCUS/python')
sys.path.append('../../../../research/FOCUS/python')
from mpi4py import MPI
from focuspy import FOCUSpy
import focus

def FOCUSrun(opt_iter, pp_iter, num_pts, extension):
    """
    inputs:
        opt_iter  :: int; number of CG optimization steps
        pp_iter   :: int; number of poincare periods to integrate
        num_pts   :: int; number of initial points for poincare plot
        extension :: string; file extension for FOUCS inputs
    """
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

sys.exit()
