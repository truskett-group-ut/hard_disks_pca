import hoomd
import hoomd.hpmc
from numpy import pi, arange
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

hoomd.context.initialize("--mode=cpu")

def Compress(eta_tgt, scale=0.999):
    
    #perform stuff based on rank
    if rank == 0:
        #get current state
        snap = system.take_snapshot()
        N = len(snap.particles.diameter)
        Vp = sum(pi*(snap.particles.diameter**2.0)/4.0)
        Vb = system.box.get_volume()
    else:
        snap = system.take_snapshot()
        N = None
        Vp = None
        Vb = None
    
    #broadcast from 0th rank
    N = comm.bcast(N, root=0)
    Vp = comm.bcast(Vp, root=0)
    Vb = comm.bcast(Vb, root=0)
    
    #assign variables
    eta = Vp/Vb
    eta_init = eta
    
    print '\nStarting from eta={}\n'.format(eta)
    
    #calculate new quantities
    Vb_tgt = Vp/eta_tgt
    
    #box compression loop
    while Vb > Vb_tgt:
        Vb = max(Vb*scale, Vb_tgt)
        eta = Vp/Vb
        new_box = system.box.set_volume(Vb)
        hoomd.update.box_resize(Lx=new_box.Lx, Ly=new_box.Ly, Lz=new_box.Lz, period=None)
        overlaps = mc.count_overlaps()
        
        #run until all overlaps are removed
        while overlaps > 0:
            hoomd.run(100, quiet=True)
            overlaps = mc.count_overlaps()
            
    print "Compressed to eta={}\n".format(eta)
    
    
d = 1.0
etas = arange(0.55, 0.82001, 0.005)
a = (((pi*d**2.0)/4.0)/etas[0])**(1.0/2.0)
n = 64    
    
system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=a), n=n)
mc = hoomd.hpmc.integrate.sphere(d=0.2, seed=1)
mc.shape_param.set('A', diameter=1.0)    
    
equil_steps = 3000000
prod_steps = 3000000
period = prod_steps/1000

for eta in etas:
    print '\n{} from rank {}\n'.format(eta, rank)
    #compress and then equilibrate
    Compress(eta)
    hoomd.run(equil_steps)

    #set up the file writer and run the production version
    d = hoomd.dump.gsd("./trajectories_wider/trajectory_{:.4f}.gsd".format(eta), 
                           period=period, group=hoomd.group.all(), overwrite=True)
    hoomd.run(prod_steps)

    #diable old file writer
    d.disable()    
        
        
comm.Disconnect()