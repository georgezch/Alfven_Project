"""*************************************************************"""
"""   Parallel version of Python code for Alfvenic Turbulence   """
"""*************************************************************"""

"""Preliminary Setup"""
from mpi4py import MPI as mpi
import numpy as np 
import numpy.random as rnd
import h5py
import os
from time import time
##------------------------------------------------------------##

"""Basic definitions"""
# constants (cgs units)
mp = 1.673e-24       # proton mass
mp2e = 1836.15267389 # proton-to-electron mass ratio
qe = 4.803e-10       # elementary charge
c = 2.998e10         # speed of light
kB = 1.3807e-16      # Boltzmann constant

# plasma properties
n0 = 1.e9               # plasma number density
T = 1.e6                # ion temperature
vth = np.sqrt(kB*T/mp)  # ion thermal speed

# box
lx, ly, lz = 1.e3, 1.e3, 1.e3 # dimensions
x_left, y_left, z_left = 0., 0., 0.                          # left boundaries
x_right, y_right, z_right = x_left+lx, y_left+ly, z_left+lz  # right boundaries

# excited modes
B0 = 1.e4                      # ambient magnetic field, in +z direction
vA = B0/np.sqrt(4*np.pi*n0*mp) # alfven speed
Wtot = 1.e-3                   # total wave energy density
q = 5./3.                      # power-spectrum index: W(k) = W0*k^-q

# frequencies
omegac = qe*B0/(mp*c)                 # ion gyrofrequency
omegap = np.sqrt(4*np.pi*n0*qe**2/mp) # ion plasma frequency
##------------------------------------------------------------##

""" Code units"""
# r0 (length): skin depth, vA/omegac = c/omegap
#r0 = vA/omegac

# m0 (mass): ion mass, mi
#m0 = mp

# t0 (time): inverse gyrofrequency, 1/Omegac
#t0 = 1./omegac

# v0 (velocity): alfven speed, vA
#v0 = vA

# B0 (magnetic field): B0

# E0 (electric field): vA*B0/c
#E0 = vA*B0/c
##------------------------------------------------------------##

"""The excited Alfven spectrum"""
Nk = 2**8  # number of excited modes

# array of theta_k values
# note: they are drawn from a (narrow) normal distribution around pi/4
mu_thetak = np.pi/4.; sigma_thetak = np.pi/10.
theta_k = rnd.normal(mu_thetak, sigma_thetak, Nk)

# wavenumbers
# note: each k forms an angle theta_k with the +z direction
# note: k range is chosen so that the waves resonate 
#       with the particles at the tail of the distribution, with
#       velocities of v~[a,b]*vth
a, b = 2.0, 4.0
vres_min, vres_max = a*vth/vA, b*vth/vA; c2 = (c/vA)**2
gres_min, gres_max = 1./np.sqrt(1. - vres_min**2/c2), 1./np.sqrt(1. - vres_max**2/c2)
ka = 1./(gres_min*vres_min) * 1./np.abs(np.sin(theta_k.min())/vres_min - 1)
kb = 1./(gres_max*vres_max) * 1./np.abs(np.sin(theta_k.max())/vres_max - 1)
kmin, kmax = np.sort([ka,kb])
#kmin, kmax = 1.e-3, 1.e3  # minimum and maximum k
#lkmin, lkmax = np.log10(kmin), np.log10(kmax)
#k_array = 10**np.linspace(lkmin, lkmax, Nk)
k_array = np.linspace(kmin, kmax, Nk)

# dispesion relation
omegak = k_array * np.sin(theta_k)

# random-phase array
phi_k = rnd.uniform(0., 2*np.pi, Nk)

# relative strength of magnetic field perturbations
R = 1.e0

# normalization factor of Wk
Ck = (1.-q) * R**2 / ( k_array[-1]**(1.-q) - k_array[0]**(1.-q) )

# amplitudes of deltaB_k and deltaE_k
amplA_k = np.sqrt(Ck) * k_array**(-1-q/2) / np.cos(theta_k)
amplB_k = np.sqrt(Ck) * k_array**(-q/2.)
amplE_k = np.sqrt(Ck) * k_array**(-q/2.) * np.tan(theta_k)
##------------------------------------------------------------##


"""--------------------------------------------------"""
""" Definitions of functions and integration routine """
"""--------------------------------------------------"""

def gamma(p):
    """
    Lorentz factor
    
    input
    -----
    p: numpy array or float
        dimensionless momentum, i.e. gamma*v
    """
    p2 = np.sum(p*p,axis=1); c2 = (c/vA)**2
    return np.sqrt(1. + p2/c2)

def Ekin(p):
    """
    Kinetic energy of particles
    (in units of mp*vA^2)
    
    input
    -----
    p: numpy array
        normalized momentum, i.e. gamma*v
    """
    return (gamma(p)-1.) * (c/vA)**2.

def pm(v,g):
    """
    Momentum (normalized to m*vA) from velocity
    input
    -----
    v: numpy array
        normalized velocity
    g: numpy array
        Lorentz factor
    """
    p = np.zeros_like(v)
    p[:,0] = v[:,0]*g; p[:,1] = v[:,1]*g; p[:,2] = v[:,2]*g
    return p

def compute_deltaB_deltaE(x, t):
    """
    Computes delta_B and delta_E at the same time at (x,t)
    
    input
    -----
    x: numpy array
        normalized position
    t: float
        time
    """
    Nx = len(x[:,0]) # number of particles = number of positions to calculate the fields at
    innx = np.indices( (Nx,Nx) )
    
    # dot product of k and x: k.x = k_x*x + k_z*z = k*x*sin(theta_k) + k*z*cos(theta_k)
    # ... compute (Np x 1) vectors of x and y, so we can multiply by k each position
    xvec = x[:,0][innx][0,:,0][:,np.newaxis]
    zvec = x[:,2][innx][0,:,0][:,np.newaxis]
    # ... and now compute the dot product
    k_dot_x = xvec*k_array*np.sin(theta_k) + zvec*k_array*np.cos(theta_k)
        
    # phase of deltaB_k
    phaseB_k = np.cos(omegak*t - k_dot_x + phi_k)
    # phase of deltaE_k
    phaseE_k = np.cos(omegak*t - k_dot_x + phi_k)
        
    # deltaB at position i, at time t
    deltaB = -np.sum(amplB_k*phaseB_k,1)
    # deltaE at position i, at time t
    deltaE = -np.sum(amplE_k*phaseE_k,1)

    return deltaB, deltaE

def compute_acc(x, v, t):
    """
    Acceleration at position x and time t
    Note: These are the equations for gamma*v, i.e. dimensionless momentum
    """
    # magnetic and electric fields at (x,t)
    deltaB, deltaE = compute_deltaB_deltaE(x, t)
    
    # rhs of equations of motion
    dpdt_x = deltaE + v[:,1] - deltaB * v[:,2]
    dpdt_y = - v[:,0]
    dpdt_z = deltaB * v[:,0]
    
    return np.array(zip(dpdt_x, dpdt_y, dpdt_z))

def DKD_step(x, v, gL, t, dt):
    """
    Symplectic stepping scheme (similar to leap-frog method)
    1) First, the position of a particle is advanced by half time step (1st-order Euler scheme)
    2) Then, the velociy is updated by full time step using the positions at half time step
    3) Finally, the new position at full step size is computed using the new velocity
    
    input
    -----
    x, v: numpy vectors
        positions, and velocities (NOT gamma*v, this is just v)
    dt: floats
        time step size
    
    output
    ------
    x, v: numpy arrays
        updated positions and velocities after one time step
    """
    # half time-step for Euler step
    dth = 0.5*dt

    # position at t_{i+1/2} + dt/2 (1st-order Euler scheme)
    x += v*dth
    
    # acceleration at position above, d(gamma*v)/dt=dp/dt (in code units)
    acc = compute_acc(x, v, t+dth)
    
    # new velocity using updated position and acceleration
    v[:,0] *= gL; v[:,1] *= gL; v[:,2] *= gL # momentum at start of time step
    v += acc*dt                              # momentum update at full step
    gL = gamma(v)                            # updated Lorentz factor at full step
    v[:,0] /= gL; v[:,1] /= gL; v[:,2] /= gL # velocity update at full step
    
    # full position update at t_{i+1} = t_i + dt = t_{i+1/2} + dt/2
    x += v*dth
    
    return x, v, gL
    
def particle_integration(Np, nsteps, nsample, nprint, dt, x, v, E, g, Ekin=Ekin, step=DKD_step):
    """
    Integrate equations of motions starting from the input vectors x, v
    for nsteps with constant time step dt
        
    input
    -----
    Np: integer
        number of particles to follow
    nsteps: integer
        the number of steps to take during integration
    nsample: integer
        record physical variables of particles (x, v, and Ekin) only each nsample-th step
    nprint: integer
        print physical quantities every nprint-th step
    dt: float
        step size
    x, v, g: vectors of floats
        coordinates, velocities, and Lorentz gamma factors of particles
    R: float
        deltaB/B0
    step: python function
        function to compute step using a given stepping scheme  
    Ekin: python function 
        function to compute kinetic energy
    acc: python function
        function to compute acceleration of particles
        
    output
    ------
    tt: numpy vector
        recorded trajectory times
    xt, vt, Ekt: numpy vectors
        coordinates, velocities, and kinetic energies     
    
        these arrays are initialized as follows: 
            tt  = np.empty(nsteps/nsample+2)
            xt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
            vt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
            Ekt = np.empty(shape=(nsteps/nsample+2,)+(Np,))
    """
    # parallelise integration by splitting particles into groups
    comm = mpi.COMM_WORLD
    # force all processors to wait here 
    comm.Barrier()
    mpi_rank, mpi_size = comm.Get_rank(), comm.Get_size()

    # initializations
    tt  = np.empty(nsteps/nsample+2)
    xt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
    vt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
    Ekt = np.empty(shape=(nsteps/nsample+2,)+(Np,))
    
    isample, t = 0, 0.
    tt[0], xt[0], vt[0], Ekt[0] = t, x, v, E
    t_esc = []; xt_esc = []; vt_esc = []; Ekin_esc = []

    if mpi_rank==0:
        # save initial data
        rep = '0'*len(str(nsteps/nsample+2))
        x_h5f    = h5py.File(dir_x+'x0%s0.h5'%(rep)      , 'w')
        v_h5f    = h5py.File(dir_v+'v0%s0.h5'%(rep)      , 'w')
        Ekin_h5f = h5py.File(dir_Ekin+'Ekin0%s0.h5'%(rep), 'w')
        x_h5f.create_dataset('data', data=xt[0])
        v_h5f.create_dataset('data', data=vt[0])
        Ekin_h5f.create_dataset('data', data=Ekt[0])
        x_h5f.close(); v_h5f.close(); Ekin_h5f.close()
    
    # split particles in groups, so that each processor can work with a separate bunch of them
    # ... i.e. define a "local" group of particles for each processor
    loc_x = x[mpi_rank*Np/mpi_size:(mpi_rank+1)*Np/mpi_size,:] # positions of particles for local integration
    loc_v = v[mpi_rank*Np/mpi_size:(mpi_rank+1)*Np/mpi_size,:] # velocities of particles for local integration
    loc_g = g[mpi_rank*Np/mpi_size:(mpi_rank+1)*Np/mpi_size]   # Lorentz factors of particles for local integration

    for i in range(nsteps):
        # advance position, velocity, and time for each group of particles separately
        loc_x, loc_v, loc_g = step(loc_x, loc_v, loc_g, t, dt)
        t += dt

        if not i%nprint:
            if mpi_rank==0:
                print("%d steps completed out of %d (%.2f pct)"%(i, nsteps, float(i)/float(nsteps) * 100.))

        if not i%nsample:            
            # check for escaped particles
            xout_check, yout_check, zout_check  = loc_x[:,0], loc_x[:,1], loc_x[:,2]
            vxout_check, vyout_check, vzout_check = loc_v[:,0], loc_v[:,1], loc_v[:,2]

            xind_outofbox = np.where( (xout_check < x_left) | (xout_check > x_right) )[0]
            yind_outofbox = np.where( (yout_check < y_left) | (yout_check > y_right) )[0]
            zind_outofbox = np.where( (zout_check < z_left) | (zout_check > z_right) )[0]
            if not ( len(xind_outofbox)==0 or len(yind_outofbox)==0 or len(zind_outofbox)==0 ):
                ind_out = list(set(np.concatenate([xind_outofbox,yind_outofbox,zind_outofbox])))
                
                loc_esc_x[:,0] = xout_check[ind_out]
                loc_esc_x[:,1] = yout_check[ind_out]
                loc_esc_x[:,2] = zout_check[ind_out]
                loc_x_esc = np.array(zip(loc_esc_x,loc_esc_y,loc_esc_z))

                loc_esc_vx[:,0] = vxout_check[ind_out]
                loc_esc_vy[:,1] = vyout_check[ind_out]
                loc_esc_vz[:,2] = vzout_check[ind_out]
                loc_v_esc = np.array(zip(loc_esc_vx,loc_esc_vy,loc_esc_vz))

                loc_g_esc = loc_g[ind_out]

                # collect results from all processors
                all_esc_x, all_esc_v, all_esc_g = comm.gather(loc_x_esc,root=0), comm.gather(loc_v_esc,root=0), comm.gather(loc_g_esc,root=0)
                if mpi_rank==0:
                    x_esc, v_esc, g_esc = np.concatenate(all_x_esc), np.concatenate(all_v_esc), np.concatenate(all_g_esc)

                    t_esc.append(t)
                    xt_esc.append(x_esc); vt_esc.append(v_esc); gt_esc.append(g_esc)
                    p_esc = pm(vt_esc[-1],gt_esc[-1])
                    Ekin_esc.append(Ekin[p_esc])
                    print(t_esc)

                loc_x[:,0] = np.delete(xout_check, ind_out)
                loc_x[:,1] = np.delete(yout_check, ind_out)
                loc_x[:,2] = np.delete(zout_check, ind_out)

                loc_v[:,0] = np.delete(vxout_check, ind_out)
                loc_v[:,1] = np.delete(vyout_check, ind_out)
                loc_v[:,2] = np.delete(vzout_check, ind_out)

                loc_g = np.delete(loc_g, ind_out)

            # collect results from all processors
            all_x, all_v, all_g = comm.gather(loc_x,root=0), comm.gather(loc_v,root=0), comm.gather(loc_g,root=0)
            if mpi_rank==0:
                x, v, g = np.concatenate(all_x), np.concatenate(all_v), np.concatenate(all_g)
                p = pm(v,g)
                isample += 1

                # save data
                rep      = '0'*(len(str(nsteps/nsample+2))-len(str(isample)))
                x_h5f    = h5py.File(dir_x+'x0%s%d0.h5'%(rep,isample)      , 'w')
                v_h5f    = h5py.File(dir_v+'v0%s%d0.h5'%(rep,isample)      , 'w')
                Ekin_h5f = h5py.File(dir_Ekin+'Ekin0%s%d0.h5'%(rep,isample), 'w')
                x_h5f.create_dataset('data', data=np.copy(x))
                v_h5f.create_dataset('data', data=np.copy(v))
                Ekin_h5f.create_dataset('data', data=np.copy(Ekin(p)))
                x_h5f.close(); v_h5f.close(); Ekin_h5f.close()
            
    return

def particle_init(Np):
    """
    Initialize positions and velocities
        Positions are uniformly distributed
        Velocities are distributed normaly
    
    input
    -----
    Np: int
        number of particles
    
    output
    ------
    x_init, v_init: numpy arrays
        initial positions and velocities (gamma * v)
    Ekin_init: numpy array
        initial kinetic energies
    gL: float
        intital Lorentz gamma factor
    """
    
    # parameters
    xmin, ymin, zmin = x_left, y_left, z_left
    xmax, ymax, zmax = x_right*2., y_right, z_right
    mu_vx = 0.; sigma_vx = vth/vA/np.sqrt(3.)
    mu_vy = 0.; sigma_vy = vth/vA/np.sqrt(3.)
    mu_vz = 0.; sigma_vz = vth/vA/np.sqrt(3.)
    
    # positions: uniform distribution
    x_init = rnd.uniform(xmin, xmax, Np)
    y_init = rnd.uniform(ymin, ymax, Np)
    z_init = rnd.uniform(zmin, zmax, Np)

    # velocities: normal distribution
    vx_init = rnd.normal(mu_vx, sigma_vx, Np)
    vy_init = rnd.normal(mu_vy, sigma_vy, Np)
    vz_init = rnd.normal(mu_vz, sigma_vz, Np)

    # pack-up initial conditions
    x_init = np.array(zip(x_init,y_init,z_init))
    v_init = np.array(zip(vx_init,vy_init,vz_init))
    
    # Lorentz factors
    v2 = np.sum(v_init*v_init,axis=1); c2 = (c/vA)**2
    gL_init = 1./np.sqrt(1. - v2/c2)
    p_init = pm(v_init,gL_init)

    # initial energies
    Ekin_init = Ekin(p_init)
    
    return x_init, v_init, Ekin_init, gL_init
##------------------------------------------------------------##

# initialize MPI for parallel integration
comm = mpi.COMM_WORLD
# name of each processor (mpi_rank) and total number of them (mpi_size)
mpi_rank, mpi_size = comm.Get_rank(), comm.Get_size() 

"""Initialize save files"""
dir_output = './Output/'
dir_x      = dir_output+'x/'
dir_v      = dir_output+'v/'
dir_Ekin   = dir_output+'Ekin/'

if mpi_rank==0:
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if not os.path.exists(dir_x):
        os.makedirs(dir_x)
    if not os.path.exists(dir_v):
        os.makedirs(dir_v)
    if not os.path.exists(dir_Ekin):
        os.makedirs(dir_Ekin)

"""Initialize particles"""
Np = 10
x_init, v_init, Ekin_init, gamma_Lor = particle_init(Np)  # intital positions, velocities, and Lorentz factor of the Np particles

"""Integrate equations of motion"""
t0 = 0.; dt = 2.e-3;            # start time and time step in code units
tmax = 1.e6 * dt                # end time in code units
nsteps = np.int(tmax/dt)-1      # number of time steps
nsample = 1000; nprint = 100000    # dump results every nsample and print every nprint

t1 = time() # start clock time
###------------------------------------------------- Call solver ------------------------------------------------###
tout, xout, vout, Eout = particle_integration(Np, nsteps, nsample, nprint, dt, x_init, v_init, Ekin_init, gamma_Lor)
###--------------------------------------------------------------------------------------------------------------###
t2 = time() # end clock time

print("integrated %d particles in %.2f seconds"%(Np,t2-t1))




