"""-------------------------------------------------------------"""
"""   Parallel version of Python code for Alfvenic Turbulence   """
"""-------------------------------------------------------------"""

"""Preliminary Setup"""
from mpi4py import MPI as mpi
import numpy as np 
import numpy.random as rnd
from time import time
##------------------------------------------------------------##

"""Basic definitions"""
# constants (cgs units)
mp = 1.673e-24       # ion (proton) mass
mp2e = 1836.15267389 # proton to electron mass ratio
qe = 4.803e-10       # elementary charge
c = 2.998e10         # speed of light
kB = 1.3807e-16      # Boltzmann constant

# plasma properties
n0 = 1.e10              # plasma number density
T = 1.e6                # ion temperature
vth = np.sqrt(kB*T/mp)  # ion thermal speed

# box
lz = 10.**9       # x dimension
lx, ly = 1., 1.   # y and z dimensions (small compared to lx)

# excited modes
B0 = 1.e4                      # ambient magnetic field, in +z direction
vA = B0/np.sqrt(4*np.pi*n0*mp) # alfven speed
Wtot = 1.e-3                   # total wave energy density
q = 2.                         # power-spectrum index: W(k) = k^-q

# frequencies
omegac = qe*B0/(mp*c)                 # ion gyrofrequency
omegap = np.sqrt(4*np.pi*n0*qe**2/mp) # ion plasma frequency

##------------------------------------------------------------##

"""Adopted code units"""
# r0 (length): skin depth, vA/omegac = c/omegap
r0 = vA/omegac

# m0 (mass): ion mass, mi
m0 = mp

# t0 (time): inverse gyrofrequency, 1/Omegac
t0 = 1./omegac

# v0 (velocity): alfven speed, vA
v0 = vA

# B0 (magnetic field): B0

# E0 (electric field): vA*B0/c
E0 = vA*B0/c

##------------------------------------------------------------##

"""The excited Alfven spectrum"""
# wavevector
# note: k forms an angle theta_k with the +z direction
# note: k_min=1/v_max -> see (Miller, Geussoum, and Ramaty (1990))
#       v_max~10^-3 => k_min~10^3
Nk = 2**8                 # number of excited modes
kmin, kmax = 1.e-1, 1.e5  # minimum and maximum k
lkmin, lkmax = np.log10(kmin), np.log10(kmax)
k_array = 10**np.linspace(lkmin, lkmax, Nk)

# array of theta_k values
# note: they are chosen in a random manner
#       for moderate angles around pi/4
theta_k = rnd.normal(np.pi/4., np.pi/16., Nk)

# dispesion relation
omegak = k_array * np.cos(theta_k)

# random phase array
phi_k = rnd.uniform(0., 2*np.pi, Nk)

# strength of magnetic field perturbations
R = 1.e-2

# normalization factor of Wk
Ck = (1.-q) * R**2 / ( k_array[-1]**(1.-q) - k_array[0]**(1.-q) )

# amplitude of deltaB_k and deltaE_k
amplB_k = np.sqrt(Ck) * k_array**(-q/2.)
amplE_k = np.sqrt(Ck) * k_array**(-1.-q/2.) * omegak

##------------------------------------------------------------##

def gamma(v):
    """
    Lorentz factor
    
    input
    -----
    v: numpy array or float
        dimensionless velocity
    """
    v2 = np.sum(v*v,1)
    return 1. / np.sqrt(1. - (vA/c)**2 * v2)

def Ekin(v):
    """
    Kinetic energy of particles
    (in units of mp*vA^2)
    
    input
    -----
    v: numpy array
        normalized velocities
    """
    return (gamma(v)-1.) * (c/vA)**2.

def compute_deltaB_deltaE(x, t):
    """
    Computes delta_B and delta_E at the same time at (x,t)
    
    input
    -----
    k: numpy array
        normalized wavenumber
    t: float
        time moments
    """
    Nx = len(x[:,0]) # number of particles = number of positions to calculate the fields at
    innx = np.indices( (Nx,Nx) )
    
    # dot product of k and x: k.x = k_x*x + k_z*z = k*x*sin(theta_k) + k*z*cos(theta_k)
    # ... compute (Np x 1) vectors of x and y, so we can multiply by k each position
    xvec = x[:,0][innx][0,:,0][:,np.newaxis]
    yvec = x[:,1][innx][0,:,0][:,np.newaxis]
    # ... and now compute the dot product
    k_dot_x = xvec*k_array*np.sin(theta_k) + yvec*k_array*np.cos(theta_k)
        
    # phase of deltaB_k
    phaseB_k = np.cos(omegak*t - k_dot_x + phi_k)
    # phase of deltaE_k
    phaseE_k = np.sin(omegak*t - k_dot_x + phi_k)
        
    # deltaB at position i, at time t
    deltaB = np.sum(amplB_k*phaseB_k,1)
    # deltaE at position i, at time t
    deltaE = np.sum(amplE_k*phaseE_k,1)

    return deltaB, deltaE

def compute_acc(x, v, t):
    """Acceleration at position x and time t"""
    # magnetic and electric fields at (x,t)
    deltaB, deltaE = compute_deltaB_deltaE(x, t)
    
    v2 = np.sum(v*v,1); gL = gamma(v)
    fact = 1. / ( (c/(gL*vA))**2 + v2 )
    
    # rhs of equations of motion
    dvdt_x = (1./gL - fact * v[:,0]**2) * deltaE + v[:,1]/gL - deltaB * v[:,2] / gL
    dvdt_y = - v[:,0] / gL - fact * deltaE * v[:,0] * v[:,1]
    dvdt_z = deltaB * v[:,0] / gL - fact * deltaE * v[:,0] * v[:,2]
    
    return np.array(zip(dvdt_x, dvdt_y, dvdt_z))

def DKD_step(x, v, t, dt, dth):
    """
    Symplectic stepping scheme (similar to leap-frog method)
    1) First, the position of a particle is advanced by half time step (1st-order Euler scheme)
    2) Then, the velociy is updated by full time step using the positions at half time step
    3) Finally, the new position at full step size is computed using the new velocity
    
    input
    -----
    x: numpy vector
        positions
    v: numpy vector
        velocities
    dt: float
        step size
    dth: float
        half step size
    
    output
    ------
    x, v: numpy arrays
        updated positions and velocities after one time step
    """
    # position at t_i + dt/2 
    # (1st-order Euler scheme)
    x += v*dth
    
    # acceleration at position above
    acc = compute_acc(x, v, t+dth)
    
    # new velocity using updated position and acceleration
    v += acc*dt
    
    # full position update at t_{i+1} = t_i + dt
    x += v*dth
    
    return x, v
    
def particle_integration(Np, nsteps, nsample, nprint, dt, x, v, E, Ekin=Ekin, step=DKD_step):
    """
    Integrate equations of motions starting from the input vectors x, v
    for nsteps with constant time step dt;
        
    input
    -----
    Np: integer
        number of particles to follow
    nsteps: integer
        the number of steps to take during integration
    nsample: integer
        record physical variables of particles (x, v, and Epot) only each nsample-th step
    nprint: integer
        print physical quantities every nprint-th step
    dt: float
        step size
    x, v: vectors of floats
        coordinates, velocities and masses of particles
    R: float
        deltaB/B0
    step: python function
        function to compute step using a given stepping scheme  
    Etot: python function 
        function to compute kinetic energy
    acc: python function
        function to compute mutual acceleration of particles
        
    output
    ------
    tt: numpy vector
        recorded trajectory times
    xt, vt, Ett: numpy vectors
        coordinates, velocities, and kinetic energies     
    
        these arrays are initialized as follows: 
            tt  = np.empty(nsteps/nsample+2)
            xt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
            vt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
            Ett = np.empty(shape=(nsteps/nsample+2,)+np.shape(x[:,0]))
    """
    # initializations
    tt  = np.empty(nsteps/nsample+2)
    xt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
    vt  = np.empty(shape=(nsteps/nsample+2,)+np.shape(x))
    Ekt = np.empty(shape=(nsteps/nsample+2,)+(Np,))
    
    isample, t = 0, 0.
    tt[0], xt[0], vt[0], Ekt[0] = t, x, v, E
    
    # half time-step for DKD scheme's Euler step
    dth = 0.5*dt
        
    # parallelise integration by splitting particles into groups
    comm = mpi.COMM_WORLD
    mpi_rank, mpi_size = comm.Get_rank(), comm.Get_size()
    
    # separate particles in groups, so each processor can work with a separate one
    # ... i.e. define a "local" group of particles for each processor
    loc_x = x[mpi_rank*Np/mpi_size:(mpi_rank+1)*Np/mpi_size,:] # positions of particles for local integration
    loc_v = v[mpi_rank*Np/mpi_size:(mpi_rank+1)*Np/mpi_size,:] # velocities of particles for local integration

    for i in range(nsteps):
        # advance position, velocity, and time
        loc_x, loc_v = step(loc_x, loc_v, t, dt, dth)
        t += dt

        if not i%nprint:
            if mpi_rank==0:
                print("%d steps completed out of %d"%(i, nsteps))
                print(v)

        if not i%nsample:
            # collect results from all processors
            all_x, all_v = comm.gather(loc_x,root=0), comm.gather(loc_v,root=0)
            if mpi_rank==0:
                x, v = np.concatenate(all_x), np.concatenate(all_v)
                isample += 1
                tt[isample] = np.copy(t)
                xt[isample] = np.copy(x)
                vt[isample] = np.copy(v)
                Ekt[isample] = np.copy(Ekin(v))
            
    return tt, xt, vt, Ekt

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
    x_init: numpy array
        initial positions -> array of coordinates (x,y,z)
    v_init: numpy array
        initial velocities -> array of velocity-space coordinates (vx,vy,vz)
    Ekin_init: numpy array
        initial kinetic energies
    """
    
    # parameters
    xmin, ymin, zmin = 0., 0., 0.
    xmax, ymax, zmax = 100., 100., 100.
    mu_v = 0.; sigma_v = vth/vA
    
    # positions: uniform distribution
    x_init = rnd.uniform(xmin, xmax, Np)
    y_init = rnd.uniform(ymin, ymax, Np)
    z_init = rnd.uniform(zmin, zmax, Np)

    # velocities: normal distribution
    vx_init = rnd.normal(mu_v, sigma_v, Np)
    vy_init = rnd.normal(mu_v, sigma_v, Np)
    vz_init = rnd.normal(mu_v, sigma_v, Np)

    # pack-up initial conditions
    x_init = np.array(zip(x_init,y_init,z_init))
    v_init = np.array(zip(vx_init,vy_init,vz_init))
    
    # initial energies
    Ekin_init = Ekin(v_init)
    
    return x_init, v_init, Ekin_init

##------------------------------------------------------------##
# initialize MPI for parallel integration
comm = mpi.COMM_WORLD
mpi_rank, mpi_size = comm.Get_rank(), comm.Get_size()

"""Initialize particles"""
Np = 5
x_init, v_init, Ekin_init = particle_init(Np)  # intital positions and velocities of the Np particles

"""Integrate equations of motion"""
t = 0.; dt = 5.e-4;           # start time and time step in cose units
tmax = 1.e6 * dt              # end time in code units
nsteps = np.int(tmax/dt)-1    # number of time steps
nsample = 100; nprint = 1e5   # dump results every nsample and print every nprint

"""Output file"""

t1 = time() # starting clock time
###----------------------------------------------- Call solver ---------------------------------------###
tout, xout, vout, Eout = particle_integration(Np, nsteps, nsample, nprint, dt, x_init, v_init, Ekin_init)
###---------------------------------------------------------------------------------------------------###
t2 = time() # end clock time

if mpi_rank==0:
    print("integrated in %.2f seconds"%(t2-t1))
    # save data to file
    np.savez('out_file', tout=tout, xout=xout, vout=vout, Eout=Eout)