import numpy as np
import numpy as np
import random
import copy

class Particle:
    """
    The Particle class encapsulates information about the particles 
    used in the simulation.
    
    Attributes
    ----------
    
    ----------
    
    """
    
    def __init__(self, pos, vel, mpw):
        """
        Initializes the Particle object with the position, 
        speed, and macrparticle weight.

        Parameters
        ----------
        pos : numpy.ndarray
            particle position vector
        vel : numpy.ndarray
            particle velocity vector
        mpw : float
            macroparticle weight  
        ----------
        
        """
        
        self.pos = pos
        self.vel = vel
        self.mpw = mpw
        

class Species:
    """
    The Species class encapsulates information about the species 
    used in the simulation.
    
    Attributes
    ----------
    
    ----------
    
    """
    def __init__(self, name, mass, charge, worldObj):
        """
        Initializes the Species object with the name, mass,
        charge.

        Parameters
        ----------
        name : str
            species name
        mass : float
            species mass
        charge : float
            species charge
        mpw : float
            macroparticle weight  
        ----------

        """
        
        self.particleList = []
        
        self.name   = name
        self.mass   = mass
        self.charge = charge
        
        self.den    = np.zeros((worldObj.ni, worldObj.nj, worldObj.nk))
        
        self.worldObj = worldObj
    
    def addParticle(self, pos, vel, mpw):
        """random.random()
        add a particle to particleList

         Parameters
        ----------
        pos : numpy.ndarray
            particle position vector
        vel : numpy.ndarray
            particle velocity vector
        mpw : float
            macroparticle weight  
        ----------
        
        """
        #get logical coordinate of particle's position
        lc = self.worldObj.XtoL(pos)

        #electric field at particle position
        ef_part = self.gather_ef(lc)

        #rewind velocity back by 0.5*dt*ef
        vel -=  self.charge/self.mass*ef_part*(0.5*self.worldObj.dt);
        
        #add particle to list
        self.particleList.append(Particle(pos, vel, mpw))
        
        
    def loadParticlesBox(self, x1, x2, num_den, num_mp):
        """
        loads randomly distributed particles in a x1-x2 box 
        representing num_den number density

        Parameters
        ----------
        x1 : numpy.ndarray
            origin of bounding box
        x2 : numpy.ndarray
            max. bound corner of box
        num_den : float
            number density
        num_mp  : number of macroparticles
        ----------

        """
        box_vol  = (x2[0]-x1[0])*(x2[1]-x1[1])*(x2[2]-x1[2]) # box vol.
        num_real = num_den * box_vol;                   #number of real particles
        mpw      = num_real/num_mp;                     # macroparticle weight
        
        self.box_vol = box_vol
        self.num_real= num_real
        self.mpw     = mpw

        #load particles on an equally spaced grid
        for p in np.arange(0,num_mp+1):
            # sample random position
            pos = np.zeros(3)
            vel = np.zeros(3)

            #rnd = random.random()

            pos[0] = x1[0] + random.random()*(x2[0]-x1[0]);
            pos[1] = x1[1] + random.random()*(x2[1]-x1[1]);
            pos[2] = x1[2] + random.random()*(x2[2]-x1[2]);

            #set initial velocity
            vel[0] = 0.0
            vel[1] = 0.0
            vel[2] = 0.0

            self.addParticle(pos,vel,mpw); # add new particle
            
    def loadParticlesBoxQS(self, x1, x2, num_den, num_mp):
        """
        loads randomly distributed particles in a x1-x2 box 
        representing num_den number density

        Parameters
        ----------
        x1 : numpy.ndarray
            origin of bounding box
        x2 : numpy.ndarray
            max. bound corner of box
        num_den : float
            number density
        num_mp  : numpy.ndarray
        ----------

        """
        box_vol  = (x2[0]-x1[0])*(x2[1]-x1[1])*(x2[2]-x1[2]) # box vol.
        num_real = num_den * box_vol;                   #number of real particles
        num_mp_tot = (num_mp[0]-1)*(num_mp[1]-1)*(num_mp[2]-1)
        mpw      = num_real/num_mp_tot;                     # macroparticle weight
        
        self.box_vol = box_vol
        self.num_real= num_real
        self.mpw     = mpw
        
        di = (x2[0]-x1[0])/(num_mp[0]-1);
        dj = (x2[1]-x1[1])/(num_mp[1]-1);
        dk = (x2[2]-x1[2])/(num_mp[2]-1);

        #load particles on a equally spaced grid
        
        for i in np.arange(0,num_mp[0]):
            for j in np.arange(0,num_mp[1]):
                for k in np.arange(0,num_mp[2]):
                    
                    # sample random position
                    pos = np.zeros(3)
                    vel = np.zeros(3)
                    
                    pos[0] = x1[0] + i*di;
                    pos[1] = x1[1] + j*dj;
                    pos[2] = x1[2] + k*dk;

                    # shift particles on max faces back to the domain
                    if (pos[0]==x2[0]): pos[0]-=1e-4*di;
                    if (pos[1]==x2[1]): pos[1]-=1e-4*dj;
                    if (pos[2]==x2[2]): pos[2]-=1e-4*dk;

                    w = 1;
                    if (i==0 or i==num_mp[0]-1): w*=0.5
                    if (j==0 or j==num_mp[1]-1): w*=0.5
                    if (k==0 or k==num_mp[2]-1): w*=0.5

                    #set initial velocity
                    vel[0] = 0.0
                    vel[1] = 0.0
                    vel[2] = 0.0

                    self.addParticle(pos,vel,mpw*w)
                    
    
    def scatter_den(self, lc, value):
        """
        scatters scalar value onto a field at logical coordinate lc

        Parameters
        ----------
        lc : numpy.ndarray
            logical coordinate 
        ----------
        """
        
        if lc [0] < 0 or \
             lc [0] >=self.worldObj.ni-1 or \
             lc [1] < 0 or \
             lc [1] >=self.worldObj.nj-1 or \
             lc [2] < 0 or \
             lc [2] >=self.worldObj.nk-1:
            
            print("WARNING: point outside domain")
             
        i  = int(lc[0])
        di = lc[0]-i

        j  = int(lc[1])
        dj = lc[1]-j

        k  = int(lc[2])
        dk = lc[2]-k

        self.den[i][j][k]      += value*(1-di)*(1-dj)*(1-dk)
        self.den[i+1][j][k]    += value*(di)*(1-dj)*(1-dk)
        self.den[i+1][j+1][k]  += value*(di)*(dj)*(1-dk)
        self.den[i][j+1][k]    += value*(1-di)*(dj)*(1-dk)
        self.den[i][j][k+1]    += value*(1-di)*(1-dj)*(dk)
        self.den[i+1][j][k+1]  += value*(di)*(1-dj)*(dk)
        self.den[i+1][j+1][k+1]+= value*(di)*(dj)*(dk)
        self.den[i][j+1][k+1]  += value*(1-di)*(dj)*(dk)
        
    def computeNumberDensity(self):
        """
        Compute particle number density

        Parameters
        ----------
        
        ----------
        """
        self.den = np.zeros((self.worldObj.ni, self.worldObj.nj, self.worldObj.nk))
        
        for particle in self.particleList:
            lc = self.worldObj.XtoL(particle.pos)
            self.scatter_den(lc, self.mpw)
            
        self.den = self.den / self.worldObj.node_vol
        
    
    def gather_ef(self, lc):
        """
        gathers field value at logical coordinate lc

        Parameters
        ----------
        lc : numpy.ndarray
            logical coordinate 
        data : numpy.ndarray
            electric field array
        ----------
        """
        
        if lc [0] < 0 or \
             lc [0] >=self.worldObj.ni-1 or \
             lc [1] < 0 or \
             lc [1] >=self.worldObj.nj-1 or \
             lc [2] < 0 or \
             lc [2] >=self.worldObj.nk-1:
            
            print("WARNING: point outside domain")
             
        i  = int(lc[0])
        di = lc[0]-i

        j  = int(lc[1])
        dj = lc[1]-j

        k  = int(lc[2])
        dk = lc[2]-k

        # gather electric field onto particle position
        
        ef_x  = self.worldObj.ef[i][j][k][0]*(1-di)*(1-dj)*(1-dk)+\
                self.worldObj.ef[i+1][j][k][0]*(di)*(1-dj)*(1-dk)+\
                self.worldObj.ef[i+1][j+1][k][0]*(di)*(dj)*(1-dk)+\
                self.worldObj.ef[i][j+1][k][0]*(1-di)*(dj)*(1-dk)+\
                self.worldObj.ef[i][j][k+1][0]*(1-di)*(1-dj)*(dk)+\
                self.worldObj.ef[i+1][j][k+1][0]*(di)*(1-dj)*(dk)+\
                self.worldObj.ef[i+1][j+1][k+1][0]*(di)*(dj)*(dk)+\
                self.worldObj.ef[i][j+1][k+1][0]*(1-di)*(dj)*(dk)
        
        ef_y  = self.worldObj.ef[i][j][k][1]*(1-di)*(1-dj)*(1-dk)+\
                self.worldObj.ef[i+1][j][k][1]*(di)*(1-dj)*(1-dk)+\
                self.worldObj.ef[i+1][j+1][k][1]*(di)*(dj)*(1-dk)+\
                self.worldObj.ef[i][j+1][k][1]*(1-di)*(dj)*(1-dk)+\
                self.worldObj.ef[i][j][k+1][1]*(1-di)*(1-dj)*(dk)+\
                self.worldObj.ef[i+1][j][k+1][1]*(di)*(1-dj)*(dk)+\
                self.worldObj.ef[i+1][j+1][k+1][1]*(di)*(dj)*(dk)+\
                self.worldObj.ef[i][j+1][k+1][1]*(1-di)*(dj)*(dk)
        
        ef_z  = self.worldObj.ef[i][j][k][2]*(1-di)*(1-dj)*(1-dk)+\
                self.worldObj.ef[i+1][j][k][2]*(di)*(1-dj)*(1-dk)+\
                self.worldObj.ef[i+1][j+1][k][2]*(di)*(dj)*(1-dk)+\
                self.worldObj.ef[i][j+1][k][2]*(1-di)*(dj)*(1-dk)+\
                self.worldObj.ef[i][j][k+1][2]*(1-di)*(1-dj)*(dk)+\
                self.worldObj.ef[i+1][j][k+1][2]*(di)*(1-dj)*(dk)+\
                self.worldObj.ef[i+1][j+1][k+1][2]*(di)*(dj)*(dk)+\
                self.worldObj.ef[i][j+1][k+1][2]*(1-di)*(dj)*(dk)
        
        ef_part = np.array([ef_x, ef_y, ef_z])
        
        return ef_part
    
    
    def advance(self):
        #get the time step
        dt = self.worldObj.dt

        #save mesh bounds
        x0 = self.worldObj.x0
        xm = self.worldObj.xm

        # loop over all particles
        for particle in self.particleList:
        
            #get logical coordinate of particle's position
            lc = self.worldObj.XtoL(particle.pos)

            #electric field at particle position
            ef_part = self.gather_ef(lc)

            #update velocity from F=qE
            particle.vel += ef_part*(dt*self.charge/self.mass)

            #update position from v=dx/dt
            particle.pos += particle.vel*dt

            #did this particle leave the domain? reflect back
            for i in np.arange(3):
            
                if particle.pos[i] < x0[i]:
                    particle.pos[i]=2*x0[i]-particle.pos[i]
                    particle.vel[i]*=-1.0
                    
                elif particle.pos[i] >= xm[i]:
                    particle.pos[i]=2*xm[i]-particle.pos[i]
                    particle.vel[i]*=-1.0