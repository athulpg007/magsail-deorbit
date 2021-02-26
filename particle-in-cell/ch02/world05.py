import numpy as np
import numpy as np
import random
import copy
from numba import jit

class World:
    """
    The World class encapsulates information about the computational domain.
    
    Attributes
    ----------
    
    ----------
    
    """
    
    def __init__(self, ni, nj, nk):
        """
        Initializes the planet object with the planetary constants.

        Parameters
        ----------
        ni : int
            number of grid points along x coordinate
        nj : int
            number of grid points along y coordinate
        nk : int
            number of grid points along z coordinate
        ----------
        
        """
        
        self.ni = ni
        self.nj = nj
        self.nk = nk
        
        self.nn = np.zeros(3)
        
        self.nn[0] = self.ni
        self.nn[1] = self.nj
        self.nn[2] = self.nk
        
        self.x0 = np.zeros(3)
        self.dh = np.zeros(3)
        self.xm = np.zeros(3)
        self.xc = np.zeros(3)
        
        self.EPS_0 = 8.85418782e-12
        self.QE    = 1.602176565e-19;
        self.AMU   = 1.660538921e-27
        self.ME    = 9.10938215e-31;
        self.K     = 1.380648e-23;
        self.EvToK = self.QE/self.K;
        
        self.phi        = np.zeros((self.ni, self.nj, self.nk))
        self.phi_new    = np.zeros((self.ni, self.nj, self.nk))
        self.R          = np.zeros((self.ni, self.nj, self.nk))
        self.rho        = np.zeros((self.ni, self.nj, self.nk))
        self.node_vol   = np.zeros((self.ni, self.nj, self.nk))
        self.ef         = np.zeros((self.ni, self.nj, self.nk, 3))
    
    def setTime(self, dt, num_ts):
        self.dt     = dt
        self.num_ts = num_ts
        
    def setExtents(self, x1, y1, z1, x2, y2, z2):
        
        """
        Set mesh extents and compute grid spacing.

        Parameters
        ----------
        x1 : float
            x-coordinate of world origin
        y1 : float
            y-coordinate of world origin
        z1 : float
            z-coordinate of world origin
        x2 : float
            x-coordinate of world max bound
        y2 : float
            y-coordinate of world max bound
        z2 : float
            z-coordinate of world max bound
        ----------
        
        """
        
        self.x0[0] = x1
        self.x0[1] = y1
        self.x0[2] = z1
        
        self.xm[0] = x2
        self.xm[1] = y2
        self.xm[2] = z2
        
        for i in range(3):
            self.dh[i] = (self.xm[i] - self.x0[i]) / (self.nn[i] - 1)
            self.xc[i] = 0.5*(self.x0[i] + self.xm[i]) 
        
        self.computeNodeVolumes()
    
    def computeDebyeLength(self, Te, ne):
        """
        Compute the Debye length.

        Parameters
        ----------
        Te : float
            electron temperature, K
        ne : float
            number density, m3
        ----------
        
        """
        
        return np.sqrt(self.EPS_0*self.K*Te/(ne*self.QE**2))
    
    
    def potentialSolver(self, max_it, tol):
        """
        Compute the potential field.

        Parameters
        ----------
        max_it : int
            max iterations for Gauss-Seidel
        tol: float
            tolerance for Gauss-Seidel
        ----------
        
        """

        dx2 = 1.0/(self.dh[0]*self.dh[0]); # dx^2
        dy2 = 1.0/(self.dh[1]*self.dh[1]); # dy^2
        dz2 = 1.0/(self.dh[2]*self.dh[2]); # dz^2
    
        L2 = 0.0 # norm
        
        converged = False
        
        # solve potential
        for it in np.arange(1,max_it+1):
            for i in np.arange(1,self.ni-1):
                for j in np.arange(1,self.nj-1):
                    for k in np.arange(1,self.nk-1):
                        #standard internal open node
                        phi_new = (self.rho[i][j][k]/self.EPS_0 +\
                                        dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                                        dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                                        dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1]))/(2*dx2+2*dy2+2*dz2)
                        
                        # sucessive over relaxation  
                        self.phi[i,j,k] = self.phi[i,j,k] + 1.4*(phi_new - self.phi[i][j][k])
        

            #check for convergence*/
            if it%25==0:
                sum = 0;
                for i in np.arange(1,self.ni-1):
                    for j in np.arange(1,self.nj-1):
                        for k in np.arange(1,self.nk-1):

                            R = -self.phi[i][j][k]*(2*dx2+2*dy2+2*dz2) +\
                            self.rho[i][j][k]/self.EPS_0 +\
                            dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                            dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                            dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1])

                            sum += R*R;


                L2 = np.sqrt(sum/(self.ni*self.nj*self.nk));
                #print("iter: "+str(it)+", L2 = "+str(L2))
                if (L2<tol):
                    converged = True
                    break
                    
        if (converged==False):
            print("Gauss-Seidel failed to converge, L2 = "+str(L2))
        
        return converged
    
    def potentialSolver2(self, max_it, tol):
        """
        Compute the potential field.

        Parameters
        ----------
        max_it : int
            max iterations for Gauss-Seidel
        tol: float
            tolerance for Gauss-Seidel
        ----------
        
        """

        dx2 = 1.0/(self.dh[0]*self.dh[0]); # dx^2
        dy2 = 1.0/(self.dh[1]*self.dh[1]); # dy^2
        dz2 = 1.0/(self.dh[2]*self.dh[2]); # dz^2
    
        L2 = 0.0 # norm
        
        converged = False
        
        
        # solve potential
        for it in np.arange(1,max_it+1):
            """
            for i in np.arange(1,self.ni-1):
                for j in np.arange(1,self.nj-1):
                    for k in np.arange(1,self.nk-1):
            """            
            """
            #standard internal open node
                        phi_new = (self.rho[i][j][k]/self.EPS_0 +\
                                        dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                                        dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                                        dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1]))/(2*dx2+2*dy2+2*dz2)
                                        
            # sucessive over relaxation  
                        self.phi[i,j,k] = self.phi[i,j,k] + 1.4*(phi_new - self.phi[i][j][k])
        
            """            
            #standard internal open node
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                            (self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                            dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1])+\
                            dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1])+\
                            dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk]))/(2*dx2+2*dy2+2*dz2)
            
            """
            # sucessive over relaxation  
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] +\
                                                        1.8*(self.phi_new[1:self.ni-1,1:self.nj-1,1:self.nk-1] - \
                                                        self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1])
            """
            
            #check for convergence*/
            if it%25==0:
                sum = 0;
                """
                for i in np.arange(1,self.ni-1):
                    for j in np.arange(1,self.nj-1):
                        for k in np.arange(1,self.nk-1):
                """

                self.R[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                -self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1]*(2*dx2+2*dy2+2*dz2) +\
                self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1]) +\
                dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1]) +\
                dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk])

                sum = np.sum(self.R**2)

                L2 = np.sqrt(sum/(self.ni*self.nj*self.nk));
                #print("iter: "+str(it)+", L2 = "+str(L2))
                if (L2<tol):
                    converged = True
                    break

        if (converged==False):
            print("Gauss-Seidel failed to converge, L2 = "+str(L2))
        
        return converged
    
    @jit
    def potentialSolver3(self, w, max_it, tol):
        """
        Compute the potential field.

        Parameters
        ----------
        max_it : int
            max iterations for Gauss-Seidel
        tol: float
            tolerance for Gauss-Seidel
        ----------
        
        """

        dx2 = 1.0/(self.dh[0]*self.dh[0]); # dx^2
        dy2 = 1.0/(self.dh[1]*self.dh[1]); # dy^2
        dz2 = 1.0/(self.dh[2]*self.dh[2]); # dz^2
    
        L2 = 0.0 # norm
        
        converged = False
        
        # Step 1: create *integer* array the same size as u 
        x = np.zeros_like(self.phi,dtype=np.int)

        # Step 2: populate all non-boundary cells with running numbers from 1 to (n-2)^2
        x[1:-1,1:-1,1:-1] = np.arange(1,(self.ni-2)*(self.nj-2)*(self.nk-2)+1).reshape(self.ni-2,self.nj-2,self.nk-2)

        # Step 3: get indices of even (red) and odd (black) points
        ir, jr, kr = np.where((x>0) & (x%2 == 0)) # indices of red pts = indices of even numbers
        ib, jb, kb = np.where((x>0) & (x%2 == 1)) # indices of black pts = indices of odd numbers


        
        # solve potential
        for it in np.arange(1,max_it+1):
            """
            for i in np.arange(1,self.ni-1):
                for j in np.arange(1,self.nj-1):
                    for k in np.arange(1,self.nk-1):
            """            
            """
            #standard internal open node
                        phi_new = (self.rho[i][j][k]/self.EPS_0 +\
                                        dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                                        dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                                        dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1]))/(2*dx2+2*dy2+2*dz2)
                                        
            # sucessive over relaxation  
                        self.phi[i,j,k] = self.phi[i,j,k] + 1.4*(phi_new - self.phi[i][j][k])
        
            """ 
            
            """
            #standard internal open node
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                            (self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                            dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1])+\
                            dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1])+\
                            dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk]))/(2*dx2+2*dy2+2*dz2)
            
            """
            """
            # sucessive over relaxation  
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] +\
                                                        1.8*(self.phi_new[1:self.ni-1,1:self.nj-1,1:self.nk-1] - \
                                                        self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1])
            """
            
            # Red point update
            self.phi[ir,jr,kr] = (1-w)*self.phi[ir,jr,kr] + (1.0/6.0)*w*(self.phi[ir+1,jr,kr] + self.phi[ir-1,jr,kr] +\
                                                                    self.phi[ir,jr+1,kr] + self.phi[ir,jr-1,kr] +\
                                                                    self.phi[ir,jr,kr+1] + self.phi[ir,jr,kr-1] +\
                                                                    (self.rho[ir,jr,kr]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            # Black point update
            self.phi[ib,jb,kb] = (1-w)*self.phi[ib,jb,kb] + (1.0/6.0)*w*(self.phi[ib+1,jb,kb] + self.phi[ib-1,jb,kb] +\
                                                                    self.phi[ib,jb+1,kb] + self.phi[ib,jb-1,kb] +\
                                                                    self.phi[ib,jb,kb+1] + self.phi[ib,jb,kb-1] +\
                                                                    (self.rho[ib,jb,kb]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            #check for convergence*/
            if it%25==0:
                sum = 0;
                """
                for i in np.arange(1,self.ni-1):
                    for j in np.arange(1,self.nj-1):
                        for k in np.arange(1,self.nk-1):
                """

                self.R[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                -self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1]*(2*dx2+2*dy2+2*dz2) +\
                self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1]) +\
                dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1]) +\
                dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk])

                sum = np.sum(self.R**2)

                L2 = np.sqrt(sum/(self.ni*self.nj*self.nk));
                #print("iter: "+str(it)+", L2 = "+str(L2))
                if (L2<tol):
                    converged = True
                    break

        if (converged==False):
            print("Gauss-Seidel failed to converge, L2 = "+str(L2))
        
        return converged
    
    def potentialSolver3(self, w, max_it, tol):
        """
        Compute the potential field.

        Parameters
        ----------
        max_it : int
            max iterations for Gauss-Seidel
        tol: float
            tolerance for Gauss-Seidel
        ----------
        
        """

        dx2 = 1.0/(self.dh[0]*self.dh[0]); # dx^2
        dy2 = 1.0/(self.dh[1]*self.dh[1]); # dy^2
        dz2 = 1.0/(self.dh[2]*self.dh[2]); # dz^2
    
        L2 = 0.0 # norm
        
        converged = False
        
        # Step 1: create *integer* array the same size as u 
        x = np.zeros_like(self.phi,dtype=np.int)

        # Step 2: populate all non-boundary cells with running numbers from 1 to (n-2)^2
        x[1:-1,1:-1,1:-1] = np.arange(1,(self.ni-2)*(self.nj-2)*(self.nk-2)+1).reshape(self.ni-2,self.nj-2,self.nk-2)

        # Step 3: get indices of even (red) and odd (black) points
        ir, jr, kr = np.where((x>0) & (x%2 == 0)) # indices of red pts = indices of even numbers
        ib, jb, kb = np.where((x>0) & (x%2 == 1)) # indices of black pts = indices of odd numbers


        
        # solve potential
        for it in np.arange(1,max_it+1):
            """
            for i in np.arange(1,self.ni-1):
                for j in np.arange(1,self.nj-1):
                    for k in np.arange(1,self.nk-1):
            """            
            """
            #standard internal open node
                        phi_new = (self.rho[i][j][k]/self.EPS_0 +\
                                        dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                                        dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                                        dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1]))/(2*dx2+2*dy2+2*dz2)
                                        
            # sucessive over relaxation  
                        self.phi[i,j,k] = self.phi[i,j,k] + 1.4*(phi_new - self.phi[i][j][k])
        
            """ 
            
            """
            #standard internal open node
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                            (self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                            dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1])+\
                            dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1])+\
                            dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk]))/(2*dx2+2*dy2+2*dz2)
            
            """
            """
            # sucessive over relaxation  
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] +\
                                                        1.8*(self.phi_new[1:self.ni-1,1:self.nj-1,1:self.nk-1] - \
                                                        self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1])
            """
            
            # Red point update
            self.phi[ir,jr,kr] = (1-w)*self.phi[ir,jr,kr] + (1.0/6.0)*w*(self.phi[ir+1,jr,kr] + self.phi[ir-1,jr,kr] +\
                                                                    self.phi[ir,jr+1,kr] + self.phi[ir,jr-1,kr] +\
                                                                    self.phi[ir,jr,kr+1] + self.phi[ir,jr,kr-1] +\
                                                                    (self.rho[ir,jr,kr]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            # Black point update
            self.phi[ib,jb,kb] = (1-w)*self.phi[ib,jb,kb] + (1.0/6.0)*w*(self.phi[ib+1,jb,kb] + self.phi[ib-1,jb,kb] +\
                                                                    self.phi[ib,jb+1,kb] + self.phi[ib,jb-1,kb] +\
                                                                    self.phi[ib,jb,kb+1] + self.phi[ib,jb,kb-1] +\
                                                                    (self.rho[ib,jb,kb]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            #check for convergence*/
            if it%25==0:
                sum = 0;
                """
                for i in np.arange(1,self.ni-1):
                    for j in np.arange(1,self.nj-1):
                        for k in np.arange(1,self.nk-1):
                """

                self.R[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                -self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1]*(2*dx2+2*dy2+2*dz2) +\
                self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1]) +\
                dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1]) +\
                dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk])

                sum = np.sum(self.R**2)

                L2 = np.sqrt(sum/(self.ni*self.nj*self.nk));
                #print("iter: "+str(it)+", L2 = "+str(L2))
                if (L2<tol):
                    converged = True
                    break

        if (converged==False):
            print("Gauss-Seidel failed to converge, L2 = "+str(L2))
        
        return converged

    def potentialSolver4(self, w, max_it, tol):
        """
        Compute the potential field.

        Parameters
        ----------
        max_it : int
            max iterations for Gauss-Seidel
        tol: float
            tolerance for Gauss-Seidel
        ----------
        
        """

        dx2 = 1.0/(self.dh[0]*self.dh[0]); # dx^2
        dy2 = 1.0/(self.dh[1]*self.dh[1]); # dy^2
        dz2 = 1.0/(self.dh[2]*self.dh[2]); # dz^2
    
        L2 = 0.0 # norm
        
        converged = False
        
        # Step 1: create *integer* array the same size as u 
        x = np.zeros_like(self.phi,dtype=np.int)

        # Step 2: populate all non-boundary cells with running numbers from 1 to (n-2)^2
        x[1:-1,1:-1,1:-1] = np.arange(1,(self.ni-2)*(self.nj-2)*(self.nk-2)+1).reshape(self.ni-2,self.nj-2,self.nk-2)

        # Step 3: get indices of even (red) and odd (black) points
        ir, jr, kr = np.where((x>0) & (x%2 == 0)) # indices of red pts = indices of even numbers
        ib, jb, kb = np.where((x>0) & (x%2 == 1)) # indices of black pts = indices of odd numbers


        
        # solve potential
        for it in np.arange(1,max_it+1):
            """
            for i in np.arange(1,self.ni-1):
                for j in np.arange(1,self.nj-1):
                    for k in np.arange(1,self.nk-1):
            """            
            """
            #standard internal open node
                        phi_new = (self.rho[i][j][k]/self.EPS_0 +\
                                        dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                                        dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                                        dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1]))/(2*dx2+2*dy2+2*dz2)
                                        
            # sucessive over relaxation  
                        self.phi[i,j,k] = self.phi[i,j,k] + 1.4*(phi_new - self.phi[i][j][k])
        
            """ 
            
            """
            #standard internal open node
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                            (self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                            dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1])+\
                            dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1])+\
                            dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk]))/(2*dx2+2*dy2+2*dz2)
            
            """
            """
            # sucessive over relaxation  
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] +\
                                                        1.8*(self.phi_new[1:self.ni-1,1:self.nj-1,1:self.nk-1] - \
                                                        self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1])
            """
            
            # Red point update
            self.phi[ir,jr,kr] = (1-w)*self.phi[ir,jr,kr] + (1.0/6.0)*w*(self.phi[ir+1,jr,kr] + self.phi[ir-1,jr,kr] +\
                                                                    self.phi[ir,jr+1,kr] + self.phi[ir,jr-1,kr] +\
                                                                    self.phi[ir,jr,kr+1] + self.phi[ir,jr,kr-1] +\
                                                                    (self.rho[ir,jr,kr]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            # Black point update
            self.phi[ib,jb,kb] = (1-w)*self.phi[ib,jb,kb] + (1.0/6.0)*w*(self.phi[ib+1,jb,kb] + self.phi[ib-1,jb,kb] +\
                                                                    self.phi[ib,jb+1,kb] + self.phi[ib,jb-1,kb] +\
                                                                    self.phi[ib,jb,kb+1] + self.phi[ib,jb,kb-1] +\
                                                                    (self.rho[ib,jb,kb]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            #check for convergence*/
            if it%25==0:
                sum = 0;
                """
                for i in np.arange(1,self.ni-1):
                    for j in np.arange(1,self.nj-1):
                        for k in np.arange(1,self.nk-1):
                """

                self.R[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                -self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1]*(2*dx2+2*dy2+2*dz2) +\
                self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1]) +\
                dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1]) +\
                dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk])

                sum = np.sum(self.R**2)

                L2 = np.sqrt(sum/(self.ni*self.nj*self.nk));
                #print("iter: "+str(it)+", L2 = "+str(L2))
                if (L2<tol):
                    converged = True
                    break

        if (converged==False):
            print("Gauss-Seidel failed to converge, L2 = "+str(L2))
        
        return converged

    @jit    
    def potentialSolver5(self, w, max_it, tol):
        """
        Compute the potential field.

        Parameters
        ----------
        max_it : int
            max iterations for Gauss-Seidel
        tol: float
            tolerance for Gauss-Seidel
        ----------
        
        """

        dx2 = 1.0/(self.dh[0]*self.dh[0]); # dx^2
        dy2 = 1.0/(self.dh[1]*self.dh[1]); # dy^2
        dz2 = 1.0/(self.dh[2]*self.dh[2]); # dz^2
    
        L2 = 0.0 # norm
        
        converged = False
        
        # Step 1: create *integer* array the same size as u 
        x = np.zeros_like(self.phi,dtype=np.int)

        # Step 2: populate all non-boundary cells with running numbers from 1 to (n-2)^2
        x[1:-1,1:-1,1:-1] = np.arange(1,(self.ni-2)*(self.nj-2)*(self.nk-2)+1).reshape(self.ni-2,self.nj-2,self.nk-2)

        # Step 3: get indices of even (red) and odd (black) points
        ir, jr, kr = np.where((x>0) & (x%2 == 0)) # indices of red pts = indices of even numbers
        ib, jb, kb = np.where((x>0) & (x%2 == 1)) # indices of black pts = indices of odd numbers


        
        # solve potential
        for it in np.arange(1,max_it+1):
            """
            for i in np.arange(1,self.ni-1):
                for j in np.arange(1,self.nj-1):
                    for k in np.arange(1,self.nk-1):
            """            
            """
            #standard internal open node
                        phi_new = (self.rho[i][j][k]/self.EPS_0 +\
                                        dx2*(self.phi[i-1][j][k] + self.phi[i+1][j][k]) +\
                                        dy2*(self.phi[i][j-1][k] + self.phi[i][j+1][k]) +\
                                        dz2*(self.phi[i][j][k-1] + self.phi[i][j][k+1]))/(2*dx2+2*dy2+2*dz2)
                                        
            # sucessive over relaxation  
                        self.phi[i,j,k] = self.phi[i,j,k] + 1.4*(phi_new - self.phi[i][j][k])
        
            """ 
            
            """
            #standard internal open node
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                            (self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                            dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1])+\
                            dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1])+\
                            dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk]))/(2*dx2+2*dy2+2*dz2)
            
            """
            """
            # sucessive over relaxation  
            self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] = self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1] +\
                                                        1.8*(self.phi_new[1:self.ni-1,1:self.nj-1,1:self.nk-1] - \
                                                        self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1])
            """
            
            # Red point update
            self.phi[ir,jr,kr] = (1-w)*self.phi[ir,jr,kr] + (1.0/6.0)*w*(self.phi[ir+1,jr,kr] + self.phi[ir-1,jr,kr] +\
                                                                    self.phi[ir,jr+1,kr] + self.phi[ir,jr-1,kr] +\
                                                                    self.phi[ir,jr,kr+1] + self.phi[ir,jr,kr-1] +\
                                                                    (self.rho[ir,jr,kr]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            # Black point update
            self.phi[ib,jb,kb] = (1-w)*self.phi[ib,jb,kb] + (1.0/6.0)*w*(self.phi[ib+1,jb,kb] + self.phi[ib-1,jb,kb] +\
                                                                    self.phi[ib,jb+1,kb] + self.phi[ib,jb-1,kb] +\
                                                                    self.phi[ib,jb,kb+1] + self.phi[ib,jb,kb-1] +\
                                                                    (self.rho[ib,jb,kb]/self.EPS_0)*(self.dh[0]*self.dh[1]))

            #check for convergence*/
            if it%25==0:
                sum = 0;
                """
                for i in np.arange(1,self.ni-1):
                    for j in np.arange(1,self.nj-1):
                        for k in np.arange(1,self.nk-1):
                """

                self.R[1:self.ni-1,1:self.nj-1,1:self.nk-1] = \
                -self.phi[1:self.ni-1,1:self.nj-1,1:self.nk-1]*(2*dx2+2*dy2+2*dz2) +\
                self.rho[1:self.ni-1,1:self.nj-1,1:self.nk-1]/self.EPS_0 +\
                dx2*(self.phi[0:self.ni-2,1:self.nj-1,1:self.nk-1] + self.phi[2:self.ni,1:self.nj-1,1:self.nk-1]) +\
                dy2*(self.phi[1:self.ni-1,0:self.nj-2,1:self.nk-1] + self.phi[1:self.ni-1,2:self.nj,1:self.nk-1]) +\
                dz2*(self.phi[1:self.ni-1,1:self.nj-1,0:self.nk-2] + self.phi[1:self.ni-1,1:self.nj-1,2:self.nk])

                sum = np.sum(self.R**2)

                L2 = np.sqrt(sum/(self.ni*self.nj*self.nk));
                #print("iter: "+str(it)+", L2 = "+str(L2))
                if (L2<tol):
                    converged = True
                    break

        if (converged==False):
            print("Gauss-Seidel failed to converge, L2 = "+str(L2))
        
        return converged

    
    def efSolver(self):
        """
        Compute the electric field from potential function.

        Parameters
        ----------
        
        ----------
        
        """
        dx = self.dh[0] # dx
        dy = self.dh[1] # dy
        dz = self.dh[2] # dz
        
        for i in np.arange(0, self.ni):
            for j in np.arange(0, self.nj):
                for k in np.arange(0, self.nk):

                    #x-component#
                    if i==0: 
                        # forward
                        self.ef[i][j][k][0] = -(-3*self.phi[i][j][k]+\
                                               4*self.phi[i+1][j][k]-\
                                               self.phi[i+2][j][k])/(2*dx)
                    elif i==self.ni-1:  
                        # backward
                        self.ef[i][j][k][0] = -(self.phi[i-2][j][k]-\
                                               4*self.phi[i-1][j][k]+\
                                               3*self.phi[i][j][k])/(2*dx)
                    else: 
                        #central
                        self.ef[i][j][k][0] = -(self.phi[i+1][j][k] - \
                                                self.phi[i-1][j][k])/(2*dx)

                    #y-component
                    if j==0:
                        self.ef[i][j][k][1] = -(-3*self.phi[i][j][k] + \
                                                4*self.phi[i][j+1][k]-\
                                                self.phi[i][j+2][k])/(2*dy)
                    elif j==self.nj-1:
                        self.ef[i][j][k][1] = -(self.phi[i][j-2][k] - \
                                                4*self.phi[i][j-1][k] +\
                                                3*self.phi[i][j][k])/(2*dy)
                    else:
                         self.ef[i][j][k][1] = -(self.phi[i][j+1][k] - \
                                                 self.phi[i][j-1][k])/(2*dy)

                    #z-component
                    if k==0:
                        self.ef[i][j][k][2] = -(-3*self.phi[i][j][k] + \
                                                4*self.phi[i][j][k+1]-
                                                self.phi[i][j][k+2])/(2*dz)
                    elif k==self.nk-1:
                        self.ef[i][j][k][2] = -(self.phi[i][j][k-2] - \
                                                4*self.phi[i][j][k-1]    + \
                                                3*self.phi[i][j][k])/(2*dz)
                    else:
                        self.ef[i][j][k][2] = -(self.phi[i][j][k+1] - \
                                                self.phi[i][j][k-1])/(2*dz)
    @jit
    def efSolver2(self):
        """
        Compute the electric field from potential function.

        Parameters
        ----------
        
        ----------
        
        """
        dx = self.dh[0] # dx
        dy = self.dh[1] # dy
        dz = self.dh[2] # dz
        
        """
        for i in np.arange(0, self.ni):
            for j in np.arange(0, self.nj):
                for k in np.arange(0, self.nk):
        """

        ##x-component#
        #if i==0: 
        #x-component#
        """
                    if i==0: 
                        # forward
                        self.ef[i][j][k][0] = -(-3*self.phi[i][j][k]+\
                                               4*self.phi[i+1][j][k]-\
                                               self.phi[i+2][j][k])/(2*dx)
        """
                    
        # forward
        self.ef[0,0:self.nj,0:self.nk,0] = -(-3*self.phi[0,0:self.nj,0:self.nk]+\
                               4*self.phi[1,0:self.nj,0:self.nk]-\
                               self.phi[2,0:self.nj,0:self.nk])/(2*dx)
        
        #elif i==self.ni-1:  
        """
        elif i==self.ni-1:  
                        # backward
                        self.ef[i][j][k][0] = -(self.phi[i-2][j][k]-\
                                               4*self.phi[i-1][j][k]+\
                                               3*self.phi[i][j][k])/(2*dx)
        """           
        # backward
        self.ef[self.ni-1,0:self.nj,0:self.nk,0] = -(self.phi[self.ni-3,0:self.nj,0:self.nk]-\
                                   4*self.phi[self.ni-2,0:self.nj,0:self.nk]+\
                                   3*self.phi[self.ni-1,0:self.nj,0:self.nk])/(2*dx)
        """
        else: 
            #central
            self.ef[i][j][k][0] = -(self.phi[i+1][j][k] - \
                                    self.phi[i-1][j][k])/(2*dx)
        """ 
        #central
        self.ef[1:self.ni-1,0:self.nj,0:self.nk,0] = -(self.phi[2:self.ni,0:self.nj,0:self.nk] - \
                                self.phi[0:self.ni-2,0:self.nj,0:self.nk])/(2*dx)


        #y-component
        #if j==0:
        """
        if j==0:
                        self.ef[i][j][k][1] = -(-3*self.phi[i][j][k] + \
                                                4*self.phi[i][j+1][k]-\
                                                self.phi[i][j+2][k])/(2*dy)
                    
        """
        self.ef[0:self.ni,0,0:self.nk,1] = -(-3*self.phi[0:self.ni,0,0:self.nk] + \
                                    4*self.phi[0:self.ni,1,0:self.nk]-\
                                    self.phi[0:self.ni,2,0:self.nk])/(2*dy)
        #elif j==self.nj-1:
        """
        elif j==self.nj-1:
                        self.ef[i][j][k][1] = -(self.phi[i][j-2][k] - \
                                                4*self.phi[i][j-1][k] +\
                                                3*self.phi[i][j][k])/(2*dy)
                    
        """
        self.ef[0:self.ni,self.nj-1,0:self.nk,1] = -(self.phi[0:self.ni,self.nj-3,0:self.nk] - \
                                    4*self.phi[0:self.ni,self.nj-2,0:self.nk] +\
                                    3*self.phi[0:self.ni,self.nj-1,0:self.nk])/(2*dy)
        #else:
        """
        else:
                         self.ef[i][j][k][1] = -(self.phi[i][j+1][k] - \
                                                 self.phi[i][j-1][k])/(2*dy)

        """
        self.ef[0:self.ni,1:self.nj-1,0:self.nk,1] = -(self.phi[0:self.ni,2:self.nj,0:self.nk] - \
                                     self.phi[0:self.ni,0:self.nj-2,0:self.nk])/(2*dy)

        #z-component
        '''
        if k==0:
            self.ef[i][j][k][2] = -(-3*self.phi[i][j][k] + \
                                    4*self.phi[i][j][k+1]-
                                    self.phi[i][j][k+2])/(2*dz)
            
        '''
        #z-component
        #if k==0:
        self.ef[0:self.ni,0:self.nj,0,2] = -(-3*self.phi[0:self.ni,0:self.nj,0] + \
                                4*self.phi[0:self.ni,0:self.nj,1]-
                                self.phi[0:self.ni,0:self.nj,2])/(2*dz)

        """
        elif k==self.nk-1:
            self.ef[i][j][k][2] = -(self.phi[i][j][k-2] - \
                                    4*self.phi[i][j][k-1]    + \
                                    3*self.phi[i][j][k])/(2*dz)
        """
        
        #elif k==self.nk-1:
        self.ef[0:self.ni,0:self.nj,self.nk-1,2] = -(self.phi[0:self.ni,0:self.nj,self.nk-3] - \
                                    4*self.phi[0:self.ni,0:self.nj,self.nk-2]    + \
                                    3*self.phi[0:self.ni,0:self.nj,self.nk-1])/(2*dz) 
        """
        else:
            self.ef[i][j][k][2] = -(self.phi[i][j][k+1] - \
                                    self.phi[i][j][k-1])/(2*dz)
        """
        #else:
        self.ef[0:self.ni,0:self.nj,1:self.nk-1,2] = -(self.phi[0:self.ni,0:self.nj,2:self.nk] - \
                                    self.phi[0:self.ni,0:self.nj,0:self.nk-2])/(2*dz)
        
        
    def computeNodeVolumes(self): 
        """
        Compute the node volumes.
        Parameters
        ----------
        
        ----------
        
        """
        for i in np.arange(0,self.ni):
            for j in np.arange(0,self.nj):
                for k in np.arange(0,self.nk):
        
                    V = self.dh[0]*self.dh[1]*self.dh[2]
                    if (i==0 or i==self.ni-1): V*=0.5
                    if (j==0 or j==self.nj-1): V*=0.5
                    if (k==0 or k==self.nk-1): V*=0.5
                    
                    self.node_vol[i][j][k] = V
                    
    def XtoL(self, x):
        """
        Determine which cell a particle at position vector 
        x belongs to.
        
        Parameters
        ----------
        x : numpy.ndarray
            position vector 
        ----------
        """
        lc = np.zeros(3)
        
        lc[0] = (x[0]-self.x0[0])/self.dh[0];
        lc[1] = (x[1]-self.x0[1])/self.dh[1];
        lc[2] = (x[2]-self.x0[2])/self.dh[2];
        
        return lc
    
    def addSpeciesList(self, speciesList):
        self.speciesList = speciesList
    
    def computeChargeDensity(self):
        """
        Compute the charge density.
        
        Parameters
        ----------
     
        ----------
        """
        
        self.rho = np.zeros((self.ni, self.nj, self.nk))
        
        for species in self.speciesList:
            if species.charge!=0:
                self.rho += species.charge*species.den         