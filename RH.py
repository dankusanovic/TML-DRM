def Assemble(a,b,k):
    """
    This function appends b to the end of a with k elements overlapped.
    If k<0, abs(k) zeros will be added to the end of a before appending b.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    a, b  : ndarray
        ndarray with shape (1,*)
    k  : int
        Number of overlapped elements, k<0 means abs(k) zeros are added to the end of a

    Returns
    -------
    c  : ndarray
        The concatenated ndarray with shape (1,a.size+b.size)
    """
    if a.size==0:
        c = b
    else:
        la = a.size
        lb = b.size
        if k<0:
            c = np.concatenate((a,np.zeros((1,-k)),b),1)
        elif k==0:
            c = np.concatenate((a,b),1)
        else:
            c = np.concatenate((a[:,0:la-k],a[:,la-k:la] + b[:,0:k],b[:,k:lb]),1)
    
    return c

def getAdmissibleEigValEigVec(Ain,Min,w,Vupper,Vlower,mode):
    """
    This function calculates the eigenvalues (Rayleigh wave velocities) and eigenvectors (mode shapes)
    of the generalized eigenvalue problem, and pick the admissible solutions. The admissible velocities
    of propagating Rayleigh modes are real values and must be in the range [Vlower, Vupper].\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    Ain, Min : ndarray
        Coefficient matrices for the generalized eigenvalue problem: A * x[i] = w[i] * M * x[i].
    w : float
        Angular frequency.
    Vupper, Vlower : float
        The upper and lower bound of Rayleigh wave velocity.
    mode : int
        The desired mode. 0: fundamental, 1: 1st higher mode

    Returns
    -------
    eigVel, eigShape : ndarray
        Velocities and mode shapes of Rayleigh natural propagating modes.

    """
    noModes = mode + 1
    m = noModes
    unfinished = True
    
    while unfinished:
        eigk, eigShape = sla.eigs(Ain, k=m, M = Min, sigma=w/Vlower, which = 'LM') #note that k is the number of modes desired
        eigv = w/eigk
        vreal = np.real(eigv[np.imag(eigv)==0]) 
        vrealAdmiss = vreal[(Vlower<=vreal) & (vreal<=Vupper)]
        
        nv = np.size(vrealAdmiss)
        if nv==0:
            #no real velocity in admissible range, increase number of modes desired
            m *= 2 
        elif nv>=noModes:
            #enough number of real admissible velocity needed, stop
            unfinished = False
        else:
            #not enough real admissible velocity needed
            if max(vreal)>Vupper:
                #other velocities are outside admissible range, useless to increase number of modes sought, stop 
                unfinished = False
            else:
                #there are potentials velocities in the admissible range, increase number of modes desired
                m *=2
                    
    indices = np.where(np.in1d(eigv, vrealAdmiss))[0]
    if nv>=noModes:
        eigVel = vrealAdmiss[0:noModes]
        eigShape = eigShape[:,indices]
        modeShape = eigShape[:,0:noModes]
    else:
        eigNaN = np.empty(noModes-nv)
        eigNaN.fill(np.nan)
        eigVel = np.concatenate((vrealAdmiss,eigNaN))
        
        eigNaN = np.empty((eigShape.shape[0],noModes-nv))
        eigNaN.fill(np.nan)
        modeShape = np.concatenate((eigShape[:,indices],eigNaN),1)
        
    return eigVel, modeShape  

def GetLayerStiffnessComponents(ns,h,Lame1,mu,rho):
    """
    This function calculates the diagonal components of the stiffness matrices
    A, B, G, M for 1 soil layer with multiple soil sublayers in Thin Layer Method.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    ns  : int
        Number of soil sublayers in the currently-considered soil layer
    h  : float
        Uniform thickness of the sublayer
    Lame1, mu, rho  : float
        The first Lame parameter, shear modulus, and mass density of the soil layer

    Returns
    -------
    A0, A2, B1, B3, G0, G2, M0, M2  : ndarray
        The diagonal components of stiffness matrices A, B, G, M 
        0, 1, and 2 mean main, 1st, and 2nd diagonal 
    """
    
    coef1 = Lame1 + 2.0*mu
    coef2 = Lame1 + mu
    coef3 = Lame1 - mu
    
    A0 = h/6.0*np.concatenate(([[2.0*coef1,2.0*mu]],4.0*np.tile([[coef1,mu]],(1,ns-1)),[[2.0*coef1,2.0*mu]]),1)
    A2 = h/6.0*np.tile([[coef1,mu]],(1,ns))
    
    B1 = 1.0/2.0*np.concatenate(([[coef3,coef2]],np.tile([[0.0,coef2]],(1,ns-1)),[[-coef3]]),1)
    B3 = 1.0/2.0*np.concatenate((np.tile([[-coef2,0.0]],(1,ns-1)),[[-coef2]]),1)
    
    G0 = 1.0/h*np.concatenate(([[mu,coef1]],2.0*np.tile([[mu,coef1]],(1,ns-1)),[[mu,coef1]]),1)
    G2 = 1.0/h*np.tile([[-mu,-coef1]],(1,ns))

    M0 = rho*h/6.0*np.concatenate(([[2.0,2.0]],4.0*np.ones((1,2*ns-2)),[[2.0,2.0]]),1)
    M2 = rho*h/6.0*np.ones((1,2*ns))
    
    return A0, A2, B1, B3, G0, G2, M0, M2

def GetRayleighVelocity(Vs, nu):
    """
    This function calculates the Rayleigh wave phase velocity in homogeneous
    elastic domain.
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    Vs  : float
        Shear wave velocity of the homogeneous soil domain 
    nu  : float
        The Poisson's ratio of soil
        
    Returns
    -------
    ratio*Vs  : float
        Rayleigh wave phase velocity
    """
    #Compute the Rayleigh wave velocity.
    lambda0 = 2.0*1.0*nu/(1.0 - 2.0*nu)
    hk = np.sqrt(1.0/(lambda0 + 2.0*1.0))
    
    p = [-16.0*(1.0 - hk**2), 24.0 - 16.0*hk**2, -8.0, 1.0]
    x = np.sort(np.roots(p))
    ratio = 1.0/np.sqrt(np.real(x[2]))
    
    return ratio*Vs

def GetRayleighDispersionAndModeShape(mode, intY, beta, rho, nu, dy1, yDRMmin, nepw, startFre, endFre, df, depthFactor):
    """
    This function calculates the dispersion curves and mode shapes of Rayleigh
    wave in stratified soil.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    mode  : int
        The chosen Rayleigh mode shape. Currently, only mode=0 for fundamental mode is available
    intY  : array
        The y-coordinate of soil layer interfaces, from free ground surface downwards 
    beta, rho, nu  : array
        Shear wave velocity, Mass density, and Poisson's ratio of soil layers, from top layer to half space 
    dy1  : float
        y-grid spacing of points used to interpolate the mode shape, from ground surface to yDRMmin
    yDRMmin  : float
        Minimum of y-coordinates among DRM nodes
    nepw  : int
        Number of element per wavelength in the extended region below yDRMmin
    startFre, endFre, df  : float
        Start, end, and increment of frequency
    depthFactor  :float
        Soil domain is extended up to depthFactor*wavelength to mimic the clamp condition
        
    Returns
    -------
    fDispersion, phaseVelDispersion  : array
        The frequency and phase velocity of the dispersion curve of chosen Rayleigh mode
    yGridModeShape  : array
        The y-coordinate grid points which is subsequently used to interpolate mode shape at DRM nodes
    uModeShape, vModeShape  : ndarray
        The mode shapes of horizontal and vertical displacements of chosen Rayleigh mode at yGridModeShape, 
        at each frequency fDispersion
    """
    Vr = np.zeros(len(intY))
    for jj in range(len(intY)):
        Vr[jj] = GetRayleighVelocity(beta[jj],nu[jj])
    
    Vlower = Vr.min()
    Vupper = beta[-1]
    
    fDispersion = np.arange(startFre, endFre, df)
    phaseVelDispersion = np.zeros(len(fDispersion))
    modeShape = []
    
    for indexFre, f in enumerate(fDispersion):
        A0 = np.array([[]])
        A2 = np.array([[]])
        B1 = np.array([[]])
        B3 = np.array([[]])
        G0 = np.array([[]])
        G2 = np.array([[]])
        M0 = np.array([[]])
        M2 = np.array([[]])
        
        #Calculate wavelength and element size in the extended region
        wavelengthMax = Vupper/f
        wavelengthMin = Vlower/f
        dy2 = wavelengthMin/nepw
        
        #Add imaginary layer at yDRMmin and at depth depthFactor*wavelength
        depth = min(intY[-1],yDRMmin,intY[0]-depthFactor*wavelengthMax)
        intAux = np.unique(np.concatenate([intY,[yDRMmin,depth]]))
        intYNew = intAux[::-1]

        indices = np.where(np.in1d(intYNew, intY))[0]
        repeats = np.hstack((np.diff(indices),[np.size(intYNew)-indices[-1]]))
        betaNew = np.repeat(beta,repeats)
        rhoNew = np.repeat(rho,repeats)
        nuNew = np.repeat(nu,repeats)
    
        muNew = rhoNew*betaNew*betaNew
        Lame1New = 2.0*muNew*nuNew/(1.0-2.0*nuNew)
        
        N = len(intYNew) #number of interfaces (including imaginary interface at yDRMmin and at depthFactor*wavelength)
        h = -np.diff(intYNew) #thickness of each soil layer
        
        #Mesh density in 2 region, from 0 to yDRMmin and yDRMmin to depthFactor*wavelength
        idx = np.where(intYNew == yDRMmin)[0][0]
        dy = np.concatenate([dy1*np.ones(idx),dy2*np.ones(N-idx-1)])
        ns = np.ceil(h/dy).astype(int) #number of sublayer in each soil layer
        dy = h/ns
        
        for ii in range(N-1):
            [A0e, A2e, B1e, B3e, G0e, G2e, M0e, M2e] = GetLayerStiffnessComponents(ns[ii],dy[ii],Lame1New[ii],muNew[ii],rhoNew[ii])     
            A0 = Assemble(A0,A0e,2)
            A2 = Assemble(A2,A2e,0)
            B1 = Assemble(B1,B1e,1)
            B3 = Assemble(B3,B3e,-1)
            G0 = Assemble(G0,G0e,2)
            G2 = Assemble(G2,G2e,0)
            M0 = Assemble(M0,M0e,2)
            M2 = Assemble(M2,M2e,0)
            
        #Add last 2x2 block for last interface
        A0 = Assemble(A0,A0e[0:1,0:2],2);
        B1 = Assemble(B1,B1e[0:1,0:1],1);
        G0 = Assemble(G0,G0e[0:1,0:2],2);
        M0 = Assemble(M0,M0e[0:1,0:2],2);
    
        Ns = np.sum(ns) #total number of sub layers
        noDoF = 2*Ns+2 #total number of degree of freedom
        A = sps.spdiags(np.vstack((np.concatenate((A2,[[0.0,0.0]]),1),A0,np.concatenate(([[0.0,0.0]],A2),1))),np.array([-2,0,2]),noDoF,noDoF)
        B = sps.spdiags(np.vstack((np.concatenate((B3,[[0.0,0.0,0.0]]),1),np.concatenate((B1,[[0.0]]),1),np.concatenate(([[0.0]],B1),1),np.concatenate(([[0.0,0.0,0.0]],B3),1))),np.array([-3,-1,1,3]),noDoF,noDoF)
        G = sps.spdiags(np.vstack((np.concatenate((G2,[[0.0,0.0]]),1),G0,np.concatenate(([[0.0,0.0]],G2),1))),np.array([-2,0,2]),noDoF,noDoF)
        M = sps.spdiags(np.vstack((np.concatenate((M2,[[0.0,0.0]]),1),M0,np.concatenate(([[0.0,0.0]],M2),1))),np.array([-2,0,2]),noDoF,noDoF)
        
        #Solve for eigenvalues and eigenvectors
        Ngrid = 2*(np.sum(ns[0:idx])+1) #number of degrees of freedom of the predefined grid used for DRM nodes interpolation
        w = 2.0*np.pi*f
        
        Ain = sps.vstack([sps.hstack([sps.csr_matrix((noDoF, noDoF), dtype = np.float64), sps.identity(noDoF,dtype=np.float64)]), sps.hstack([w*w*M-G, -B])])
        Min = sps.vstack([sps.hstack([sps.identity(noDoF,dtype=np.float64),sps.csr_matrix((noDoF, noDoF), dtype = np.float64)]), sps.hstack([sps.csr_matrix((noDoF, noDoF), dtype = np.float64),A])])
        
        #Avoids mode kissing for Rayleigh wave mode with extremly high layer contrast
        eigVel, eigShape = getAdmissibleEigValEigVec(Ain,Min,w,Vupper,Vlower,mode)
        phaseVelDispersion[indexFre] = eigVel[mode]
        modeShape.append(eigShape[0:Ngrid,mode])
    
    modeShapeReshape = np.transpose(np.vstack(modeShape)) #reshape the matrix, column is indexFre, row is interlacing of u and v of each point     
    uModeShape = modeShapeReshape[0::2,:]
    vModeShape = modeShapeReshape[1::2,:]
    
    yGridModeShape = [intYNew[0]]
    for nn in range(idx):
        yGridModeShape.append(np.linspace(intYNew[nn]-dy[nn],intYNew[nn+1],ns[nn]))
    yGridModeShape = np.hstack(yGridModeShape)
    
    return fDispersion, phaseVelDispersion, yGridModeShape, uModeShape, vModeShape

def GetRayleighFFTfields(Disp, endFrequency, dt, df, nt):
    '''
    '''
    stepsNeeded    = max(nt, 1.0/dt/df)
    nextPowerOfTwo = int(np.ceil(np.log2(stepsNeeded)))
    Nt = 2**nextPowerOfTwo

    Disp  = np.concatenate([Disp, np.zeros(Nt-nt)])

    n       = np.arange(1, Nt/2+1)
    wVec    = 2.0*np.pi/dt/Nt*n
    wVec    = wVec[wVec <= 2.0*np.pi*endFrequency]

    FFTdisp = np.fft.rfft(Disp)
    FFTdisp = FFTdisp[1:len(wVec)+1]

    df = 1.0/dt/Nt
    startFrequency = df
    endFrequency  += 0.001*df #to include the endFre if there is a multiple of df, fDispersion = np.arange(startFre,endFre,df)

    return wVec, FFTdisp, df, Nt, startFrequency, endFrequency


def RHbackground2Dfield(FdispIn,wVec,interpDispersion,interpuMmodeShape,interpvMmodeShape,x,y,Nt,x0,y0):
    """
    This function calculates the displacements at a specific point in time domain for Rayleigh
    wave in stratified soil.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    wVec  : array
        The angular frequency
    interpDispersion  : object of class scipy.interpolate.interp1d
        The 1d interpolation function to interpolate the phase velocity at chosen angular frequency 
    interpuMmodeShape, interpvMmodeShape  : objects of class scipy.interpolate.RectBivariateSpline
        The 2d interpolation function to interpolate the horizontal and vertical displacement mode shape
        at the query point. Functions of y and angular frequency 
    x, y  : float
        x- and y-coordinate of the query point
    Nt  : int
        Length of the Disp, Vels, Accel after zero padding
    FdispIn  : array
        FFT of the input displacement
    x0, y0  : float
        x- and y-coordinate of the reference point (where the incoming signal time series is prescribed)
        
    Returns
    -------
    U, V  : array
        The horizontal and vertical displacements at the query point
    """
    U_fft = np.zeros(int(Nt/2 + 1), dtype=complex) ###############
    V_fft = np.zeros(int(Nt/2 + 1), dtype=complex) ###############
    
    for indexFre, w in enumerate(wVec):
        vr = interpDispersion(w)
        k = w/vr
        uRef = interpuMmodeShape(y0,w)
        Ufi = interpuMmodeShape(y,w)/uRef*FdispIn[indexFre]
        Vfi = interpvMmodeShape(y,w)/uRef*FdispIn[indexFre]
        
        U_fft[indexFre+1] = Ufi[0,0]*np.exp(-1j*k*(x-x0)) #since zero frequency is not accounted for, so it start from index [1]
        V_fft[indexFre+1] = -1j*Vfi[0,0]*np.exp(-1j*k*(x-x0))
        
    U = np.real(np.fft.irfft(U_fft, Nt, axis=0))
    V = np.real(np.fft.irfft(V_fft, Nt, axis=0))
    
    return U, V


import numpy as np
from scipy import interpolate
import scipy.sparse as sps
from scipy.sparse import linalg as sla
import matplotlib.pyplot as plt

nx = 250
ny = 250

#SOIL PARAMETERS
layers = [0,-10, -30]
beta   = [10.0, 20.0, 40.0]
rho    = [2000.0,2000.0,2000.0]
nu     = [0.35,0.35,0.35]

#GRID MESH
xmin, xmax = -25.0,  25.0
ymin, ymax = -50.0,   0.0

#RICKER PULSE PARAMETERS
f   = 0.50
to  = 1.00 

endFrequency = 10.0
depthFactor = 3.0
dy1  = np.min(np.array(beta)/endFrequency/16.0)
nepw = 40
mode = 0
df   = 0.1

#TIME-SPACE GRID
time = np.linspace(0.0, 20.0, 200)
Y, X = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(xmin, xmax, nx))
nt = len(time)
dt = 20.0/200.0

#RICKER PULSE 
factor = np.square(np.pi*f*(time - to))
Disp = np.multiply((1 - 2.0*factor), np.exp(-factor))

wVec, FFTdisp, df, Nt, startFrequency, endFrequency = GetRayleighFFTfields(Disp, endFrequency, dt, df, nt)

fDispersion, phaseVelDispersion, yGridModeShape, uModeShape, vModeShape = GetRayleighDispersionAndModeShape(mode, layers, beta, rho, nu, dy1, ymin, nepw, startFrequency, endFrequency, df, depthFactor)

yGridModeShape = np.flipud(yGridModeShape)
uModeShape = np.flipud(np.real(uModeShape))
vModeShape = np.flipud(np.real(vModeShape))

interpDispersion = interpolate.interp1d(2.0*np.pi*fDispersion, phaseVelDispersion,kind='linear', fill_value='extrapolate')
interpuMmodeShape = interpolate.RectBivariateSpline(yGridModeShape,2.0*np.pi*fDispersion, uModeShape)
interpvMmodeShape = interpolate.RectBivariateSpline(yGridModeShape,2.0*np.pi*fDispersion, vModeShape)

#PLOT THE WAVE PROPAGATION
fig, ax = plt.subplots()

U = np.zeros([nx,ny,Nt])
V = np.zeros([nx,ny,Nt])
for j in range(ny):
    for i in range(nx):
        U[i,j,:], V[i,j,:] = RHbackground2Dfield(FFTdisp, wVec, interpDispersion, interpuMmodeShape, interpvMmodeShape, X[i,j], Y[i,j], Nt, xmin, ymax)

for k in range(150):
    Z = np.sqrt(U[:,:,k]**2 + V[:,:,k]**2)
    ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=0.0, vmax=1.0)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.axis('equal')
    plt.draw()
    plt.pause(0.1)
    plt.cla()
