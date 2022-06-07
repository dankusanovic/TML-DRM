def GetKofFullSpace(k,p,s,mu):
    """
    This function calculates the stiffness matrix of full space.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    k  : float
        The horizontal wavenumber (along x-axis)
    p, s  : complex
        Complex coefficients
    mu  : float
        The shear modulus of soil

    Returns
    -------
    K  : array
        2x2 matrix, stiffness matrix of full space
    """
    K = np.zeros((2,2), dtype=complex)
    coef = 2.0*k*mu*(1.0-s*s)/(1.0-p*s)
    K[0,0] = coef*p
    K[1,1] = coef*s
    
    return K
    
def GetKofHalfSpace(k,p,s,mu):
    """
    This function calculates the stiffness matrix of half space.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    k  : float
        The horizontal wavenumber (along x-axis)
   p, s  : complex
        Complex coefficients
    mu  : float
        The shear modulus of soil

    Returns
    -------
    K  : array
        2x2 matrix, stiffness matrix of half space
    """
   
    K = np.zeros((2,2), dtype=complex)
    coef = (1.0-s*s)/2.0/(1.0-p*s)
    K[0,0] = coef*p
    K[0,1] = -coef + 1.0
    K[1,0] = K[0,1]
    K[1,1] = coef*s
    K *= 2.0*k*mu
    
    return K
    
def GetKofLayer(k,p,s,h,mu,aSP):
    """
    This function calculates the K00 and K01 components of the stiffness matrix
    of a layer with finite thickness.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    k  : float
        The horizontal wavenumber (along x-axis)
    p, s  : complex
        Complex coefficients
    h  : float
        The thickness of soil layer
    mu  : float
        The shear modulus of soil
    aSP  : float
        The ratio between shear wave velocity and dilatational wave velocity

    Returns
    -------
    K00  : array
        2x2 matrix, block component of stiffness matrix of a layer 
    K01  : array
        2x2 matrix, block component of stiffness matrix of a layer 
    """
    K00 = np.zeros((2,2), dtype=complex)
    K01 = np.zeros((2,2), dtype=complex)
    
    a = np.real(k*p*h)
    b = np.imag(k*p*h)
    c = np.real(k*s*h)
    d = np.imag(k*s*h)
    
    cb = np.cos(b)
    sb = np.sin(b)
    
    plusA  = 1.0/2.0*(1.0+np.exp(-2.0*a))
    minusA = 1.0/2.0*(1.0-np.exp(-2.0*a))
    
    C1 = plusA*cb + 1j*minusA*sb
    S1 = minusA*cb + 1j*plusA*sb
    
    cd = np.cos(d)
    sd = np.sin(d)
    
    plusC  = 1.0/2.0*(1.0+np.exp(-2.0*c))
    minusC = 1.0/2.0*(1.0-np.exp(-2.0*c))
    
    C2 = plusC*cd + 1j*minusC*sd
    S2 = minusC*cd + 1j*plusC*sd
    
    D0 = 2.0*(np.exp(-a-c)-C1*C2)+(1.0/p/s+p*s)*S1*S2
    
    K00[0,0] = (1.0-s*s)/2.0/s*(C1*S2-p*s*C2*S1)/D0
    K00[0,1] = (1.0-s*s)/2.0*(np.exp(-a-c)-C1*C2+p*s*S1*S2)/D0 + (1.0+s*s)/2.0
    K00[1,0] = K00[0,1]
    K00[1,1] = (1.0-s*s)/2.0/p*(C2*S1-p*s*C1*S2)/D0
    K00 *= 2.0*k*mu
    
    K01[0,0] = 1.0/s*(p*s*S1*np.exp(-c) - S2*np.exp(-a))/D0
    K01[0,1] = (C1*np.exp(-c) - C2*np.exp(-a))/D0
    K01[1,0] = -K01[0,1]
    K01[1,1] = 1.0/p*(p*s*S2*np.exp(-a) - S1*np.exp(-c))/D0
    K01 *= 2.0*k*mu*(1.0-s*s)/2.0
    
    return K00, K01
    
def PSVbackground2Dfield(us, Layers, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N, Nt, x0, x, y):
    """
    This function calculates the displacement time series in 2D wave propagation
    problem, in which the SV or P wave incoming from the half space
    underneath under arbitrary incident angle from 0 to 90 degrees and propagating 
    through stratified soil domain. 
    Note: 
    [1] The coordinate system and displacement positive axes:

        y(V) ^
             |
             |
             o-----> x(U)
             
    [2] At each frequency, the horizontal and vertical displacements are calculated 
        based on Eduardo Kausel's Stiffness Matrix Method, in "Fundamental 
        Solutions in Elastodynamics, A Compendium", chap. 10, pp. 140--159 

    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    us  : array
        Displacement at the layer interfaces in frequency domain
    Layers  : array
        The y-coordinate of soil layer interfaces, including imaginary and half-space interfaces
    wVec  : array
        Angular frequency spectrum 
    p, s  : array
        Complex coefficients
    h  : array
        The thickness of soil layers
    mu  : array
        The shear modulus of soil layers
    aSP  : array
        The ratio between shear wave velocity and dilatational wave velocity of soil layers
    phaseVelIn  : float
        The phase velocity of incoming wave in the half space underneath
    sinTheta  : float
        Sine of the incoming angle
    N  : int
        Number of soil layers, including imaginary layer and half space
    Nt  : int
        Length of the Disp, Vels, Accel after zero padding
    x0  :float
        x-coordinate of the reference point (where the incoming signal time series is prescribed)
    x, y :float
        x- and y-coordinate of the query point
        
    Returns
    -------
    Z  : array
        Displacement time series at the query point
    """
    nfi = len(wVec)
    U_fft = np.zeros(nfi, dtype=complex)
    V_fft = np.zeros(nfi, dtype=complex) 

    for fi in range(nfi):
        w = wVec[fi]
        k = w*sinTheta/phaseVelIn
   
        #Displacement at interface
        uInterface = us[:,:, fi]

        #find parent layer where yTopLayer>=y>yBotLayer
        parentLayer = N - 1 - np.searchsorted(Layers[::-1], y, side = "left") 
            
        if parentLayer == (N-1):
            if y==Layers[-1]:
                uz = uInterface[2*N-2:2*N,:]
        elif parentLayer < (N-1):
            yTop = Layers[parentLayer]
            yBot = Layers[parentLayer+1]
            uTop = uInterface[2*parentLayer:2*parentLayer+2,:]
            uBot = uInterface[2*parentLayer+2:2*parentLayer+4,:]
            if y < yTop:
                uz = GetDisplacementAtInteriorLayer(y,yTop,yBot,uTop,uBot,k,p[parentLayer],s[parentLayer],mu[parentLayer],aSP[parentLayer])
            elif np.isclose(y, yTop, rtol=1e-05):
                uz = uTop
                    
        U_fft[fi] = uz[0,0]*np.exp(-1j*k*(x-x0))
        V_fft[fi] = 1j*uz[1,0]*np.exp(-1j*k*(x-x0))

    #Add 0 for zero frequency and frequency larger than cutOffFrequency
    U_fft = np.concatenate((np.zeros(1, dtype=complex), U_fft, np.zeros(int(Nt/2)-nfi, dtype=complex)), axis=0)
    V_fft = np.concatenate((np.zeros(1, dtype=complex), V_fft, np.zeros(int(Nt/2)-nfi, dtype=complex)), axis=0)

    U = np.real(np.fft.irfft(U_fft, Nt, axis=0))
    V = np.real(np.fft.irfft(V_fft, Nt, axis=0))

    return U, V
    
def DataPreprocessing(Disp, Layers, beta, rho, nu, angle, yDRMmin, nt, dt):
    """
    This function performs the data pre-processing for the wave propagation
    problem, such as adding an imaginary layer at the bottom of DRM nodes 
    (if necessary), zero padding, transform wave signal into frequency domain.
    Particularly, it also calculates the motion at the position of half-space surface, 
    assuming that wave is propagating in the full space having same properties 
    as the half space. This full-space motion is subsequently used to 
    generate the force vector while calculating the response of soil interface 
    based on substructure technique.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    Disp, Vels, Accel  : array
        Displacement, Velocity, and Acceleration time series of incoming wave
    Layers  : array
        The y-coordinate of soil layer interfaces, from free ground surface to half-space surface 
    beta, rho, nu  : array
        Shear wave velocity, Mass density, and Poisson's ratio of soil layers, from top layer to half space 
    angle  : float
        Angle of incoming wave, with respect to vertical axis (y-axis), from 0 to 90 degrees
    yDRMmin  : float
        The minimum of y-coordinates among DRM nodes
    nt  : int
        The original number of time step of the wave signal
    dt  : float
        The time step increment 
    fun  : dict
        The dictionary that contains the information of DRM
        
    Returns
    -------
    ufull, vfull, afull  : array
        Displacement, Velocity, and Acceleration at the position of half-space surface
        Note that the vertical components are multiplied with -1j
    Layers, beta, rho, nu: array
        Similar as above explaination
    wVec  : array
        Angular frequency spectrum 
    p, s  : array
        Complex coefficients
    h  : array
        The thickness of soil layers
    mu  : array
        The shear modulus of soil layers
    aSP  : array
        The ratio between shear wave velocity and dilatational wave velocity of soil layers
    phaseVelIn  : float
        The phase velocity of incoming wave in the half space underneath
    sinTheta  : float
        Sine of the incoming angle
    N  : int
        Number of soil layers, including imaginary layer and half space
    Nt  : int
        Length of the Disp, Vels, Accel after zero padding
    """
    CutOffFrequency = 10.0
    df = 0.2

    #ADDING IMAGINARY LAYER AT yDRMmin IF NECESSARY
    if yDRMmin < Layers[-1]:
        Layers = np.append(Layers, np.array([yDRMmin]))
        beta = np.append(beta, [beta[-1]])
        rho = np.append(rho, [rho[-1]])
        nu = np.append(nu, [nu[-1]])

    N = len(Layers)

    #PADDING ZERO TO HAVE DESIRED DISCRETIZED FREQUENCY STEP
    stepsNeeded = max(nt,1.0/dt/df)
    nextPowerOfTwo = int(np.ceil(np.log2(stepsNeeded)))

    Nt = 2**nextPowerOfTwo

    Disp = np.concatenate([Disp, np.zeros(Nt-nt)])

    n = np.arange(1,Nt/2+1)
    wVec = 2.0*np.pi/dt/Nt*n
    wVec = wVec[wVec<=2.0*np.pi*CutOffFrequency]

    FdispIn = np.fft.rfft(Disp)
    FdispIn = FdispIn[1:len(wVec)+1]

    sinTheta = np.sin(angle/180.0*np.pi)
    cosTheta = np.cos(angle/180.0*np.pi)

    alpha = beta*np.sqrt(2.0*(1.0 - nu)/(1.0 - 2.0*nu))
    aSP = beta/alpha
    mu = rho*beta*beta

    #Polarization angle for SV wave
    phaseVelIn = beta[-1]
    polarization = np.array([cosTheta, -sinTheta])

    p = np.lib.scimath.sqrt(1.0-np.square(phaseVelIn/alpha/sinTheta))
    s = np.lib.scimath.sqrt(1.0-np.square(phaseVelIn/beta/sinTheta))
    h = -np.diff(Layers)

    zrelHalfSpace = Layers[-1] - yDRMmin

    #FULL SPACE PROPAGATION OF SV WAVE IN FREQUENCY DOMAIN, RESULT AT HALF-SPACE INTERFACE y=yN
    ufullx = FdispIn*polarization[0]*np.exp(-1j*wVec*cosTheta/phaseVelIn*zrelHalfSpace)
    ufullz = FdispIn*polarization[1]*np.exp(-1j*wVec*cosTheta/phaseVelIn*zrelHalfSpace)
    ufull = np.vstack((ufullx, -1j*ufullz))

    return ufull, Layers, beta, rho, nu, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N, Nt

def GetDisplacementAtInteriorLayer(y,yTop,yBot,uTop,uBot,k,p,s,mu,aSP):
    """
    This function calculates the displacement at interior points of a layer.\n
    
    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    y  : float
        The y-coordinate of the query point
    yTop, yBot  : float
        The y-coordinate of the top and bottom of the parent layer containing query point
    uTop, uBot  : complex
        The displacement at the top and bottom of the parent layer containing query point 
    k  : float
        The horizontal wavenumber (along x-axis)
    p, s  : complex
        Complex coefficients
    h  : float
        The thickness of parent layer
    mu  : float
        The shear modulus of parent layer
    aSP  : float
        The ratio between shear wave velocity and dilatational wave velocity of parent layer
        
    Returns
    -------
    uz  : array
        displacement at the specific query point
    """
    xi = yTop - y
    eta = y - yBot
    [K00xi, K01xi]   = GetKofLayer(k,p,s,xi,mu,aSP) 
    [K00eta, K01eta] = GetKofLayer(k,p,s,eta,mu,aSP) 
    A = K00eta + K00xi*np.array([[1.0,-1.0],[-1.0,1.0]])
    b = -(np.dot(K01xi.T,uTop) + np.dot(K01eta,uBot))
    uz = np.linalg.solve(A, b)
    
    return uz
    
def SoilInterfaceResponse(ufull, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N):
    """
    This function calculates the displacement time series at soil interface positions
    in wave propagation problem, in which the SV or P wave incoming from the half space
    underneath under arbitrary incident angle from 0 to 90 degrees and propagating 
    through stratified soil domain. 
    Note: 
    [1] The coordinate system and displacement positive axes:

        y(V) ^
             |
             |
             o-----> x(U)
             
    [2] At each frequency, the horizontal and vertical displacements are calculated 
        based on Eduardo Kausel's Stiffness Matrix Method, in "Fundamental 
        Solutions in Elastodynamics, A Compendium", chap. 10, pp. 140--159 

    @visit  https://github.com/SeismoVLAB/SVL\n
    @author Kien T. Nguyen 2021, ORCID: 0000-0001-5761-3156
    
    Parameters
    ----------
    ufull  : array
        Displacement of full-space problem at the position of half-space surface 
        Note that the vertical components are multiplied with -1j
    wVec  : array
        Angular frequency spectrum 
    p, s  : array
        Complex coefficients
    h  : array
        The thickness of soil layers
    mu  : array
        The shear modulus of soil layers
    aSP  : array
        The ratio between shear wave velocity and dilatational wave velocity of soil layers
    phaseVelIn  : float
        The phase velocity of incoming wave in the half space underneath
    sinTheta  : float
        Sine of the incoming angle
    N  : int
        Number of soil layers, including imaginary layer and half space
        
    Returns
    -------
    uInterface  : array
        Displacements at the soil layer interface positions, in frequency domain
    """
    nfi = len(wVec)
    uInterface = np.zeros((2*N,2*N,nfi), dtype=complex)

    for fi in range(nfi):
        w = wVec[fi]
        k = w*sinTheta/phaseVelIn
        Kglobal = np.zeros((2*N,2*N), dtype=complex)

        #Assemble each layer
        for i in range(N-1):
            [K00, K01] = GetKofLayer(k, p[i], s[i], h[i], mu[i], aSP[i])
            Kglobal[2*i:2*i+2,2*i:2*i+2]     += K00
            Kglobal[2*i:2*i+2,2*i+2:2*i+4]   += K01
            Kglobal[2*i+2:2*i+4,2*i:2*i+2]   += K01.T
            Kglobal[2*i+2:2*i+4,2*i+2:2*i+4] += K00*np.array([[1.0,-1.0],[-1.0,1.0]])

        #Assemble each layer 
        Khalfspace = GetKofHalfSpace(k, p[-1], s[-1], mu[-1])
        Kglobal[2*N-2:2*N,2*N-2:2*N]  += Khalfspace
    
        #Assembel force vector
        forceVec = np.zeros((2*N,1), dtype=complex)
        Kfull = GetKofFullSpace(k, p[-1], s[-1], mu[-1])
        forceVec[2*N-2:2*N,:] = (Kfull.dot(ufull[:,fi])).reshape(2,1) 
    
        #Displacement at interface
        uInterface[:,:,fi] = np.linalg.solve(Kglobal, forceVec)

    return uInterface

import numpy as np
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

#INCLINATION ANGLE
angle = 20.0

#RICKER PULSE PARAMETERS
f   = 2.00
to  = 1.00 

#TIME-SPACE GRID
time = np.linspace(0.0, 20.0, 200)
Y, X = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(xmin, xmax, nx))
nt = len(time)
dt = 20.0/200.0

#RICKER PULSE 
factor = np.square(np.pi*f*(time - to))
Disp = np.multiply((1 - 2.0*factor), np.exp(-factor))

ufull, layers, beta, rho, nu, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N, Nt = DataPreprocessing(Disp, layers, beta, rho, nu, angle, ymin, nt, dt)

uInterface = SoilInterfaceResponse(ufull, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N)

#PLOT THE WAVE PROPAGATION
fig, ax = plt.subplots()

U = np.zeros([nx,ny,Nt])
V = np.zeros([nx,ny,Nt])
for j in range(ny):
    for i in range(nx):
        U[i,j,:], V[i,j,:] = PSVbackground2Dfield(uInterface, layers, wVec, p, s, h, mu, aSP, phaseVelIn, sinTheta, N, Nt, xmin, X[i,j], Y[i,j])

for k in range(200):
    Z = np.sqrt(U[:,:,k]**2 + V[:,:,k]**2)
    ax.pcolormesh(X, Y, Z, cmap='RdBu', vmin=0.0, vmax=1.0)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.axis('equal')
    plt.draw()
    plt.savefig("/home/danilo/Documents/DRM/output_"+ str(k) +".png")
    plt.pause(0.1)
    plt.cla()
