import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt
from scipy.special import expn

#-------------------Constants----------------------------------------------#
G = 6.67384e-8 # cm^3/g/s^2
c = 2.99792458e10 # speed of light  / cm/s
hP = 6.6260688e-27 # Planck's const / erg s
hbar = hP/2/np.pi
kB = 1.3806488e-16 # Boltzmann const / erg/K
mH = 1.66053892e-24 # atomic mass unit / g
me = 9.1093898e-28 # electron mass / g
alphaEM = 1/137.0359895 # fine structure constant
sigmaT = 8*np.pi/3*(alphaEM*hbar/me/c)**2 # Thomson cross section / cm^2
a_rad =  np.pi**2/15*kB*(kB/hbar/c)**3 # radiation const / erg/cm^3/K^4
stefan = a_rad*c/4 # Stefan-Boltzmann const / erg/cm^2/s/K^4
eV = 1.60217733e-12 # 1 electron volt / erg
keV = 1e3*eV
Msun = 1.98892e33 # solar mass / g
LEsun = 4*np.pi*c*G*Msun*mH/sigmaT # Eddington luminosity / erg/s
MEsun = LEsun/c**2
pc = 3.08567758e18 # 1 parsec / cm

kappa_es = 0.4 # opacity by electron scattering / cm^2/g
kappa_0 = 6.45e22 # krammers opacity at (rho,T)=(1,1) in cgs
nu_0 = 2.4e21*15 # electron-ion coupling (Coulomb log = 15)
#-------------------Radiation----------------------------------------------#

def kappa_ff(rho,T):
    """ krammers law for opacity by free-free absorption """
    return kappa_0*rho/T**3.5


def nu_E(rho,T):
    """ collision rate of ions with electrons """
    return nu_0*rho/T**1.5

def BlackBody(nu,T):
    """ black-body spectrum """
    return 2*hP*nu**4/c**2/np.expm1(hP*nu/kB/T)

def InvCompton(nu,T,Es):
    """ spectrum of inverse-compton scattered photons
    assuming compton y parameter is unity
    return: normalized spectrum
      dI/dln(nu) /[int dI/d(nu) d(nu)] if nu >= Es/hP
      0 if nu < Es/hP
    reference: Shapiro++ 1976
    """
    x = hP*nu/kB/T
    y = Es/kB/T
    I = (1 + x*(1 + x/2*(1 + x/3*(1 + x/4))))*np.exp(-x)
    I /= (50 + y*(26 + y*(7 + y)))/24*np.exp(-y) + expn(1,y)
    return np.where(x<y, 0, I)

def my_newton(f,x0,Df,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None



class StandardModel:
    """ Standard model of accretion disks reference: 
    Shakura and Sunyaev 1973
    """
    def __init__(self, M, M_dot, alpha, r_out, r_in=3, N=256):
        """
        M = mass of black hole / solar mass
        M_dot = mass accretion rate 
        alpha = viscosity parameter (0<alpha<=1)
        r_out = radius of disk's outer edge 
        r_in = radius of disk's inner edge 
        N = number of grid points in radial coordinate
            (excluding inner boundary)
        """
        rg = 2*G*M*Msun/2.99792458e10**2
        self.M = M*Msun
        self.M_dot = M_dot*M*MEsun
        self.alpha = alpha
        self.r_in = r_in*rg
        self.r_out = r_out*rg
        self.r = np.geomspace(self.r_in, self.r_out, N+1)[1:]
        self.y = StandardModel.solve(self, self.r)

    def solve(self, r, Sigma=None): 
        """
        r = distance from disk center (black hole) 
        Sigma = initial guess for Surface density at r 
          if r is array, Sigma is used only at r[-1];
          if Sigma is None, asymptotic solution at large r
          is used as an initial guess
        return y = (Sigma,p,T,H,F,tau) where
          Sigma = surface density at r 
          p = pressure at r 
          T = temperature on disk at r 
          H = scale height at r 
          F = flux from one side of the disk 
          tau = optical depth at r
        if r is a scalar, y is tuple of length 6
        else if r has shape (n,), y has shape (6,n)
        """
        if hasattr(r, '__len__'):
            Y,y = [],[Sigma]
            for r in r[::-1]:
                y = StandardModel.solve(self, r, y[0])
                # print('y is ', y)
                Y.append(y)
                # print('Y is ', Y)
            return np.array(Y[::-1]).T

        if r <= self.r_in: return (0,)*6
        Omega = np.sqrt(G*self.M/r**3) # Keplerian angular velocity
        f = self.M_dot*(1 - (self.r_in/self.r)**0.5)*Omega/2/np.pi
        kBmH = 2*kB/mH
        Sigma0 = (256*stefan*f**7/Omega**2/(9*alpha**8*kappa_0*kBmH**7.5))**0.1
        # print('Sigma0 is ', Sigma0)

        def equation(Sigma):
            global p,T,H,F,tau
            # if Sigma <= 0: 
            #     Sigma = Sigma0
            cs2 = f/Sigma/alpha #self.alpha # sound spped squared
            H = cs2**0.5/Omega # disk height
            rho = Sigma/2/H # density
            p = cs2*rho # pressure
            if T is None: 
                T = cs2/kBmH # temperature
            temp = lambda x: rho*kBmH*x + a_rad/3*x**4 - p # temperature
            # print('temp is ', T)
            dtemp = lambda x: rho*kBmH + 4*a_rad/3*x**3 # derivative
            T = newton(temp, T, dtemp)
            
            kappa = kappa_es + kappa_ff(rho,T) # opacity
            tau = kappa*Sigma/2 # optical depth
            Q = 1.5*f*Omega # heat generation
            F = 16/3*stefan*T**4/tau # flux from one side
            return Q - 2*F # equation for radiative equilibrium

        if Sigma is None: 
            Sigma = Sigma0
        # p,T,H,F,tau = [None]*5
        Sigma = newton(equation, Sigma, tol=10e-10)
        return (Sigma,p,T,H,F,tau)

    def radius(self): return self.r
    def surface_density(self): return self.y[0]
    def pressure(self): return self.y[1]
    def temperature(self): return self.y[2]
    def height(self): return self.y[3]
    def flux(self): return self.y[4]
    def optical_depth(self): return self.y[5]
    def infall_velocity(self): return self.M_dot/2/np.pi/self.r/self.y[0]
    def rotation_velocity(self): return (G*self.M/self.r)**0.5
    def photon_pressure(self): return a_rad*self.y[2]**4/3
    def gas_pressure(self): return 2*kB/mH*self.density()*self.y[2]
    def density(self): return self.y[0]/2/self.y[3]
    def effective_temperature(self): return (self.y[4]/stefan)**0.25
    def opacity(self):
        return kappa_es + kappa_ff(self.density(), self.y[2])

    # def spectrum(self, nu, D=10*pc, i=0):
    #     """ spectrum of emitted photons from disk surface
    #     nu = photon frequency / Hz
    #     D = distance to obersever / cm
    #     i = inclination angle wrt line of sight / radian
    #     return d(intensity)/dln(nu) / erg/s/cm^2
    #     """
    #     Teff = self.effective_temperature()
    #     I = BlackBody(np.expand_dims(nu, -1), Teff)
    #     I = np.trapz(I*self.r, self.r)
    #     return 2*np.pi*np.cos(i)/D**2*I


p,T,H,F,tau = [None]*5
M = np.array([10]*4) # BH mass / solar mass
M_dot = np.array([1, 0.1, 1, 0.1])
alpha = np.array([1, 1, 0.1, 0.1])
r_max = 3e3 # radius of disk's outer edge 

color = ['C0', 'C1', 'C2', 'C3']

plt.figure(figsize=(6,8))
plt.subplots_adjust(top=0.98, bottom=0.1,
                    right=0.98, left=0.12, hspace=0)

for M,M_dot,alpha,c in zip(M,M_dot,alpha,color):
    s = StandardModel(M, M_dot, alpha, r_max)
    r = s.radius()
    H = s.height()
    u = s.infall_velocity()
    v = s.rotation_velocity()
    tau = s.optical_depth()
    label = r'$\dot m=%s, \alpha=%s$'%(str(M_dot),str(alpha))
    plt.subplot(3,1,1); plt.loglog(r, u[5]/v[5], c=c)
    plt.subplot(3,1,2); plt.loglog(r, tau[5], c=c)
    plt.subplot(3,1,3); plt.loglog(r, H[5]/r[5], c=c, label=label)

y_label = ['$u/v$', 'optical depth', '$H/r$']
y_ticks = [[1e-5,1e-4,1e-3], [1e2,1e3], [1e-3,1e-2]]

for i in range(3):
    plt.subplot(3,1,i+1)
    plt.ylabel(y_label[i], fontsize=14)
    plt.yticks(y_ticks[i])
    if i<2: plt.tick_params('x', which='both', direction='in')
    plt.xlim([s.r_in, s.r_out])

plt.xlabel('$r$ = distance from black hole / cm', fontsize=14)
plt.legend()
plt.show()



p,T,H,F,tau = [None]*5
M = np.array([10]*4) # BH mass / solar mass
M_dot = np.array([1, 0.1, 1e-2, 5e-3])
alpha = np.array([1, 1, 0.1, 0.1])
r_max = 3e3 # radius of disk's outer edge 

color = ['C0', 'C1', 'C2', 'C3']

plt.figure(figsize=(6,8))
plt.subplots_adjust(top=0.98, bottom=0.1,
                    right=0.98, left=0.12, hspace=0)

for M,M_dot,alpha,c in zip(M,M_dot,alpha,color):
    s = StandardModel(M, M_dot, alpha, r_max)
    r = s.radius()
    T_eff = s.effective_temperature()
    p = s.pressure()
    p_gas = s.gas_pressure()
    kappa = s.opacity()
    label = r'$\dot m=%s, \alpha=%s$'%(str(M_dot),str(alpha))
    plt.subplot(3,1,1);
    plt.loglog(r, T_eff[5], c=c, label=label)
    plt.subplot(3,1,2);
    plt.loglog(r, p[5], c=c)
    l = plt.loglog(r, p_gas[5], c=c, ls='--')
    if c=='b': l[0].set_label('gas pressure')
    plt.subplot(3,1,3);
    plt.loglog(r, kappa[5], c=c)
    l = plt.loglog(r, [kappa_es]*len(r), c=c, ls='--')
    if c=='r': l[0].set_label('electron scattering')

y_label = ['$T_{eff}$ / K',
           'pressure / dyne/cm$^2$', 'opacity / cm$^2$/g']
y_ticks = [[1e5,1e6], [1e9,1e12,1e15], [1e-1,1e0]]

for i in range(3):
    plt.subplot(3,1,i+1)
    plt.ylabel(y_label[i], fontsize=14)
    plt.yticks(y_ticks[i])
    if i<2: plt.tick_params('x', which='both', direction='in')
    plt.xlim([s.r_in, s.r_out])
    plt.legend()

plt.xlabel('$r$ = distance from black hole / cm', fontsize=14)
plt.show()



