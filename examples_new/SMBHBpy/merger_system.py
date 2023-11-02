import numpy as np
import SMBHBpy.cosmo as cosmo
import SMBHBpy.halo as halo
import SMBHBpy.constants as c


class SystemProp:
    """
    A class describing the properties of a binary system

    Attributes:
        m1 (float): The mass of the first component - in units of (solar_mass *G/c**2)
        m2 (float): The mass of the second component - in units of (solar_mass *G/c**2)
        D  (float): The luminosity distance to the binary
        halo1 (DMHalo): The object describing the dark matter halo around m1
        halo2 (DMHalo): The object describing the dark matter halo around m2
        sigma (float): Velocity dispersion of the host galaxy-bulge
        b_max (float): Maximum impact parameter for the Coulomb logarithm
    """


    def __init__(self, m1, m2, halo1, halo2, sigma, b_max, k, H, rho_hardening, D=1., inclination_angle = 0., pericenter_angle=0., includeHaloInTotalMass=False, relcorr = False):
        """
        The constructor for the SystemProp class

        Parameters:
            m1 : float
                The mass of the first component
            m2 : float
                The mass of the second component
            halo1 (DMHalo): 
                The object describing the dark matter halo around m1
            halo2 (DMHalo): 
                The object describing the dark matter halo around m2
            D :  float
                The luminosity distance to the binary
            sigma : float 
                Velocity dispersion of the host galaxy-bulge
            b_max : float
                Maximum impact parameter for the Coulomb logarithm
            k : float
                Indicates whether the halos are static (k = 0) or non-static, i.e. rotating (k = 1).
                k = 0 can be selected independently of the eccentricity, k = 1 only if circular orbits are involved.
            inclination_angle : float
                The inclination angle (usually denoted iota) at which the system is oriented, see https://arxiv.org/pdf/1807.07163.pdf
            pericenter_angle : float
                The angle at which the pericenter is located wrt the observer, denoted as beta in https://arxiv.org/pdf/1807.07163.pdf
            includeHaloInTotalMass : bool
                Whether to include the dark matter halo mass in the calculation of the enclosed mass
            relcorr : bool
                Specifies whether relativistic effects should be considered in the calculation of dynamic friction or not.
                https://arxiv.org/pdf/2204.12508.pdf (except for the typo in the gamma factor)
        """
        self.m1 = m1
        self.m2 = m2

        self.sigma = sigma
        self.b_max = b_max
        self.k = k
        self.D = D
        
        #New:
        self.H = H                            # for hard binary, determined from scattering experiments
        self.rho_hardening = rho_hardening    # Have to be constant a priori
        #
        
        self.halo1 = halo1
        self.halo2 = halo2
        
        self.halo1.r_min = self.r_isco_1() if halo1.r_min == 0. else halo1.r_min
        self.halo2.r_min = self.r_isco_2() if halo2.r_min == 0. else halo2.r_min

        self.inclination_angle = inclination_angle
        self.pericenter_angle = pericenter_angle

        self.includeHaloInTotalMass = includeHaloInTotalMass
        self.relcorr = relcorr


    def r_isco_1(self):
        """
        The function returns the radius of the Innermost Stable Circular Orbit (ISCO) of the Schwarzschild black hole with mass m1

        Returns:
            out : float
                The radius of the ISCO of m1
        """
        return 6.*self.m1
    
    def r_isco_2(self):
        """
        The function returns the radius of the Innermost Stable Circular Orbit (ISCO) of the Schwarzschild black hole with mass m2

        Returns:
            out : float
                The radius of the ISCO of m2
        """
        return 6.*self.m2
    
    def r_infl(self): # Also known as r_h
        """
        The function returns the (gravitational) influence radius of the binary with mass M = m1+m2

        Returns:
            out : float
                The (gravitational) influence radius
        """
        return self.m_total()/(self.sigma)**2

    def r_schwarzschild_1(self):
        """
        The function returns the Schwarzschild radius of the Schwarzschild black hole with mass m1

        Returns:
            out : float
                The Schwarzschild radius of m1
        """
        return 2.*self.m1
    
    def r_schwarzschild_2(self):
        """
        The function returns the Schwarzschild radius of the Schwarzschild black hole with mass m1

        Returns:
            out : float
                The Schwarzschild radius of m2
        """
        return 2.*self.m2

    def m_reduced(self):
        """
        The function returns the reduced mass of the binary system of m1 and m2

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float
                The reduced mass
        """
        return self.m1 * self.m2 / (self.m1 + self.m2)

    def redshifted_m_reduced(self):
        """
        The function returns the redshifted reduced mass of the binary system of m1 and m2

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float
                The redshifted reduced mass
        """
        return (1. + self.z()) * self.m_reduced()

    def m_total(self):
        """
        The function returns the total mass of the binary system of m1 and m2

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass
        Returns:
            out : float
                The total mass
        """
        return self.m1 + self.m2

    def m_chirp(self):
        """
        The function returns the chirp mass of the binary system of m1 and m2

        Returns:
            out : float
                The chirp mass
        """
        return self.m_reduced()**(3./5.) * self.m_total()**(2./5.)

    def redshifted_m_chirp(self):
        """
        The function returns the redshifted chirp mass of the binary system of m1 and m2

        Returns:
            out : float
                The redshifted chirp mass
        """
        return (1.+self.z()) * self.m_chirp()

    def z(self):
        """
        The function returns the redshift as a measure of distance to the system
        According to the Hubble Law

        Returns:
            out : float
                The redshift of the system
        """
        return cosmo.HubbleLaw(self.D)
    
    def omega_s(self, r):
        """
        The function returns the angular frequency of the binary in a circular orbit 
        
        Parameters:
            r : float or array_like
                The radius at which to evaluate the orbital frequency

        Returns:
            out : float or array_like (depending on r)
                The orbital frequency
        """
        return np.sqrt(self.m_total()/r**3)

    def mass(self, r):
        """
        The function returns the total mass enclosed in a sphere of radius r.
            This includes the central mass and the mass of the dark matter halo if includeHaloInTotalMass=True

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or array_like (depending on r)
                The enclosed mass
        """
        return np.ones(np.shape(r))*self.m1 + (self.halo.mass(r) if self.includeHaloInTotalMass else 0.)
    