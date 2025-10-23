import numpy as np
from scipy.integrate import quad
import collections

import SMBHBpy.constants as c
import SMBHBpy.halo


class DissipativeForce:
    """
    This is a model class from which the dissipative forces should be derived
    It defines standard integration function that calculate dE/dt, dL/dt from an arbitrary F, which is to be defined by the class
    Of course, these functions can be overriden in case of an analytic solution (e.g. gravitational wave losses)
    """
    name = "DissipativeForce"

    @staticmethod
    def get_orbital_elements(sp, a, e, phi, opt):
        r = a*(1. - e**2)/(1. + e*np.cos(phi))
        v = np.sqrt(sp.m_total() *(2./r - 1./a))
        return r, v


    def F(self, sp, r, v, e, opt):
        """
        Placeholder function that models the dissipative force strength.

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            v  (float)      : Total velocity of m1 and m2 on Keplerian orbits
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The strength of the dissipative force
        """
        pass


    def dE_dt(self, sp, a, e, opt):
        """
        The function calculates the energy loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to accretion
        """
        if  isinstance(a, (collections.Sequence, np.ndarray)):
            return np.array([self.dE_dt(sp, a_i, e, opt) for a_i in a])
        if e == 0.:
            r, v = self.get_orbital_elements(sp, a, 0., 0., opt)
            return -self.F(sp, a, v, e, opt)*v
        else:
            def integrand(phi):
                r, v = self.get_orbital_elements(sp, a, e, phi, opt)
                return self.F(sp, r, v, e, opt)*v / (1.+e*np.cos(phi))**2
            return -(1.-e**2)**(3./2.)/2./np.pi * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def dL_dt(self, sp, a, e, opt):
        """
        The function calculates the angular momentum loss due to a force F(r,v) by averaging over a Keplerian orbit
           with semimajor axis a and eccentricity e
        According to https://arxiv.org/abs/1908.10241

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to accretion
        """
        def integrand(phi):
            r, v = self.get_orbital_elements(sp, a, e, phi, opt)
            return self.F(sp, r, v, e, opt)/v / (1.+e*np.cos(phi))**2
        return -(1.-e**2)**(3./2.)/2./np.pi *np.sqrt(sp.m_total() * a*(1.-e**2)) * quad(integrand, 0., 2.*np.pi, limit = 100)[0]


    def __str__(self):
        return self.name

class GWLoss(DissipativeForce):
    name = "GWLoss"

    def dE_dt(self, sp, a, e, opt):
        """
        The function gives the energy loss due to radiation of gravitational waves
            for a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy loss due to radiation of gravitational waves of an Keplerian orbit
        """
        return -32./5. * sp.m_reduced()**2 * sp.m_total()**3 / a**5  / (1. - e**2)**(7./2.) * (1. + 73./24. * e**2 + 37./96. * e**4)

    def dL_dt(self, sp, a, e, opt):
        """
        The function gives the loss of angular momentum due to radiation of gravitational waves
           for a Keplerian orbit with semimajor axis a and eccentricity e
        According to Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum loss due to radiation of gravitational waves
        """
        return -32./5. * sp.m_reduced()**2 * sp.m_total()**(5./2.) / a**(7./2.)  / (1. - e**2)**2 * (1. + 7./8.*e**2)


class DynamicalFriction(DissipativeForce):
    name = "DynamicalFriction"

    def __init__(self, ln_Lambda=-1):
        self.ln_Lambda = ln_Lambda

    def F(self, sp, r, v, e, opt):
        """
        The function gives the force of the dynamical friction of an object inside a dark matter halo at r (distance between the SMBHs) (since we assume a spherically symmetric halo)
        and with velocity v
        The self.ln_Lambda is the Coulomb logarithm, for which different authors use different values. Set to -1 so that ln_Lambda = sp.Coulomb_log
        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            r  (float)      : The radius of the orbiting object
            v  (float or tuple)   : The speed of the orbiting object
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The magnitude of the dynamical friction force
        """
        ln_Lambda = self.ln_Lambda
        halo1 = sp.halo1
        halo2 = sp.halo2

        if ln_Lambda < 0.:
            ln_Lambda = sp.Coulomb_log

        relCovFactor = 1.
        if sp.relcorr == True:
            relCovFactor = (1. + v**2)**2 / (1. - v**2)

        density1 = halo1.density(r)
        density2 = halo2.density(r)
            
        if e == 0.:
            F_df = 4.*np.pi * ln_Lambda * sp.m_reduced() * (sp.m1*density2/(1-sp.k*np.sqrt(sp.m2/sp.m_total()))**2 + sp.m2*density1/(1-sp.k*np.sqrt(sp.m1/sp.m_total()))**2)/v**2 * relCovFactor
            return np.nan_to_num(F_df)
        else:
            F_df = 4.*np.pi * ln_Lambda * sp.m_reduced() * (sp.m1*density2 + sp.m2*density1)/v**2 * relCovFactor
            return np.nan_to_num(F_df)
   