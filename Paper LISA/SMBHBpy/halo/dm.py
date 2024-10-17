from .halo import *

class NFW(MatterHalo):
    """
    A class describing a Navarro-Frenk-White (NFW) halo profile.
    The density is given by
        rho(r) = rho_s / (r/r_s) / (1 + r/r_s)**2

    Attributes:
        r_min (float): An minimum radius below which the density is always 0
        rho_s (float): The density parameter of the profile
        r_s   (float): The scale radius of the profile
    """

    def __init__(self, rho_s, r_s):
        """
        The constructor for the NFW class

        Parameters:
            rho_s : float
                The density parameter of the NFW profile
            r_s : float
                The scale radius of the NFW profile
        """
        MatterHalo.__init__(self)
        self.rho_s = rho_s
        self.r_s = r_s

    def density(self, r):
        """
        The density function of the NFW halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_s / (r/self.r_s) / (1. + r/self.r_s)**2, 0.)


    def mass(self, r):
        """
        The mass that is contained in the NFW halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        def NFWmass(r):
            return 4.*np.pi*self.rho_s * self.r_s**3 * (np.log((self.r_s + r)/self.r_s) + self.r_s / (self.r_s + r) - 1.)

        return np.where(r > self.r_min, NFWmass(r) - NFWmass(self.r_min), 0.)


    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"NFW(rho_s={self.rho_s:0.1e}, r_s={self.r_s:0.1e})"
    

class Spike(MatterHalo):
    """
    A class describing a spike halo profile
    The density is given by
        rho(r) = rho_sp * (r_sp/r)**(alpha)

    Attributes:
        r_min  (float): An minimum radius below which the density is always 0
        rho_sp (float): The density parameter of the spike profile
        r_sp   (float): The scale radius of the spike profile
        alpha  (float): The power-law index of the spike profile, with condition 0 < alpha < 3
    """

    def __init__(self, rho_sp, r_sp, alpha):
        """
        The constructor for the Spike class

        Parameters:
            rho_sp : float
                The density parameter of the spike profile
            r_sp : float
                The scale radius of the spike profile
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3
        """
        MatterHalo.__init__(self)
        self.rho_sp = rho_sp
        self.alpha = alpha
        self.r_sp = r_sp

    def density(self, r):
        """
        The density function of the spike halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_sp * (self.r_sp/r)**self.alpha, 0.)

    def mass(self, r):
        """
        The mass that is contained in the spike halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        def spikeMass(r):
            return 4*np.pi*self.rho_sp*self.r_sp**self.alpha * r**(3.-self.alpha) / (3.-self.alpha)

        return np.where(r > self.r_min, spikeMass(r) - spikeMass(self.r_min),0.)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"Spike(rho_sp={self.rho_sp:0.1e}, r_sp={self.r_sp:0.1e}, alpha={self.alpha:0.1e})"


class SpikedNFW(NFW, Spike):
    """
    A class describing a Navarro-Frenk-White (NFW) halo profile with a spike below a given radius r_sp.
    The density is given by
        rho(r) = rho_s / (r/r_s) / (1 + r/r_s)**2    for r > r_sp
        rho(r) = rho_sp * (r_sp/r)**(alpha)    for r < r_sp

    with a continuity condition rho_sp= rho_s / (r_sp/r_s) / (1 + r_sp/r_s)**2

    Attributes:
        r_min  (float): An minimum radius below which the density is always 0
        rho_s  (float): The density parameter of the NFW profile
        r_s    (float): The scale radius of the NFW profile
        r_sp   (float): The scale radius of the spike profile
        rho_sp (float): The density parameter of the spike profile
        alpha  (float): The power-law index of the spike profile, with condition 0 < alpha < 3
    """

    def __init__(self, rho_s, r_s, r_sp, alpha):
        """
        The constructor for the SpikedNFW class

        Parameters:
            rho_s : float
                The density parameter of the NFW profile
            r_s : float
                The scale radius of the NFW profile
            r_sp : float
                The scale radius of the spike profile
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3
        """
        NFW.__init__(self, rho_s, r_s)
        rho_sp = rho_s * r_s/r_sp / (1+r_sp/r_s)**2
        Spike.__init__(self, rho_sp, r_sp, alpha)

    def density(self, r):
        """
        The density function of the NFW+spike halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r < self.r_sp, Spike.density(self, r), NFW.density(self, r))

    def mass(self, r):
        """
        The mass that is contained in the NFW+spike halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        return np.where(r < self.r_sp, Spike.mass(self, r), NFW.mass(self, r) - NFW.mass(self, self.r_sp) + Spike.mass(self,self.r_sp))


    def FromNFW(nfw, M_bh, alpha):
        """
        This function takes an NFW object and computes the corresponding SpikedNFW profile, given the size of a massive Black Hole at the center,
            according to the description in II.B of https://arxiv.org/pdf/1408.3534.pdf

        Parameters:
            nfw : NFW object
                The NFW object in question
            M_bh : float
                The mass of the Massive Black Hole in the center
            alpha : float
                The power-law index of the spike profile, with condition 0 < alpha < 3

        Returns:
            out : SpikedNFW object
                The SpikedNFW object with the corresponding parameters
        """
        r = np.geomspace(1e-3*nfw.r_s, 1e3*nfw.r_s)
        M_to_r = interp1d(nfw.mass(r), r, kind='cubic', bounds_error=True)
        r_h = M_to_r(2.* M_bh)
        r_sp = 0.2*r_h
        return SpikedNFW(nfw.rho_s, nfw.r_s, r_sp, alpha)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return f"SpikedNFW(rho_s={self.rho_s:0.1e}, r_s={self.r_s}, r_sp={self.r_sp:0.1e}, alpha={self.alpha:0.1e})"
    