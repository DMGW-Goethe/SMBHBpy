import numpy as np

class MatterHalo:
    """
    A class describing a spherically symmetric, static Matter Halo

    Attributes:
        r_min (float): An minimum radius below which the density is always 0, this is initialized to 0
    """

    def __init__(self):
        """
        The constructor for the MatterHalo class
        """
        self.r_min = 0.

    def density(self, r):
        """
        The density function of the halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        pass
    

class ConstHalo(MatterHalo):
    """
    A class describing a spherically symmetric, static, and constant Matter Halo

    Attributes:
        r_min (float): An minimum radius below which the density is always 0, this is initialized to 0
        rho_0 (float): The constant density of the halo
    """

    def __init__(self, rho_0):
        """
        The constructor for the ConstHalo class

        Parameters:
            rho_0 : float
                The constant density of the halo
        """
        MatterHalo.__init__(self)
        self.rho_0 = rho_0

    def density(self, r):
        """
        The constant density function of the halo

        Parameters:
            r : float or array_like
                The radius at which to evaluate the density

        Returns:
            out : float or array_like (depending on r)
                The density at the radius r
        """
        return np.where(r > self.r_min, self.rho_0, 0.)


    def mass(self, r):
        """
        The mass that is contained in the halo in the spherical shell of size r.

        Parameters:
            r : float or array_like
                The radius at which to evaluate the mass

        Returns:
            out : float or np.ndarray (depending on r)
                The mass inside the spherical shell of size r
        """
        return 4./3.*np.pi *self.rho_0 * (r**3 - self.r_min**3)

    def __str__(self):
        """
        Gives the string representation of the object

        Returns:
            out : string
                The string representation
        """
        return "ConstHalo"
