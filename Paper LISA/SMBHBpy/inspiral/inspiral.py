import numpy as np
from scipy.integrate import solve_ivp, quad

import time
import SMBHBpy.constants as c
from . import forces

class Classic:
    """
    A class bundling the functions to simulate an inspiral with basic energy conservation arguments
    This class does not need to be instantiated
    """

    class EvolutionOptions:
        """
        This class allows to modify the behavior of the evolution of the differential equations

        Attributes:
            accuracy : float
                An accuracy parameter that is passed to solve_ivp
            verbose : int
                A verbosity parameter ranging from 0 to 2
            elliptic : bool
                Whether to model the inspiral on eccentric orbits, is set automatically depending on e0 passed to Evolve
            dissipativeForces : list of forces.DissipativeForces
                List of the dissipative forces employed during the inspiral
            gwEmissionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            dynamicalFrictionLoss : bool
                These parameters are for backwards compatibility - Only applies if dissipativeForces=None - then forces.GWLoss is added to the list
            **kwargs : additional parameter
                Will be saved in opt.additionalParameters and will be available throughout the integration

        """
        def __init__(self, accuracy=1e-11, verbose=1, elliptic=True,
                                    dissipativeForces=None, gwEmissionLoss = True, dynamicalFrictionLoss = True,
                                    considerRelativeVelocities=False, progradeRotation = True,
                                    **kwargs):
            self.accuracy = accuracy
            self.verbose = verbose
            self.elliptic = elliptic
            if dissipativeForces is None:
                dissipativeForces = []
                if gwEmissionLoss:
                    dissipativeForces.append(forces.GWLoss())
                if dynamicalFrictionLoss:
                    dissipativeForces.append(forces.DynamicalFriction())
            self.dissipativeForces = dissipativeForces
            self.considerRelativeVelocities = considerRelativeVelocities
            self.progradeRotation = progradeRotation
            self.additionalParameters = kwargs


        def __str__(self):
            s = "Options: dissipative Forces emplyed {"
            for df in self.dissipativeForces:
                s += str(df) + ", "
            s += "}" + f", accuracy = {self.accuracy:.1e}"
            for key, value in self.additionalParameters.items():
                s += f", {key}={value}"
            return s


    def E_orbit(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the orbital energy of the binary for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit, default is 0
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The energy of the Keplerian orbit
        """
        return - sp.m1*sp.m2/(2.*a)


    def dE_orbit_da(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the derivative of the orbital energy wrt the semimajor axis a
           of the binary for a Keplerian orbit with semimajor axis a and eccentricity e
        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The derivative of the orbital energy wrt to a of the Keplerian orbit
        """
        return sp.m1*sp.m2/(2.*a**2)

    def L_orbit(sp, a, e, opt=EvolutionOptions()):
        """
        The function gives the angular momentum of the binary for a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The angular momentum of the Keplerian orbit
        """
        return np.sqrt(a * (1-e**2) * sp.m_total()) * sp.m_reduced()


    def dE_dt(sp, a, e=0., opt=EvolutionOptions()):
        """
        The function gives the total energy loss during the inspiral due to the dissipative effects
           on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total energy loss
        """
        dE_dt_tot = 0.
        dE_dt_out = ""
        for df in opt.dissipativeForces:
            dE_dt = df.dE_dt(sp, a, e, opt)
            dE_dt_tot += dE_dt
            if opt.verbose > 2:
                dE_dt_out += f"{df.name}:{dE_dt}, "

        if opt.verbose > 2:
            print(dE_dt_out)
        return  dE_dt_tot


    def dL_dt(sp, a, e, opt=EvolutionOptions()):
        """
        The function gives the total angular momentum loss of the binary
            on a Keplerian orbit with semimajor axis a and eccentricity e

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The total angular momentum loss
        """
        dL_dt_tot = 0.
        dL_dt_out = ""
        for df in opt.dissipativeForces:
            dL_dt = df.dL_dt(sp, a, e, opt)
            dL_dt_tot += dL_dt
            if opt.verbose > 2:
                dL_dt_out += f"{df.name}:{dL_dt}, "

        if opt.verbose > 2:
            print(dL_dt_out)
        return  dL_dt_tot


    def da_dt(sp, a, e=0., opt=EvolutionOptions(), return_dE_dt=False):
        """
        The function gives the secular time derivative of the semimajor axis a (or radius for a circular orbit) due to gravitational wave emission and dynamical friction
        The equation is obtained by the relation
            E = -m_1 * m_2 / 2a

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            opt (EvolutionOptions): The options for the evolution of the differential equations
            dE_dt (bool)    : Whether to return dE_dt in addition to da_dt, to save computation time

        Returns:
            da_dt : float
                The secular time derivative of the semimajor axis
            dE_dt : float
                The secular time derivative of the orbital energy
        """
        dE_dt = Classic.dE_dt(sp, a, e, opt)
        dE_orbit_da = Classic.dE_orbit_da(sp, a, e, opt)

        if return_dE_dt:
            return dE_dt / dE_orbit_da, dE_dt

        return    ( dE_dt / dE_orbit_da )


    def de_dt(sp, a, e, dE_dt=None, opt=EvolutionOptions()):
        """
        The function gives the secular time derivative of the eccentricity due to gravitational wave emission and dynamical friction
        The equation is obtained by the time derivative of the relation
            e^2 = 1 + 2EL^2 / m_total^2 / m_reduced^3
           as given in Maggiore (2007)

        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a  (float)      : The semimajor axis of the Keplerian orbit, or the radius of a circular orbit
            e  (float)      : The eccentricity of the Keplerian orbit
            dE_dt (float)   : Optionally, the dE_dt value if it was computed previously
            opt (EvolutionOptions): The options for the evolution of the differential equations

        Returns:
            out : float
                The secular time derivative of the eccentricity
        """
        if e <= 0. or not opt.elliptic:
            return 0.

        dE_dt = Classic.dE_dt(sp, a, e, opt) if dE_dt is None else dE_dt
        E = Classic.E_orbit(sp, a, e, opt)
        dL_dt = Classic.dL_dt(sp, a, e, opt)
        L = Classic.L_orbit(sp, a, e, opt)

        if opt.verbose > 2:
            print("dE_dt/E=", dE_dt/E, "2dL_dt/L=", 2.*dL_dt/L, "diff=", dE_dt/E + 2.*dL_dt/L)

        return - (1.-e**2)/2./e *(dE_dt/E + 2. * dL_dt/L)


    class EvolutionResults:
        """
        This class keeps track of the evolution of an inspiral.

        Attributes:
            sp : merger_system.SystemProp
                The system properties used in the evolution
            opt : Classic.EvolutionOptions
                The options used during the evolution
            t : np.ndarray
                The time steps of the evolution
            a,R : np.ndarray
                The corresponding values of the semimajor axis - if e=0, this is also called R
            e  : float/np.ndarray
                The corresponding values of the eccentricity, default is zero
            msg : string
                The message of the solve_ivp integration
        """
        def __init__(self, sp, options, t, a, msg=None):
            self.sp = sp
            self.options = options
            self.msg=msg
            self.t = t
            self.a = a
            if not options.elliptic:
                self.e = np.zeros(np.shape(t))
                self.R = a



    def Evolve(sp, a_0, e_0=0., a_fin=0., t_0=0., t_fin=None, opt=EvolutionOptions()):
        """
        The function evolves the coupled differential equations of the semimajor axis and eccentricity of the Keplerian orbits of the inspiralling system
            by tracking orbital energy and angular momentum loss due to gravitational wave radiation and dynamical friction
            
        Parameters:
            sp (SystemProp) : The object describing the properties of the inspiralling system
            a_0  (float)    : The initial semimajor axis
            e_0  (float)    : The initial eccentricity
            a_fin (float)   : The semimajor axis at which to stop evolution
            t_0    (float)  : The initial time
            t_fin  (float)  : The time until the system should be evolved, if None then the estimated coalescence time will be used
            opt   (EvolutionOptions) : Collecting the options for the evolution of the differential equations

        Returns:
            ev : Evolution
                An evolution object that contains the results
        """
        opt.elliptic = e_0 > 0.

        # calculate relevant timescales
        def g(e):
            return e**(12./19.)/(1. - e**2) * (1. + 121./304. * e**2)**(870./2299.)

        t_coal =  5./256. * a_0**4/sp.m_total()**2 /sp.m_reduced()
        if opt.elliptic:
            t_coal = t_coal * 48./19. / g(e_0)**4 * quad(lambda e: g(e)**4 *(1-e**2)**(5./2.) /e/(1. + 121./304. * e**2), 0., e_0, limit=100)[0]   # The inspiral time according to Maggiore (2007)

        if t_fin is None:
            t_fin = 1.2 * t_coal *( 1. - a_fin**4 / a_0**4)    # This is the time it takes with just gravitational wave emission

        if a_fin == 0.:
            a_fin = sp.r_isco_1()+sp.r_isco_2()    # Stop evolution at r_isco_1()+r_isco_2()

        # set scales to get rescale integration variables
        a_scale = a_0
        t_scale = t_fin

        t_step_max = np.inf
        if opt.verbose > 0:
            print("Evolving from ", a_0/(sp.r_isco_1()+sp.r_isco_2()), " to ", a_fin/(sp.r_isco_1()+sp.r_isco_2()),"r_isco_1+r_isco_2 ", ("with initial eccentricity " + str(e_0)) if opt.elliptic else " on circular orbits", " with ", opt)

        # Define the evolution function
        def dy_dt(t, y, *args):
            sp = args[0]; opt = args[1]
            t = t*t_scale

            # Unpack array
            a, e = y
            a *= a_scale

            if opt.verbose > 1:
                tic = time.perf_counter()

            da_dt, dE_dt = Classic.da_dt(sp, a, e, opt=opt, return_dE_dt=True)
            de_dt = Classic.de_dt(sp, a, e, dE_dt=dE_dt, opt=opt) if opt.elliptic else 0.

            if opt.verbose > 1:
                toc = time.perf_counter()
                print("t=", t, "a=", a, "da/dt=", da_dt, "e=", e, "de/dt=", de_dt, " elapsed real time: ", toc-tic)

            dy = np.array([da_dt/a_scale, de_dt])
            return dy * t_scale

        # Termination condition
        fin_reached = lambda t,y, *args: y[0] - a_fin/a_scale
        fin_reached.terminal = True

        # Initial conditions
        y_0 = np.array([a_0 / a_scale, e_0])

        # Evolve
        tic = time.perf_counter()
        Int = solve_ivp(dy_dt, [t_0/t_scale, (t_0+t_fin)/t_scale], y_0, dense_output=True, args=(sp,opt), events=fin_reached, max_step=t_step_max/t_scale,
                                                                                        method = 'RK45', atol=opt.accuracy, rtol=opt.accuracy)
        toc = time.perf_counter()

        # Collect results
        t = Int.t*t_scale
        a = Int.y[0]*a_scale;
        ev = Classic.EvolutionResults(sp, opt, t, a, msg=Int.message)
        ev.e = Int.y[1] if opt.elliptic else np.zeros(np.shape(ev.t))

        if opt.verbose > 0:
            print(Int.message)
            print(f" -> Evolution took {toc-tic:.4f}s")

        return ev
