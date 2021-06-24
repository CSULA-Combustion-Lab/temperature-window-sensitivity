# -*- coding: utf-8 -*-
"""
Find the temperature-window sensitivity of flame speed.

Python/cantera version of ideas presented in:
    Zhao, Li, Kazakov, Dryer 2005
    http://dx.doi.org/10.1002/kin.20080
"""

import cantera
import numpy as np
import os


# Global Variables
PMIN = 0
PMAX = 1e50
WORKINGDIR = r"C:\Users\jsantne\Documents\GitHub\temperature-window-sensitivity\Outputs"

def add_perturbed_rxn(gas, rxn_num):
    """
    Add a Chebyshev reaction duplicate that will be used to perturb the sims.

    Parameters
    ----------
    gas : cantera.Solution
        Solution object containing the chemistry.
    rxn_num : int
        0-indexed reaction number

    Returns
    -------
    gas : cantera.Solution
        Solution object with the extra reaction
    """
    rxn = gas.reaction(rxn_num)
    rxn.duplicate = True
    new_rxn = cantera.ChebyshevReaction(rxn.reactants, rxn.products)
    new_rxn.set_parameters(100, 3000, PMIN, PMAX, [[-10], [0], [0]])
    new_rxn.duplicate = True
    gas.add_reaction(new_rxn)
    return gas


def perturb_reaction(gas, T, width, mag_factor, rxn_num):
    """
    Perturb the reaction.

    Using the Chebyshev version of rxn_num, add a Gaussian-ish perturbation of
    the given width and magnitude, where the maximum rate of the perturbed
    reaction is (1 + mag_factor) x k_orig(T)

    Parameters
    ----------
    gas : cantera.Solution
        Solution object containing the chemistry.
    T : float
        Temperature (Kelvin) at the center of the perturbation.
    width : float
        Width of the perturbation in Kelvin
    mag_factor : float
        Magnitude of the perturbation
    rxn_num : int
        0-indexed reaction number

    Returns
    -------
    gas : cantera.Solution
        Solution object where the perturbation has been applied.

    """
    Tmin = T - width  # Note: width is not divided by 2. See test plots.
    Tmax = T + width
    magnitude = gas.reaction(rxn_num).rate(T) * mag_factor # TODO: if the reaction is a duplicate, this magnitude should include dupliates.
    # TODO: How to deal with efficiencies? Is that impossible? Does it matter?
    coeffs = [[0], [0], [-1 * np.log10(magnitude)]]
    i_chebyshev = len(gas.reactions()) - 1
    rxn = gas.reaction(i_chebyshev)
    rxn.set_parameters(Tmin, Tmax, PMIN, PMAX, coeffs)
    gas.modify_reaction(i_chebyshev, rxn)
    return gas


def sensitivity(mixture, T, P, chemfile, rxn_num, mingrid=200, loglevel=0,
                mult_soret=False, resolution=100, width=10, mag=0.01,
                workingdir=WORKINGDIR):
    flame_run_opts = {'mingrid': mingrid, 'loglevel': loglevel-1,
                      'mult_soret': mult_soret}
    gas = cantera.Solution(chemfile)
    gas = add_perturbed_rxn(gas, rxn_num)

    su_base, Tad = flame_speed(mixture, P, T, gas, workingdir, **flame_run_opts)

    temperatures = np.linspace(T + 100, Tad, resolution)
    sens = []
    for temperature in temperatures:
        log('\nPerturbation centered at {:.0f} K'.format(temperature), loglevel)
        gas = perturb_reaction(gas, temperature, width, mag, rxn_num)
        su, _ = flame_speed(mixture, P, T, gas, workingdir, **flame_run_opts)
        sens.append(((su - su_base) / su_base) / (mag * width))

    return np.array([temperatures, sens]).T


def flame_speed(mixture, P, Tin, gas, workingdir, name=None, mingrid=200,
                loglevel=0, restart=None, mult_soret=False):
    """


    Parameters
    ----------
    mixture : dict
        DESCRIPTION.
    P : float
        Pressure in atm.
    Tin : TYPE
        DESCRIPTION.
    gas : TYPE
        DESCRIPTION.
    workingdir : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.
    mingrid : TYPE, optional
        DESCRIPTION. The default is 200.
    loglevel : TYPE, optional
        DESCRIPTION. The default is 0.
    restart : TYPE, optional
        DESCRIPTION. The default is None.
    mult_soret : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    # Initial conditions and convergence parameters
    initial_grid = [0, 0.02, 0.04, 0.06, 0.08, 0.5]
    # initial_grid in meters. One point at 40% of the domain is included
    # regardless because that is the default flame location.
    f = cantera.FreeFlame(gas, initial_grid)  # Create flame object
    refine_criteria = {'slope': 0.85, 'curve': 0.99, 'prune': 0.01,
                       'ratio': 2}
#        f.show_solution() #debug

    if restart is not None:
        if type(restart) is str:
            log('Restoring from ' + restart, loglevel)
            f.restore(os.path.join(workingdir, restart+'.xml'),
                      loglevel=loglevel-1)
            refine_criteria = f.get_refine_criteria()
        else:
            data, refine_criteria = restart
            log('Restoring from flame data NOT FROM FILE.', loglevel)
            f.from_solution_array(data)  # I think this uses the restart data as the initial guess
            log('restored solution with {} grid points'.format(f.flame.n_points), loglevel)

    try:
        f.set_refine_criteria(**refine_criteria)

        f.inlet.T = Tin
        f.inlet.X = mixture
        f.P = P * cantera.one_atm

        if mult_soret:
            f.transport_model = 'Multi'  # 'Mix' is default
            f.soret_enabled = True  # False is default
        f.solve(loglevel=loglevel-1, refine_grid=True, auto=True)
        # Refine the grid and check for grid independence.
        f.energy_enabled = True
        _grid_independence(f, mingrid, loglevel)
        log('Finished calculation - S_u =  {:.2f} cm/s'.format(f.velocity[0] * 100),
            loglevel)
        if name:
            try:
                os.remove(
                    os.path.join(workingdir, name+'.xml'))
            except OSError:
                pass
            # Save solution to restart using f.restore()
            f.save(os.path.join(workingdir, name+'.xml'), loglevel=loglevel-1)

    except Exception as e:  # Except all errors
        log(e, loglevel)
        raise

    return f.velocity[0], f.T[-1]


def log(msg, level):
    if level > 0:
        print(msg)


def _grid_independence(flame, mingrid, loglevel=0):
    """ Refine until flame is independent of the grid, and has at least
    mingrid points.
    flame:
        cantera.FreeFlame object
    mingrid:
        minimum number of grid points
    loglevel:
        integer specifying amount of information to print.
    """
    grid = flame.flame.n_points
    speed = flame.velocity[0] / 10  # Make the while loop below run at least once.
    while grid < mingrid or abs(speed-flame.velocity[0])/speed > 0.05:
        speed = flame.velocity[0]  # save previous speed
        flame = _refine(flame, mingrid)  # Adjust refinement params
        msg = ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' +
               '{} grid points, {} needed\n' +
               'refining to slope = {:.3f}, curve = {:.3f}, prune = {:.4f}\n' +
               '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        refine_criteria = flame.get_refine_criteria()
        log(msg.format(flame.flame.n_points, mingrid,
                       refine_criteria['slope'], refine_criteria['curve'],
                       refine_criteria['prune'], ), loglevel - 1)
        flame.solve(loglevel=loglevel - 1, refine_grid=True)

        grid = flame.flame.n_points  # Final number of points

        log('Grid independence? Su = {:.2f} cm/s with {:} points'.format(
                flame.velocity[0]*100, grid), loglevel)

def _refine(fl, mingrid):
        """
        Adjust refinement criteria to get to mingrid points.

        Returns cantera.FreeFlame object
        """
        refine_criteria = fl.get_refine_criteria()
        grid = fl.flame.n_points

        # For example, if you want to double the number of grid
        # points, divide grad and curv by two.
        # If factor > 3, or grid is approaching max (1000),
        # loosen criteria to remove points
        factor = grid / mingrid
        if 2 > factor > 0.7 and grid < 900:
            factor = 0.7  # Do some refining
        elif factor < 0.1:
            factor = 0.1  # Don't refine too much all at once

        # Refine the mesh criteria.
        for key in ['slope', 'curve', 'prune']:
            refine_criteria[key] *= factor
        if factor > 1:
            refine_criteria['prune'] *= factor # Make sure pruning happens??
        max_criteria = max([refine_criteria[key] for key in
                            ['slope', 'curve', 'prune']])
        if max_criteria > 1.0:
            for key in ['slope', 'curve', 'prune']:
                refine_criteria[key] *= 1 / max_criteria

        fl.set_refine_criteria(**refine_criteria)
        return fl
