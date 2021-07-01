# -*- coding: utf-8 -*-
"""
Find the temperature-window sensitivity of flame speed.

Python/cantera version of ideas presented in:
    Zhao, Li, Kazakov, Dryer 2005
    http://dx.doi.org/10.1002/kin.20080
"""

import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import numpy as np
from tqdm import tqdm
import cantera


# Global Variables
PMIN = 0
PMAX = 1e50
WORKINGDIR = r"C:\Users\jsantne\Documents\GitHub\temperature-window-sensitivity\Outputs"


def duplicate_reactions(gas, rxn_num):
    """Find and report duplicate reactions in a model.

    Parameters
    ----------
    gas : object
        Cantera generated gas object created using user provided mechanism
    rxn_num : int
        Check this reaction to see if it is a duplicate.

    Returns
    -------
    dup_rxns : list
        List of reaction numbers for matching duplicate reactions. If rxn_num
        is not a duplicate, then this will just be [rxn_num]
    """


    duplicate_inds = [rxn_num]
    if gas.reaction(rxn_num).duplicate:
        equation = gas.reaction_equation(rxn_num)
        eqns = gas.reaction_equations()
        for i in range(len(eqns)):
            if i != rxn_num:
                if eqns[i] == equation:
                    duplicate_inds.append(i)
        assert len(duplicate_inds) > 1

    return duplicate_inds


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
    if not isinstance(rxn, cantera._cantera.ElementaryReaction):
        print('WARNING: {} is a {}. This code has only been tested on '
              'cantera._cantera.ElementaryReaction. '.format(rxn.equation, type(rxn)))
    if isinstance(rxn, cantera._cantera.ThreeBodyReaction):
        print('WARNING: {} has third-body efficiencies. This code may give '
              'behave strangely with third-body reactions'.format(rxn.equation))
    elif isinstance(rxn, cantera._cantera.FalloffReaction):
        raise TypeError('{} is a FalloffReaction. '
                        'This is not supported.'.format(rxn.equation))
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

    The perturbation is centered at the harmonic mean of the low and high
    temperatures (2/(1/TL + 1/TH)).

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
    rxn_num : list
        List of 0-indexed reaction numbers. Include more than one reaction
        number if they are duplicates.

    Returns
    -------
    gas : cantera.Solution
        Solution object where the perturbation has been applied.

    """
    # These equations ensure that the center of the perturbation is
    # actually at T, and that Tmax - Tmin = 2*width
    Tmax = (T + 2 * width + np.sqrt((T + 2 * width)**2 - 4 * width * T))/2
    Tmin = Tmax - 2 * width

    magnitude = 0
    for num in rxn_num:
        magnitude += gas.reaction(num).rate(T) * mag_factor
    # TODO: How to deal with efficiencies? Is that impossible? Does it matter?
    coeffs = [[0], [0], [-1 * np.log10(magnitude)]]
    i_chebyshev = len(gas.reactions()) - 1
    rxn = gas.reaction(i_chebyshev)
    rxn.set_parameters(Tmin, Tmax, PMIN, PMAX, coeffs)
    gas.modify_reaction(i_chebyshev, rxn)
    return gas


def sensitivity(mixture, T, P, chemfile, rxn_num, loglevel=0, resolution=100,
                width=75, mag=0.05, parallel=True, timeout=10, **kwargs):
    """
    Find the temperature-window sensitivity

    Parameters
    ----------
    mixture : dict
        Mixture dictionary. {'component1': quantity, 'component2': quantity2...}.
    T : float
        Unburned temperature, Kelvin.
    P : float
        Pressure, atm.
    chemfile : str
        Path to the chemistry file.
    rxn_num : int
        0-based index of the reaction of interest.
    loglevel : int, optional
        Controls the amount of detail printed to the screen. Higher values
        print more information. The default is 0.
    resolution : int, optional
        Number of temperatures at which to perform the perturbation.
        The default is 100.
    width : float, optional
        Width of the perturbation (Kelvin). The default is 75.
    mag : float, optional
        Magnitude of the perturbation. The default is 0.05.
    parallel : bool, optional
        Turn on parallel processing. The default is True.
    timeout : float, optional
        Maximum time for each flame simulation, in seconds. The default is 10.
    **kwargs : optional keyword arguments
        These arguments are passed to flame_speed.

    Returns
    -------
    sens_array : array
        Array with two columns - Temperature (K) and Sensitivity (K^-1)
    window : tuple
        Output from window_stats: (Tu, TL, T_max, TH, T_ad)

    """
    log('*******\nParallel: {}\n'
        'Width = {}, magnitude = {}\n'
        '{:.0f} K, {:.1f} atm\n'
        'Modifying reaction {}\n'
        'T-resolution: {}'.format(parallel, width, mag, T, P, rxn_num,
                                        resolution), loglevel)
    log(kwargs, loglevel)

    flame_run_opts = {**kwargs, 'mixture': mixture, 'Tin': T, 'P': P,
                      'loglevel': loglevel-1}

    # Calculate base case using model with blank Chebyshev rate, just in case.
    gas = cantera.Solution(chemfile)
    rxns = duplicate_reactions(gas, rxn_num)
    gas = add_perturbed_rxn(gas, rxn_num)
    su_base, Tad = flame_speed(gas, **flame_run_opts)

    temperatures = np.linspace(T + 100, Tad, resolution)
    # If the perturbation starts closer to T instead of T+100, there is a
    # very strange error that causes the kernel to restart!

    if parallel:
        try:
            __IPYTHON__  # This code is running in an ipython console - serialize!
            parallel = False
            print('Simulations performed in series. Run outside IPython for parallel operation.')
        except NameError:
            pass

    if not parallel:
        speeds = []
        for temperature in temperatures:
            log('\nPerturbation centered at {:.0f} K'.format(temperature), loglevel)
            gas = perturb_reaction(gas, temperature, width, mag, rxns)
            su, _ = flame_speed(gas, **flame_run_opts)
            speeds.append(su)
    elif parallel:
        n_proc = multiprocessing.cpu_count() - 1
        pool = multiprocessing.Pool(n_proc, maxtasksperchild=1)
        arglist = [(T_c, chemfile, rxn_num, width, mag, loglevel - 1, flame_run_opts)
                   for T_c in temperatures]
        result = []
        for arg in arglist:
            abortable_func = partial(abortable_worker, one_sensitivity,
                                     timeout=timeout)
            result.append(pool.apply_async(abortable_func, args=arg))
        pool.close()

        if loglevel < 2: # no other messages being printed
            speeds = []
            for p in tqdm(result):
                speeds.append(p.get())
        else:
            pool.join()
            speeds = [p.get() for p in result]

        # speeds = pool.starmap(one_sensitivity, arglist)

    sens = [((x - su_base) / su_base) / (mag * width) for x in speeds if x is not None]
    T = [t for t, x in zip(temperatures, speeds) if x is not None]
    sens_array = np.array([T, sens]).T
    return sens_array, window_stats(sens_array)


def window_stats(sens):
    """
    Find the 5 temperatures for the temperature window.

    Parameters
    ----------
    sens : array
        Numpy array of temperature, sensitivity

    Returns
    -------
    stats : tuple
        Five values indicating (T_unburned, T_10%_low, T_max, T_10%_high, T_ad)
        where the sensitivity is greater than 10% of its maximum between
        T_10%_low and T_10%_high.

    """
    Tu = sens[0, 0] - 100
    T_ad = sens[-1, 0]
    Smax = sens.max(0)[1]
    T_max = sens[sens.argmax(0)[1], 0]
    inds = np.where(np.abs(sens[:, 1]) > 0.1 * np.abs(Smax))
    TL = sens[inds[0][0], 0]
    TH = sens[inds[0][-1], 0]

    return (Tu, TL, T_max, TH, T_ad)

def one_sensitivity(T_center, chemfile, rxn_num, width, mag, loglevel,
                    flame_run_opts):
    """Calculate perturbed flame speed at one temperature, called from
    sensitivity() in parallel"""
    log('\nPerturbation centered at {:.0f} K'.format(T_center), loglevel)
    gas = cantera.Solution(chemfile)
    rxns = duplicate_reactions(gas, rxn_num)
    gas = add_perturbed_rxn(gas, rxn_num)
    gas = perturb_reaction(gas, T_center, width, mag, rxns)
    su, _ = flame_speed(gas, **flame_run_opts)
    return su


def flame_speed(gas, mixture, Tin, P, workingdir=WORKINGDIR, name=None,
                mingrid=200, loglevel=0, restart=None, mult_soret=False):
    """
    Simulate a flame

    Parameters
    ----------
    gas : cantera.Solution
        cantera.Solution object containing the chemistry information
    mixture : dict
        Mixture dictionary. {'component1': quantity, 'component2': quantity2...}.
    Tin : float
        Inlet temperature, K.
    P : float
        Pressure, atm.
    workingdir : str, optional
        Directory for simulations. The default is WORKINGDIR.
    name : str, optional
        If given, save an xml file with this name. The default is None.
    mingrid : int, optional
        Minimum number of grid points. The default is 200.
    loglevel : int, optional
        Controls the amount of detail printed to the screen. Higher values
        print more information. The default is 0.
    restart : str or tuple, optional
        string: Name of a simulation to restart from. The default is None.
        tuple: (cantera solution array, refine criteria) to restart from
        In my testing with stoichimetric hydrogen/air, restarting made the
        simulations slower. This might not be true for larger models, or
        off-stoichiometric conditions.
    mult_soret : bool, optional
        Turn on multicomponent diffusivity and Soret effect.
        The default is False.loglevel : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    Su : float
        Unburned flame speed, m/s.
    Tad : float
        Adiabatic flame temperature, Kelvin.

    """
    f = cantera.FreeFlame(gas)  # Create flame object
    refine_criteria = {'slope': 0.85, 'curve': 0.99, 'prune': 0.01,
                       'ratio': 2}
#        f.show_solution() #debug

    if loglevel <= 0:
        lower_log = 0
    else:
        lower_log = loglevel - 1

    if restart is not None:
        if isinstance(restart, str):
            log('Restoring from ' + restart, loglevel)
            f.restore(os.path.join(workingdir, restart+'.xml'),
                      loglevel=lower_log)
            refine_criteria = f.get_refine_criteria()
        else:
            data, refine_criteria = restart
            log('Restoring from flame data NOT FROM FILE.', loglevel)
            f.from_solution_array(data)
            log('restored solution with {} grid points'.format(f.flame.n_points), loglevel)

    try:
        f.set_refine_criteria(**refine_criteria)

        f.inlet.T = Tin
        f.inlet.X = mixture
        f.P = P * cantera.one_atm

        f.solve(loglevel=lower_log, refine_grid=True, auto=True)

        # Refine the grid and check for grid independence.
        if mult_soret:
            f.transport_model = 'Multi'  # 'Mix' is default
            f.soret_enabled = True  # False is default
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
            f.save(os.path.join(workingdir, name+'.xml'), loglevel=lower_log)

    except Exception as e:  # Except all errors
        log(e, loglevel)
        return None, None

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

def abortable_worker(func, *args, **kwargs):
    """ Allow parallel processing to timeout.
    Copied from https://stackoverflow.com/questions/29494001/how-can-i-abort-a-task-in-a-multiprocessing-pool-after-a-timeout"""
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        return None
