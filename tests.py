# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:58:28 2021

@author: jsantne
"""
import cantera
import numpy as np
import matplotlib.pyplot as plt
import T_dependent_sensitivity as sens
import time

def test_plot_rates():
    gas = cantera.Solution('h2_burke2012.cti')
    i = 3
    gas = sens.add_perturbed_rxn(gas, i)
    i_pert = len(gas.reactions()) - 1

    width = 50
    mag = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='1000/T', ylabel='k', yscale='log',
                         title=gas.reaction(i).equation)
    T_plot = np.linspace(800, 2200, 500)

    # Plot unperturbed rate
    unp_rate = []
    true_rate = []
    for Temp in T_plot:
        gas.TP = (Temp, 101325)
        unp_rate.append(gas.forward_rate_constants[i] + gas.forward_rate_constants[i_pert])
        true_rate.append(gas.forward_rate_constants[i])
    ax.plot([1000/x for x in T_plot], unp_rate, ls='-', marker='x', label='Unperturbed')

    for T in (1000, 1500, 2000):
        rate = []
        gas = sens.perturb_reaction(gas, T, width, mag, i)
        for Temp in T_plot:
            gas.TP = (Temp, 101325)
            rate.append(gas.forward_rate_constants[i] + gas.forward_rate_constants[i_pert])
        ax.plot([1000/x for x in T_plot], rate, ls='-', marker='', label=str(T))
    fig.legend()
    fig.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111,xlabel='1000/T', ylabel='Unperturbed / True unperturbed')
    ax2.plot([1000/x for x in T_plot], [x / y for x, y in zip(unp_rate, true_rate)], ls='-', marker='')
    fig2.show()


def test_sensitivity():
    "Recreate something like figure 3 in Zhao, Li, Kazakov, Dryer paper"
    mixture = {'H2': 1, 'O2': 0.5, 'N2': 0.5 * 3.76}
    chemfile = 'h2_li_19.cti'
    rxn_num = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='T [K]', ylabel='Sensitivity')

    fmt = 'Width={:.0f} K, magnitude = {:.2f}'.format
    # for width, mag in ((100, 0.01), (100, 0.05), (10, 0.05), (100, 0.5), (500, 0.05)):
    start = time.time()
    for width, mag in ((100, 0.05),):
        print(width, mag)
        sensitivity = sens.sensitivity(mixture, 298, 1, chemfile, rxn_num,
                                       mingrid=200, loglevel=2, resolution=25,
                                       width=width, mag=mag)
        ax.plot(sensitivity[:, 0], sensitivity[:, 1], ls='-', marker='', label=fmt(width, mag))
    plt.legend()
    print('This took {:.0f} seconds'.format(time.time() - start))
# test_plot_rates()
test_sensitivity()