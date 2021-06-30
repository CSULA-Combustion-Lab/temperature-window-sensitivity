# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:58:28 2021

@author: jsantne
"""
import cantera
import os
import numpy as np
import matplotlib.pyplot as plt
import T_dependent_sensitivity as sens
import time

def test_plot_rates():
    gas = cantera.Solution('h2_burke2012.cti')
    i = 20
    nums = sens.duplicate_reactions(gas, i)
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
        rates = gas.forward_rate_constants
        rate = sum([rates[j] for j in nums])
        unp_rate.append(rate + rates[i_pert])
        true_rate.append(rate)
    ax.plot([1000/x for x in T_plot], unp_rate, ls='-', marker='x', label='Unperturbed')

    for T in (1000, 1500, 2000):
        rate = []
        gas = sens.perturb_reaction(gas, T, width, mag, nums)
        for Temp in T_plot:
            gas.TP = (Temp, 101325)
            rates = gas.forward_rate_constants
            temp_rate = sum([rates[j] for j in nums])
            rate.append(temp_rate + rates[i_pert])
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
    ax = fig.add_subplot(111, xlabel='T [K]', ylabel='Sensitivity [$K^{-1}$]',
                         title='Analogous to Zhao et al. Figure 3')

    fmt = 'Width={:.0f} K, magnitude = {:.3f}'.format
    start = time.time()
    for width, mag in ((2500, 0.05), (500, 0.05), (100, 0.05), (50, 0.05), (100, 1), (100, 0.005)):
        # From extensive testing, the best conditions seem to be:
        # Width = 50 to 200
        # Magnitude = 0.01 - 1
        sensitivity, _ = sens.sensitivity(
            mixture, 298, 1, chemfile, rxn_num, mingrid=200, loglevel=1,
            resolution=100, width=width, mag=mag, parallel=True, timeout=10)
        ax.plot(sensitivity[:, 0], sensitivity[:, 1], ls='-', marker='', label=fmt(width, mag))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('Demonstration Figures', 'Zhao Fig 3.png'))
    plt.close(fig)
    print('This took {:.0f} seconds'.format(time.time() - start))


def Zhao_fig_five():
    """ Recreate Figure 5 in Zhao, Li, Kazakov, Dryer paper.

    for plotting example, see https://matplotlib.org/stable/gallery/lines_bars_and_markers/hat_graph.html#sphx-glr-gallery-lines-bars-and-markers-hat-graph-py"""
    chemfile = 'h2_li_19.cti'
    rxn_num = 0
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
    axes[0].set_ylabel('Temperature[K]')
    axes[0].set_ylim([300, 2700])
    axes[0].set_yticks(np.arange(300, 3000, 300))
    # spacing = 0.3
    # width = (1 - spacing) / 5

    for P, ax in zip((1, 10, 40), axes):
        ax.set_xticks([0.5, 1, 2, 3, 4])
        ax.set_xlabel(r'$\phi$')
        for phi in (0.5, 1, 2, 3, 4):
            mixture = {'H2': phi, 'O2': 0.5, 'N2': 0.5 * 3.76}
            _, window = sens.sensitivity(
                mixture, 300, P, chemfile, rxn_num, mingrid=200, loglevel=1,
                resolution=60, parallel=True, timeout=60)
            Tu, TL, Tm, TH, Tad = window
            ax.plot([phi, phi], [Tu, Tad], ls='--', marker='')
            ax.bar(phi, TH - TL, width=0.2, bottom=TL)
            ax.plot([phi-0.1, phi+0.1], [Tm, Tm], ls='-', marker='x', color='k')
            ax.set_title('P = {} atm'.format(P))
    fig.tight_layout()
    plt.savefig(os.path.join('Demonstration Figures', 'Zhao Fig 5.png'))
    plt.close(fig)


def compare_perturbation_shapes():
    """ Compare the perturbation to a gaussian, and make sure the shape doesn't
    change with temperature shifts.

    The shape widens slightly at lower T."""
    gas = cantera.Solution('h2_burke2012.cti')
    i = 0
    nums = sens.duplicate_reactions(gas, i)
    gas = sens.add_perturbed_rxn(gas, i)
    i_pert = len(gas.reactions()) - 1

    mag = 1

    for width in (10, 100, 500):
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=r'$T - T_c$',
                             xlim=[-1*width, 2*width],
                             title='Width = {}\nmagnitude = {}'.format(width, mag))
        T_plot = np.linspace(400, 2100, 5000)

        for T in (500, 1000, 1500, 2000):
            rate = []
            gas = sens.perturb_reaction(gas, T, width, mag, nums)
            for Temp in T_plot:
                gas.TP = (Temp, 101325)
                rates = gas.forward_rate_constants
                rate.append(rates[i_pert])
            gas.TP = (T, 101325)
            norm = gas.forward_rate_constants[i]
            ax.plot([x - T for x in T_plot], [k/norm for k in rate], ls='-',
                    marker='', label='k_pert / k($T_c) at T_c$ = {}'.format(T))

        # Compare to gaussian
        sigma = width/10
        x = np.linspace(-2 * width, width, 200)
        y = mag * np.exp(-1 * x**2 / (2 * sigma**2))
        ax.plot(x, y, ls='--', marker='', c='k',
                label=r'Gaussian, $T_\sigma$ = width/10, $T_P/(T_\sigma \sqrt{2\pi}$) = magnitude')

        fig.legend(loc='right')
        fig.tight_layout()
        fig.savefig(os.path.join('Demonstration figures',
                                 'Check shape width={}.png'.format(width)))
        plt.close(fig)


if __name__ == '__main__':
    # test_plot_rates()
    # test_sensitivity()
    # compare_perturbation_shapes()
    Zhao_fig_five()