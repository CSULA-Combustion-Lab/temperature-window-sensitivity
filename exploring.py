# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:35:57 2021

@author: jsantne
"""
import cantera
import numpy as np
import matplotlib.pyplot as plt


def calc(gas, temps, rxn=0):
    rate = []
    for T in temps:
        gas.TP = (T, 101325)
        rate.append(gas.forward_rate_constants[rxn])
    return rate


def Tmin_effect():
    gas = cantera.Solution('chem.cti')
    fig = plt.figure()
    coeffs = np.array([[0], [0], [-2]])
    for Tmin in [260, 300, 340]:
        rxn = gas.reaction(0)
        rxn.set_parameters(Tmin, rxn.Tmax, rxn.Pmin, rxn.Pmax, coeffs)
        # rxn.set_parameters(Tmin, rxn.Tmax, rxn.Pmin, rxn.Pmax, rxn.coeffs)
        gas.modify_reaction(0, rxn)
        plt.plot(temps, calc(gas, temps), ls='-', label=Tmin)
        plt.axvline(Tmin, 0, 1e20, ls='--', marker='')
    plt.axvline(350, 0, 1e20, ls='--', marker='')
    plt.legend()


def coeff_effect():
    gas = cantera.Solution('chem.cti')
    fig = plt.figure()
    coeffs = np.array([[1], [0], [0]])
    for i in [1, 2, 3]:
        rxn = gas.reaction(0)
        rxn.set_parameters(rxn.Tmin, rxn.Tmax, rxn.Pmin, rxn.Pmax, 1.1**i * coeffs)
        gas.modify_reaction(0, rxn)
        plt.plot(temps, calc(gas, temps), ls='-', label=i)
    plt.legend()


temps = np.linspace(200, 5000, 500)
# Tmin_effect()
coeff_effect()


""" It seems like this won't work. The chebyshev polynomial doesn't give a reaction rate near zero outside of its T range.
NEVERMIND!  With an odd polynomial order in T, it's ~zero outside the range!

COefficients: First one should move up and down, second controls linear-ish function, third controls parabola-ish
1st coefficient: k = 10^coefficient
2nd coefficient: looks like an exponential rise in rate with T
3rd coefficient: looks like an upside-down parabola or gaussian when third coeff is negative!


If coeffs are [[0], [0], [-x]], then the rate looks like a gaussian with a value between zero and 10^x
Tmin and Tmax do a good job of setting the thickness of the "gaussian"

This is doable!!!!



"""


