# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:08:20 2021

@author: jsantne
"""
import cantera


gas = cantera.Solution('h2_burke2012.cti')
f = cantera.FreeFlame(gas)

f.inlet.T = 300
f.inlet.X = {'H2': 1, 'O2': 0.5, 'N2': 0.5*3.76}
f.P = cantera.one_atm
f.set_refine_criteria(slope=0.85, curve=0.99, prune=0.01, ratio=2)
f.solve(loglevel=0, refine_grid=True, auto=True)

print('Complete! flame speed is {:.2f} m/s'.format(f.velocity[0]))
print(len(f.grid))
print(f.get_refine_criteria())

array = f.to_solution_array()

# create new flame and restore from solution array.

new_flame = cantera.FreeFlame(gas)
print(new_flame.get_refine_criteria())
# new_flame.set_initial_guess(data=array)
new_flame.from_solution_array(array)
print('Finished loading solution array')
print(new_flame.get_refine_criteria())
new_flame.inlet.T = 300
new_flame.inlet.X = {'H2': 1, 'O2': 0.5, 'N2': 0.5*3.76}
new_flame.P = cantera.one_atm
# new_flame.set_refine_criteria(slope=0.1, curve=0.1, prune=0.01, ratio=2)
new_flame.solve(loglevel=0, refine_grid=True, auto=True)
print('Complete! flame speed is {:.2f} m/s'.format(new_flame.velocity[0]))
print(len(new_flame.grid))
print(new_flame.get_refine_criteria())