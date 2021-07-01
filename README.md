# temperature-window-sensitivity
Find the sensitivity of flame speed to reaction rates in narrow temperature windows.

This code uses Chebyshev reactions to perturb a reaction in a narrow temperature window in order to determine the sensitivity of flame speed to a reaction rate *at a particular temperature.* This type of analysis was introduced by [Zhao et al.](http://dx.doi.org/10.1002/kin.20080), and I have adapted it to cantera in this repository.

The [Demonstration Figures](https://github.com/CSULA-Combustion-Lab/temperature-window-sensitivity/tree/main/Demonstration%20Figures) folder includes figures that are analogous to Zhao et al., in order to demonstrate the outputs of this code. tests.py includes the code for generating these figures, as well as other examples and tests.

# Limitations
* Currently, this software operates using a "brute force" method, so it is slow. I have only tested it with a small H2 mechanism, where calculation time is reasonable. But, I expect calculations to be very slow with larger mechanisms, or where more accuracy is needed.

* The software uses a Chebyshev reaction to perturb reactions. This perturbation is similar to a gaussian perturbation (see the demonstration figures). A modification of cantera itself would be needed to use a gaussian perturbation instead.

* This software only works with the ElementaryReaction type. It cannot be used with ThreeBodyReaction or FalloffReaction types. This is a major limitation of this code.
