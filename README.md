# SMBHBpy

## General information
This python code allows the simulation of individual inspirals of supermassive black holes (SMBHs) in binary systems by gravitational wave (GW) emission and dynamical friction (DF) due to the interaction with dark matter (DM) spikes. The mass ratio, the parameters of the density distributions, and eccentricity can be chosen arbitrarily. In addition, the code computes the gravitational waveform for both circular and eccentric orbits.

The SMBHBpy code is an extension of the IMRIpy code by my colleague Niklas Becker. See [here](https://github.com/DMGW-Goethe/imripy/tree/main).

## How the code works
In order to calculate the orbital evolution of SMBH binaries, the code requires the corresponding equation for the energy and angular momentum loss for each effect through which the system loses energy (GW emission, DF). If the corresponding energy and angular momentum balance is then established, the coupled differential equations for the oribit parameters $e$ and $a$ can be solved numerically. In addition, the code works in geometrized units with $c=G=1$.

## Usage
See "BasicExample-Plots.ipynb".

## Install
Clone the repository and run.

## License
See LICENSE File.
