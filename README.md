Summary
=======

A plugin for the phy-gui (https://github.com/cortex-lab/phy) spike sorting package. Allows the plotting of 2D ratemaps (think place cells etc), polar head-direction plots and so on.

Longer summary
==============

Assumes that data has been processed using KiloSort (https://github.com/MouseLand/Kilosort) and that there is a numpy (https://numpy.org/) format file that contains x,y data and timestamps (so a 3 x nSamples array.)

Installation
============
git clone or download the SpatialRateMapPlugin.py file into the folder usually located in $HOME/.phy/plugins/

You also need to add a line to $HOME/.phy/phy_config.py which looks like:

>>> c.TemplateGUI.plugins = ['SpatialRateMapPlugin']

Usage
=====
The first time you start up phy with the stuff above done you'll need to go to View->SpatialRateMap or something. Hopefully that'll add the default view of the plugin to the phy-GUI.

It will fail if you don't have a file called 'xy.npy' in the same folder as your KiloSort (KS) session. Specifically, there are a few files from a KS session the plugin needs; spike_times.npy and spike_clusters.npy being the most conspicuous.

Dependencies
============
- astropy - this is the most obvious one; it's used for the spatial autocorrelograms as it handles NaNs in its convolution routines better than scipy/ numpy.

Contributors
============
Robin Hayman


