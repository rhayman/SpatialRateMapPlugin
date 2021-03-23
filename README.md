Summary
=======

A plugin for the phy-gui (https://github.com/cortex-lab/phy) spike sorting package. Allows the plotting of 2D ratemaps (think place cells etc), polar head-direction plots and so on.

Longer summary
==============

Assumes that data has been processed using KiloSort (https://github.com/MouseLand/Kilosort) and that there is a numpy (https://numpy.org/) format file that contains x,y data and timestamps (so a 3 x nSamples array.)

Installation
============
git clone or download the SpatialRateMapPlugin.py file and save in the folder usually located in $HOME/.phy/plugins/

You also need to add a line to $HOME/.phy/phy_config.py which looks like:

`c.TemplateGUI.plugins = ['SpatialRateMapPlugin']`

Usage
=====
The first time you start up phy with the stuff above done you'll need to go to View->SpatialRateMap or something. Hopefully that'll add the default view of the plugin to the phy-GUI.

It will fail if you don't have a file called 'xy.npy' in the same folder as your KiloSort (KS) session. Specifically, there are a few files from a KS session the plugin needs; spike_times.npy and spike_clusters.npy being the most conspicuous.

As mentioned above the xy.npy file that contains the positional data is a 3 x nSamples sized array. I've used positional data collected using a position tracker plugin I wrote for open-ephys (https://github.com/rhayman/PosTracker) which has a sample rate of ~30Hz as I've been using off-the-shelf webcams (I'm cheap). There is a variable called something like pos_sample_rate which has a default value of 30 for that reason but it shouldn't matter if that changes. Indeed it should have no impact in terms of what this plugin does. Regardless, you can change this if you want to (see the __init__ method of the plugin)

xy.npy - a 3 x m numpy.array where column 0 is x, column 1 is y and column 2 is timestamps IN SAMPLES. The timestamps are converted to seconds internally so make sure you save them in the xy.npy file as samples.

There are a few menu entries that I'm enhancing currently (23/03/21) to allow things like different numbers of bins, different smoothing (of the pos data and, separately, the ratemaps etc).

Dependencies
============
- astropy (https://www.astropy.org/) - this is the most obvious one; it's used for the spatial autocorrelograms as it handles NaNs in its convolution routines better than scipy/ numpy.

Contributors
============
Robin Hayman


