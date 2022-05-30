Summary
=======

A plugin for the phy-gui (https://github.com/cortex-lab/phy) spike sorting package. Allows the plotting of 2D ratemaps (think place cells etc), polar head-direction plots and so on.

Longer summary
==============

Assumes that data has been processed using KiloSort (https://github.com/MouseLand/Kilosort) and that there is a numpy (https://numpy.org/) format file that contains x/y data (so a 2 x nSamples array.)

Installation
============
First of all install a Python packge available on PyPi called ephysiopy.

Activate the conda environment as you would if starting up phy and then do this:

```python
python3 -m pip install ephysiopy -U
```

git clone or download the SpatialRateMapPlugin.py file and save in the folder usually located in $HOME/.phy/plugins/

You also need to add a line to $HOME/.phy/phy_config.py which looks like:

`c.TemplateGUI.plugins = ['SpatialRateMapPlugin']`

Now your phy_config.py file should look something like this:

```python
# You can also put your plugins in ~/.phy/plugins/.

from phy import IPlugin

# Plugin example:
#
# class MyPlugin(IPlugin):
#     def attach_to_cli(self, cli):
#         # you can create phy subcommands here with click
#         pass

c = get_config()
c.Plugins.dirs = [r'/home/robin/.phy/plugins']
c.TemplateGUI.plugins = ['SpatialRateMapPlugin']
```

Usage
=====
The most important thing the plugin needs (beyond ephysiopy) is a folder called "pos_data", **in the same place you've loaded phy from i.e where the spike_times.npy and spike_clusters.npy files are located from running KiloSort**. The pos_data folder must contain two files, one called data_array.npy and the other called timestamps.npy. They should look like this:

data_array.npy - a 2 x m array where column 0 is x and column 1 is y.

timestamps.npy - a vector of timestamps, again in samples, that matches the length of the pos samples in data_array (i.e. m samples long in the above convention)

I've used positional data collected using a position tracker plugin I wrote for open-ephys (https://github.com/rhayman/PosTracker) which has a sample rate of ~30Hz as I've been using off-the-shelf webcams (I'm cheap). There is a variable called something like pos_sample_rate which has a default value of 30 for that reason but it shouldn't matter if that changes. Indeed it should have no impact in terms of what this plugin does. Regardless, you can change this if you want to (see the __init__ method of the plugin)

The first time you start up phy with the stuff above done you'll need to go to View->SpatialRateMap or something. Hopefully that'll add the default view of the plugin to the phy-GUI.


Dependencies
============
- astropy (https://www.astropy.org/) - this is the most obvious one; it's used for the spatial autocorrelograms as it handles NaNs in its convolution routines better than scipy/ numpy.

Contributors
============
Robin Hayman


