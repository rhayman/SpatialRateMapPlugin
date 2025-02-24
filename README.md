Summary
=======

A plugin for the phy-gui (https://github.com/cortex-lab/phy) spike sorting package. Allows the plotting of 2D ratemaps (place cells, grid cells etc), polar head-direction plots and a couple more.

Longer summary
==============

Assumes data has been processed using KiloSort (https://github.com/MouseLand/Kilosort) and that there are two additional numpy format files that containing xy data (as an n_samples x 2 array) and positional timestamps (n_samples x 1 array)

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
The most important thing the plugin requires is position data and the corresponding timestamps.

You need to provide the position data and the timestamps yourself. The function that loads the position data and
timestamps is near the top of the SpatialRateMapPlugin.py file and is called load_position_data(). That assumes that
the data is saved in the top level folder; in a folder structure like the following the two files are position_timestamps.npy and
xy_data.npy and are in the folder called EAA-1123947_2025-01-22_11-36-18 (the top level folder):

EAA-1123947_2025-01-22_11-36-18
├── position_timestamps.npy
├── Record Node 104
│   ├── experiment1
│   │   └── recording1
│   │       ├── continuous
│   │       │   └── Acquisition_Board-100.Rhythm Data-A
│   │       │       ├── amplitudes.npy
│   │       │       ├── channel_map.npy
│   │       │       ├── channel_positions.npy
│   │       │       ├── cluster_Amplitude.tsv
│   │       │       ├── cluster_ContamPct.tsv
│   │       │       ├── cluster_group.tsv
│   │       │       ├── cluster_KSLabel.tsv
│   │       │       ├── continuous.dat
│   │       │       ├── params.py
│   │       │       ├── pc_feature_ind.npy
│   │       │       ├── pc_features.npy
│   │       │       ├── phy.log
│   │       │       ├── rez.mat
│   │       │       ├── sample_numbers.npy
│   │       │       ├── similar_templates.npy
│   │       │       ├── spike_clusters.npy
│   │       │       ├── spike_templates.npy
│   │       │       ├── spike_times.npy
│   │       │       ├── template_feature_ind.npy
│   │       │       ├── template_features.npy
│   │       │       ├── templates_ind.npy
│   │       │       ├── templates.npy
│   │       │       ├── temp_wh.dat
│   │       │       ├── timestamps.npy
│   │       │       ├── whitening_mat_inv.npy
│   │       │       └── whitening_mat.npy
│   │       ├── events
│   │       │   ├── Acquisition_Board-100.Rhythm Data-A
│   │       │   │   └── TTL
│   │       │   │       ├── full_words.npy
│   │       │   │       ├── sample_numbers.npy
│   │       │   │       ├── states.npy
│   │       │   │       └── timestamps.npy
│   │       │   ├── MessageCenter
│   │       │   │   ├── sample_numbers.npy
│   │       │   │   ├── text.npy
│   │       │   │   └── timestamps.npy
│   │       │   └── Ripple_Detector-102.Rhythm Data-A
│   │       │       └── TTL
│   │       │           ├── full_words.npy
│   │       │           ├── sample_numbers.npy
│   │       │           ├── states.npy
│   │       │           └── timestamps.npy
│   │       ├── structure.oebin
│   │       └── sync_messages.txt
│   └── settings.xml
└── xy_data.npy


xy_data.npy - an n_samples x 2 array where column 0 is x and column 1 is y.

position_timestamps.npy - a vector of timestamps, again in samples, that matches the length of the pos samples in data_array

The first time you start up phy you'll need to go to View->SpatialRateMap to add the plugin to the GUI.


Contributors
============
Robin Hayman


