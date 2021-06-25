from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
from phy.apps import capture_exceptions
from phy.utils import selected_cluster_color
import numpy as np
import os
from pathlib import Path
from astropy.convolution import convolve # deals with nans unlike other convs
from ephysiopy.openephys2py.OEKiloPhy import OpenEphysNPX
# Suppress warnings generated from doing the ffts for the spatial autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

capture_exceptions()

class SpatialRateMap(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

    def __init__(self, features=None):
        """features is a function (cluster_id => Bunch(spike_times, ...)) where data is a 3D array."""
        super(SpatialRateMap, self).__init__()
        self.features = features
        # do this for now - maybe give loading option in future
        this_folder = os.getcwd()
        path_to_top_folder = Path(this_folder).parents[3]
        path2PosData = Path(this_folder).joinpath('pos_data')
        if not os.path.exists(path2PosData):
            return
        npx = OpenEphysNPX(path_to_top_folder)
        setattr(npx, 'ppm', 400)
        setattr(npx, 'cmsPerBin', 3)
        npx.load()
        setattr(npx, 'pos_sample_rate', 1.0/np.mean(np.diff(npx.xyTS)))
        self.plot_type = "ratemap"
        setattr(self, 'npx', npx)

        self.overlay_spikes = False

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return
        self.plot()

        # Use this to update the matplotlib figure.
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI.
        Perform the following:
        - Add the view to the GUI.
        - Update the view's attribute from the GUI state
        - Add the default view actions (auto_update, screenshot)
        - Bind the on_select() method to the select event raised by the supervisor.
        """
        super(SpatialRateMap, self).attach(gui)

        self.actions.add(callback=self.plotRateMap, name="ratemap", menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotSpikesOnPath, name="spikes_on_path", menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotHeadDirection, name="Head direction(x) by speed(y)", menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotSAC, name="SAC", menu="Test", view=self, show_shortcut=False)
        self.actions.separator()
        self.actions.add(callback=self.setPPM, name='Set pixels per metre', prompt=True, prompt_default=lambda: self.npx.ppm)
        self.actions.add(callback=self.setCmsPerBin, name='Set cms per bin', prompt=True, n_args=1, prompt_default=lambda: self.npx.cmsPerBin)
        self.actions.add(callback=self.overlaySpikes, name='Overlay spikes', checkable=True, checked=False)
        self.actions.add(callback=self.speedFilter, name='Filter speed', prompt=True, n_args=2)

    def replot(self):
        if 'ratemap' in self.plot_type:
            self.plotRateMap()
        elif 'head_direction' in self.plot_type:
            self.plotHeadDirection()
        elif 'spikes_on_path' in self.plot_type:
            self.plotSpikesOnPath()
        elif 'SAC' in self.plot_type:
            self.plotSAC()

    def get_spike_times(self, id):
        b = self.features(id, load_all=True)
        return np.array(b.data * 3e4).astype(int)

    def setCmsPerBin(self, cms_per_bin):
        self.npx.cmsPerBin = cms_per_bin
        self.replot()

    def setPPM(self, ppm):
        self.npx.ppm = ppm
        self.replot()
    
    def overlaySpikes(self, checked):
        if checked:
            self.overlay_spikes = True
        else:
            self.overlay_spikes = False

    def speedFilter(self, _min, _max):
        d = {'speed': [_min, _max]}
        self.npx.filterPosition(d)
        self.replot()

    def plotSpikesOnPath(self):
        self.canvas.ax.clear()
        if self.overlay_spikes:
            clusters = self.cluster_ids
        else:
            clusters = [self.cluster_ids[0]]
        for idx, cluster in enumerate(clusters):
            spk_times = self.get_spike_times(cluster)
            col = selected_cluster_color(idx)[0:3]
            self.npx.makeSpikePathPlot(
                spk_times, ax=self.canvas.ax, markersize=3, c=col)
        self.plot_type = "spikes_on_path"
        self.canvas.update()

    def plotHeadDirection(self):
        self.canvas.ax.clear()
        spk_times = self.get_spike_times(self.cluster_ids[0])
        self.npx.makeSpeedVsHeadDirectionPlot(spk_times, self.canvas.ax)
        self.canvas.ax.set_aspect(10)
        self.plot_type = "head_direction"
        self.canvas.update()

    def plotRateMap(self):
        spk_times = self.get_spike_times(self.cluster_ids[0])
        self.canvas.ax.clear()
        self.npx.makeRateMap(spk_times, self.canvas.ax)
        self.plot_type = "ratemap"
        self.canvas.update()

    def plotSAC(self):
        self.canvas.ax.clear()
        spk_times = self.get_spike_times(self.cluster_ids[0])
        self.npx.makeSAC(spk_times, self.canvas.ax)
        self.plot_type = "SAC"
        self.canvas.update()

class SpatialRateMapPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_ratemap_view():
            """A function that creates and returns a view."""
            return SpatialRateMap(features=controller._get_feature_view_spike_times)
        controller.view_creator['SpatialRateMap'] = create_ratemap_view
