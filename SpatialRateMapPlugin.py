from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
from phy.apps import capture_exceptions
import numpy as np
import os
from astropy.convolution import convolve # deals with nans unlike other convs
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from ephysiopy.visualise.plotting import FigureMaker
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
        assert os.path.exists(os.path.join(os.getcwd(), 'xy.npy'))
        self.xy = np.load(os.path.join(os.getcwd(), 'xy.npy'))
        xyts = self.xy[:,2] / 3e4
        xy_lower = np.min(self.xy, 0)
        xy_upper = np.max(self.xy, 0)
        self.xlims = [xy_lower[0], xy_upper[0]]
        self.ylims = [xy_lower[1], xy_upper[1]]
        self.pixels_per_metre = 400
        self.jumpmax = 100
        # Need to do some pre-processing of position data before all this...
        posProcessor = PosCalcsGeneric(self.xy[:,0], self.xy[:,1], self.pixels_per_metre, jumpmax=self.jumpmax)
        # If the spatial extent of xy data is provided in the xy file specify here
        # otherwise the the range of the xy data is used
        xy, hdir = posProcessor.postprocesspos(dict())
        setattr(self, 'dir', hdir)
        setattr(self, 'speed', posProcessor.speed)
        setattr(self, 'ppm', self.pixels_per_metre)
        setattr(self, 'xyTS', xyts)
        setattr(self, 'pos_sample_rate', 1.0/np.mean(np.diff(xyts)))
        spk_times = np.squeeze(np.load(os.path.join(os.getcwd(), 'spike_times.npy')))
        spk_times = spk_times / 3e4
        setattr(self, 'spk_times', spk_times)
        clusters = np.load(os.path.join(os.getcwd(), 'spike_clusters.npy'))
        setattr(self, 'clusters', clusters)
        # start out with 2D ratemaps as the default plot type
        self.plot_type = "ratemap"
        F = FigureMaker()
        setattr(F, 'xy', xy)
        setattr(F, 'dir', hdir)
        setattr(F, 'speed', posProcessor.speed)
        setattr(F, 'xyTS', xyts)
        setattr(F, 'pos_sample_rate', 1.0/np.mean(np.diff(xyts)))
        setattr(self, 'FigureMaker', F)

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return
        if 'ratemap' in self.plot_type:
            self.plotRateMap()
        elif 'head_direction' in self.plot_type:
            self.plotHeadDirection()
        elif 'spikes_on_path' in self.plot_type:
            self.plotSpikesOnPath()

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
        self.actions.add(callback=self.plotHeadDirection, name="head_direction", menu="Test", view=self, show_shortcut=False)

    def plotSpikesOnPath(self):
        self.canvas.ax.clear()
        spk_times = self.spk_times[self.clusters == self.cluster_ids[0]]
        self.FigureMaker.makeSpikePathPlot(spk_times, self.canvas.ax, markersize=3)
        self.plot_type = "spikes_on_path"
        self.canvas.update()

    def plotHeadDirection(self):
        self.canvas.ax.clear()
        spk_times = self.spk_times[self.clusters == self.cluster_ids[0]]
        self.FigureMaker.makeHDPlot(spk_times, self.canvas.ax)
        self.plot_type = "head_direction"
        self.canvas.update()

    def plotRateMap(self):
        self.canvas.ax.clear()
        spk_times = self.spk_times[self.clusters == self.cluster_ids[0]]
        self.FigureMaker.makeRateMap(spk_times, self.canvas.ax)
        self.plot_type = "ratemap"
        self.canvas.update()


class SpatialRateMapPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_feature_density_view():
            """A function that creates and returns a view."""
            return SpatialRateMap(features=controller._get_features)
        controller.view_creator['SpatialRateMap'] = create_feature_density_view
