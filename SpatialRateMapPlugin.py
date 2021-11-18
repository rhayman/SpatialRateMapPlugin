import logging
from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
# from phy.apps import capture_exceptions
from phy.utils import selected_cluster_color
import numpy as np
import os
from pathlib import Path, PurePath
from ephysiopy.openephys2py.OEKiloPhy import OpenEphysBinary, OpenEphysNPX
# Suppress warnings generated from doing the ffts for the spatial autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
logger = logging.getLogger("phy")


def fileContainsString(pname: str, searchStr: str) -> bool:
    if os.path.exists(pname):
        with open(pname, 'r') as f:
            strs = f.read()
        lines = strs.split('\n')
        found = False
        for line in lines:
            if searchStr in line:
                found = True
        return found
    else:
        return False


def do_path_walk(pname: Path):
    import os
    import re
    APdata_match = re.compile('Rhythm_FPGA-[0-9][0-9][0-9].0')
    LFPdata_match = re.compile('Rhythm_FPGA-[0-9][0-9][0-9].1')
    PosTracker_match = re.compile('Pos_Tracker-[0-9][0-9][0-9].[0-9]/BINARY_group_[0-9]')

    print(f"Doing walk on {pname}")
    for d, c, f in os.walk(pname):
        for ff in f:
            if '.' not in c:  # ignore hidden directories
                if 'data_array.npy' in ff:
                    if PurePath(d).match('*Pos_Tracker*/BINARY_group*'):
                        path2PosData = os.path.join(d)
                        print(f"Found pos data at: {path2PosData}")
                if 'continuous.dat' in ff:
                    if APdata_match.search(d):
                        path2APdata = os.path.join(d)
                        print(f"Found continuous data at: {path2APdata}")
                    if LFPdata_match.search(d):
                        path2LFPdata = os.path.join(d)
                        print(f"Found continuous data at: {path2LFPdata}")
                if 'sync_messages.txt' in ff:
                    sync_file = os.path.join(
                        d, 'sync_messages.txt')
                    if fileContainsString(sync_file, 'Processor'):
                        sync_message_file = sync_file
                        print(f"Found sync_messages file at: {sync_file}")

class SpatialRateMap(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

    def __init__(self, features=None):
        """features is a function (cluster_id => Bunch(spike_times, ...)) where data is a 3D array."""
        super(SpatialRateMap, self).__init__()
        self.features = features
        # do this for now - maybe give loading option in future
        this_folder = os.getcwd()
        path_to_top_folder = Path(this_folder).parents[4]
        npx = OpenEphysNPX(path_to_top_folder)
        setattr(npx, 'ppm', 400)
        setattr(npx, 'cmsPerBin', 3)
        setattr(npx, 'nchannels', 32)
        npx.load()
        setattr(npx, 'pos_sample_rate', 1.0/np.mean(np.diff(npx.xyTS)))
        setattr(self, 'plot_type', 'ratemap')
        x_lims = (np.nanmin(npx.xy[0]), np.nanmax(npx.xy[0]))
        y_lims = (np.nanmin(npx.xy[1]), np.nanmax(npx.xy[1]))
        setattr(npx, 'x_lims', x_lims)
        setattr(npx, 'y_lims', y_lims)
        setattr(self, 'npx', npx)
        self.overlay_spikes = False

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return
        self.replot()

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

        self.actions.add(callback=self.plotSpikesOnPath, name="spikes_on_path", menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotRateMap, name="ratemap", menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotHeadDirection, name="Head direction(x) by speed(y)", menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotSAC, name="SAC", menu="Test", view=self, show_shortcut=False)
        self.actions.separator()
        self.actions.add(callback=self.setPPM, name='Set pixels per metre', prompt=True, prompt_default=lambda: self.npx.ppm)
        self.actions.add(callback=self.setCmsPerBin, name='Set cms per bin', prompt=True, n_args=1, prompt_default=lambda: self.npx.cmsPerBin)
        self.actions.add(callback=self.speedFilter, name='Filter speed (min max) cm/s', prompt=True, n_args=2)
        self.actions.add(callback=self.directionFilter, name='Filter direction ("w", "e", "n" or "s")', prompt=True, n_args=1)
        self.actions.add(callback=self.timeFilter, name='Filter times(s) (start -> stop)', prompt=True, n_args=2)
        self.actions.add(callback=self.overlaySpikes, name='Overlay spikes', checkable=True, checked=False)

    def replot(self, plot2do='ratemap'):
        if hasattr(self, 'plot_type'):
            plot2do = getattr(self, 'plot_type')
        if 'ratemap' in plot2do:
            self.plotRateMap()
        elif 'head_direction' in plot2do:
            self.plotHeadDirection()
        elif 'spikes_on_path' in plot2do:
            self.plotSpikesOnPath()
        elif 'SAC' in plot2do:
            self.plotSAC()

    def get_spike_times(self, id: int):
        b = self.features(id, load_all=True)
        return np.array(b.data * 3e4).astype(int)

    def setCmsPerBin(self, cms_per_bin: int):
        self.npx.cmsPerBin = cms_per_bin
        self.replot()

    def setPPM(self, ppm: int):
        self.npx.ppm = ppm
        self.npx.x_lims = None
        self.npx.y_lims = None
        self.replot()
    
    def overlaySpikes(self, checked: bool):
        self.overlay_spikes = checked

    def speedFilter(self, _min: int, _max: int):
        if not _min or not _max:
            d = None
        else:
            d = {'speed': [_min, _max]}
        self.npx.filterPosition(d)
        self.replot()

    def directionFilter(self, dir2filt: str):
        if not dir2filt:
            d = None
        else:
            d = {'dir': dir2filt}
        self.npx.filterPosition(d)
        self.replot()

    def timeFilter(self, start: int, stop: int):
        d = {'time' : (start, stop)}
        self.npx.filterPosition(d)

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
            self.canvas.update()
        self.plot_type = "spikes_on_path"
        

    def plotHeadDirection(self):
        self.canvas.ax.clear()
        spk_times = self.get_spike_times(self.cluster_ids[0])
        print(f"npx speed masked: {np.ma.is_masked(self.npx.speed)}")
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
        # ----------- TEMP CODE FOR TEXT ANNOTATION DEBUG ----------
        self.npx.initialise()
        spk_times_in_pos_samples = self.npx.getSpikePosIndices(spk_times)
        spk_weights = np.bincount(
            spk_times_in_pos_samples, minlength=self.npx.npos)
        rmap = self.npx.RateMapMaker.getMap(spk_weights)
        from ephysiopy.common import gridcell
        S = gridcell.SAC()
        nodwell = ~np.isfinite(rmap[0])
        sac = S.autoCorr2D(rmap[0], nodwell)
        measures = S.getMeasures(sac)
        gs = measures['gridscore']
        if ~np.isnan(gs):
            gs = str(gs)[0:5]
        else:
            gs = 'NaN'
        self.canvas.ax.text(
            0.95, 0.05, 
            gs,
            c='w', fontsize=12,
            ha='center', va='top',
            transform=self.canvas.ax.transAxes
        )
         # ----------- END TEXT ANNOTATION DEBUG ----------
        self.plot_type = "SAC"
        self.canvas.update()

class SpatialRateMapPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_ratemap_view():
            """A function that creates and returns a view."""
            return SpatialRateMap(features=controller._get_feature_view_spike_times)
        controller.view_creator['SpatialRateMap'] = create_ratemap_view
