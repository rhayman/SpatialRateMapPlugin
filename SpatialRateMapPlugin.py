import logging
from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
# from phy.apps import capture_exceptions
from phy.utils import selected_cluster_color
import numpy as np
import os
from pathlib import Path, PurePath
from ephysiopy.openephys2py.OEKiloPhy import OpenEphysNPX
from ephysiopy.__about__ import __version__ as ephysiopy_vers
# Suppress warnings generated from doing the ffts for the
# spatial autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in \
    subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in \
    greater")
warnings.filterwarnings("ignore", message="invalid value encountered in \
    true_divide")
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


def do_path_walk(pname: Path) -> dict:
    import os
    import re
    APdata_match = re.compile('Rhythm_FPGA-[0-9][0-9][0-9].0')
    LFPdata_match = re.compile('Rhythm_FPGA-[0-9][0-9][0-9].1')
    NPX_APdata_match = re.compile('Neuropix-PXI-[0-9][0-9][0-9].0')
    NPX_LFPdata_match = re.compile('Neuropix-PXI-[0-9][0-9][0-9].1')
    PosTracker_str = 'Pos_Tracker-[0-9][0-9][0-9].[0-9]/BINARY_group_[0-9]'
    TrackingPlugin_str = 'Tracking_Port-[0-9][0-9][0-9].0/BINARY_group_[0-9]'

    data_locations_keys = ['path2PosData', 'posDataType', 'path2APdata',
                           'path2LFPdata', 'path2NPXAPdata', 'path2NPXLFPdata',
                           'sync_message_file']
    data_locations = dict.fromkeys(data_locations_keys)

    print(f"Doing walk on {pname}")
    for d, c, f in os.walk(pname):
        for ff in f:
            if '.' not in c:  # ignore hidden directories
                if 'data_array.npy' in ff:
                    if PurePath(d).match(PosTracker_str):
                        data_locations['path2PosData'] = os.path.join(d)
                        data_locations['posDataType'] = 'PosTracker'
                    if PurePath(d).match(TrackingPlugin_str):
                        data_locations['path2PosData'] = os.path.join(d)
                        data_locations['posDataType'] = 'TrackingPlugin'
                if 'continuous.dat' in ff:
                    if APdata_match.search(d):
                        data_locations['path2APdata'] = os.path.join(d)
                    if LFPdata_match.search(d):
                        data_locations['path2LFPdata'] = os.path.join(d)
                    if NPX_APdata_match.search(d):
                        data_locations['path2NPXAPdata'] = os.path.join(d)
                    if NPX_LFPdata_match.search(d):
                        data_locations['path2NPXLFPdata'] = os.path.join(d)
                if 'sync_messages.txt' in ff:
                    sync_file = os.path.join(
                        d, 'sync_messages.txt')
                    if fileContainsString(sync_file, 'Processor'):
                        data_locations['sync_message_file'] = sync_file

    for k in data_locations.keys():
        if data_locations[k] is not None:
            print(f"{k} : {data_locations[k]}")

    return data_locations


class SpatialRateMap(ManualClusteringView):
    #  use matplotlib instead of OpenGL (the default)
    plot_canvas_class = PlotCanvasMpl

    def __init__(self, features=None):
        """features is a function (cluster_id => Bunch(spike_times, ...))
        where data is a 3D array."""
        super(SpatialRateMap, self).__init__()
        self.features = features
        # do this for now - maybe give loading option in future
        print(f"Working with ephysiopy version: {ephysiopy_vers}")
        this_folder = os.getcwd()
        path_to_top_folder = Path(this_folder).parents[3]
        npx = OpenEphysNPX(path_to_top_folder)
        data_locations = do_path_walk(path_to_top_folder)
        if 'path2PosData' in data_locations.keys():
            setattr(npx, 'path2PosData', data_locations['path2PosData'])
        if data_locations['posDataType'] == 'PosTracker':
            setattr(npx, 'pos_timebase', 3e4)
            setattr(npx, 'pos_data_type', 'PosTracker')
        if data_locations['posDataType'] == 'TrackingPlugin':
            setattr(npx, 'pos_timebase', 1e7)
            setattr(npx, 'pos_data_type', 'TrackingPlugin')
        setattr(npx, 'ppm', 400)
        setattr(npx, 'cmsPerBin', 3)
        setattr(npx, 'nchannels', 32)
        npx.load()
        setattr(self, 'plot_type', 'ratemap')
        x_lims = (np.nanmin(npx.xy[0]).astype(int),
                  np.nanmax(npx.xy[0]).astype(int))
        y_lims = (np.nanmin(npx.xy[1]).astype(int),
                  np.nanmax(npx.xy[1]).astype(int))
        setattr(npx, 'x_lims', x_lims)
        setattr(npx, 'y_lims', y_lims)
        setattr(self, 'npx', npx)
        self.overlay_spikes = False
        
        print("---------------------------- DEBUG --------------------")
        print(f"npx pos: {npx.xy[0,1:100]}")
        print(f"(npx xyTS: {npx.xyTS[0:50]}")

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
        - Bind the on_select() method to the select event raised by the
          supervisor.
        """
        super(SpatialRateMap, self).attach(gui)

        self.actions.add(callback=self.plotSpikesOnPath, name="spikes_on_path",
                         menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotRateMap, name="ratemap",
                         menu="Test", view=self, show_shortcut=False)
        self.actions.add(callback=self.plotHeadDirection,
                         name="Head direction(x) by speed(y)", menu="Test",
                         view=self, show_shortcut=False)
        self.actions.add(callback=self.plotSAC, name="SAC", menu="Test",
                         view=self, show_shortcut=False)
        self.actions.separator()
        self.actions.add(callback=self.setPPM, name='Set pixels per metre',
                         prompt=True, prompt_default=lambda: self.npx.ppm)
        self.actions.add(callback=self.setJumpMax,
                         name='Max pos jump in pixels', prompt=True,
                         prompt_default=lambda: self.npx.jumpmax)
        self.actions.add(callback=self.setCmsPerBin, name='Set cms per bin',
                         prompt=True, n_args=1,
                         prompt_default=lambda: self.npx.cmsPerBin)
        self.actions.add(callback=self.setXLims, name='Set x limits',
                         prompt=True, n_args=2,
                         prompt_default=lambda: str(self.npx.x_lims).strip(")").strip("(").replace(",", ""))
        self.actions.add(callback=self.setYLims, name='Set y limits',
                         prompt=True, n_args=2,
                         prompt_default=lambda: str(self.npx.y_lims).strip(")").strip("(").replace(",", ""))
        self.actions.add(callback=self.speedFilter,
                         name='Filter speed (min max) cm/s',
                         prompt=True, n_args=2)
        self.actions.add(callback=self.directionFilter,
                         name='Filter direction ("w", "e", "n" or "s")',
                         prompt=True, n_args=1)
        self.actions.add(callback=self.timeFilter,
                         name='Filter times(s) (start -> stop)',
                         prompt=True, n_args=2)
        self.actions.add(callback=self.overlaySpikes, name='Overlay spikes',
                         checkable=True, checked=False)

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
        return np.array(b.data)

    def setCmsPerBin(self, cms_per_bin: int):
        self.npx.cmsPerBin = cms_per_bin
        self.replot()

    def setPPM(self, ppm: int):
        self.npx.ppm = ppm
        self.npx.x_lims = None
        self.npx.y_lims = None
        self.replot()

    def setJumpMax(self, val: int):
        self.npx.jumpmax = val
        self.npx.loadPos()  # reload pos
        self.replot()

    def setXLims(self, _min: int, _max: int):
        setattr(self.npx, 'x_lims', (_min, _max))
        self.replot()

    def setYLims(self, _min: int, _max: int):
        setattr(self.npx, 'y_lims', (_min, _max))
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
        d = {'time': (start, stop)}
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
        self.plot_type = "SAC"
        self.canvas.update()


class SpatialRateMapPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_ratemap_view():
            """A function that creates and returns a view."""
            return SpatialRateMap(features=controller._get_feature_view_spike_times)
        controller.view_creator['SpatialRateMap'] = create_ratemap_view
