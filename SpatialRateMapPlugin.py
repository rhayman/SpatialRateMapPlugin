import logging
import os

import warnings
from pathlib import Path

import numpy as np
from ephysiopy.__about__ import __version__ as ephysiopy_vers
from ephysiopy.io.recording import OpenEphysBase
from ephysiopy.common.ephys_generic import PosCalcsGeneric
from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas

# from phy.apps import capture_exceptions
from phy.utils import selected_cluster_color

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in \
    subtract",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in \
    greater",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in \
    true_divide",
)
logger = logging.getLogger("phy")


def load_position_data() -> tuple:
    """
    Load position data including the timestamps for those positions

    The organisation of the position data should be n_samples x 2 (i.e. x-y)
    The organisation of the timestamps data should be n_samples x 1

    The format of both files should be .npy

    Returns
    -------
    A tuple of xy position (n_samples x 2) and position timestamps (n_samples x 1)

    """
    this_folder = os.getcwd()
    # this should be the main recording folder under openephys recording directory structure:
    path_to_top_folder = Path(this_folder).parents[4]
    # you could define the location of the position and position_timestamps files with
    # respect to this
    xy_data = np.load(path_to_top_folder / Path("xy_data.npy"))
    ts_data = np.load(path_to_top_folder / Path("position_timestamps.npy"))
    assert len(ts_data) == len(xy_data[:, 0])
    return xy_data, ts_data


def fileContainsString(pname: str, searchStr: str) -> bool:
    if os.path.exists(pname):
        with open(pname, "r") as f:
            strs = f.read()
        lines = strs.split("\n")
        found = False
        for line in lines:
            if searchStr in line:
                found = True
        return found
    else:
        return False


class SpatialRateMap(ManualClusteringView):
    #  use matplotlib instead of OpenGL (the default)
    plot_canvas_class = PlotCanvasMpl

    def __init__(self, features=None):
        """features is a function (cluster_id => Bunch(spike_times, ...))
        where data is a 3D array."""
        super(SpatialRateMap, self).__init__()
        self.state_attrs += ("ppm", "binsize", "x_lims", "y_lims")
        self.features = features
        # do this for now - maybe give loading option in future
        print(f"Using ephysiopy version: {ephysiopy_vers}")
        this_folder = os.getcwd()
        path_to_top_folder = Path(this_folder).parents[4]
        OEBase = OpenEphysBase(path_to_top_folder)
        ppm = 800
        setattr(OEBase, "ppm", ppm)
        jumpmax = 100
        # this should be set based on the metadata (settings.xml or structure.oebin)
        # depending on version I think
        setattr(OEBase, "nchannels", 32)
        # try and load the pos data using the built in mechanism for the OpenEphysBase class...
        warnings.filterwarnings(
            "error"
        )  # set this to catch the warning, reset in a bit...
        try:
            OEBase.load_pos_data(ppm, jumpmax, cm=False)
        except UserWarning:  # need to inject the position data into the OEBase instance
            try:
                xy, xy_ts = load_position_data()
                P = PosCalcsGeneric(
                    xy[:, 0], xy[:, 1], cm=True, ppm=ppm, jumpmax=jumpmax
                )
                P.xyTS = xy_ts
                pos_sample_rate = 50
                P.sample_rate = pos_sample_rate
                P.postprocesspos({"SampleRate": pos_sample_rate})
                print("Loaded pos data from user file")
                OEBase.PosCalcs = P
            except Exception as e:
                warnings.warn("Could not load position data")
                print(e)
        # ...reset warning to default
        warnings.resetwarnings()
        OEBase.initialise()
        setattr(OEBase.RateMap, "binsize", 8)
        setattr(self, "plot_type", "ratemap")
        x_lims = (
            np.nanmin(OEBase.PosCalcs.xy[0]).astype(int),
            np.nanmax(OEBase.PosCalcs.xy[0]).astype(int),
        )
        y_lims = (
            np.nanmin(OEBase.PosCalcs.xy[1]).astype(int),
            np.nanmax(OEBase.PosCalcs.xy[1]).astype(int),
        )
        setattr(OEBase, "x_lims", x_lims)
        setattr(OEBase, "y_lims", y_lims)
        setattr(self, "OEBase", OEBase)
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
        - Bind the on_select() method to the select event raised by the
          supervisor.
        """
        super(SpatialRateMap, self).attach(gui)

        self.actions.add(
            callback=self.plotSpikesOnPath,
            name="spikes_on_path",
            menu="Test",
            view=self,
            show_shortcut=False,
        )
        self.actions.add(
            callback=self.plotRateMap,
            name="ratemap",
            menu="Test",
            view=self,
            show_shortcut=False,
        )
        self.actions.add(
            callback=self.plotHeadDirection,
            name="Head direction(x) by speed(y)",
            menu="Test",
            view=self,
            show_shortcut=False,
        )
        self.actions.add(
            callback=self.plotSAC,
            name="SAC",
            menu="Test",
            view=self,
            show_shortcut=False,
        )
        self.actions.separator()
        self.actions.add(
            callback=self.setPPM,
            name="Set pixels per metre",
            prompt=True,
            prompt_default=lambda: self.OEBase.ppm,
        )
        self.actions.add(
            callback=self.setJumpMax,
            name="Max pos jump in pixels",
            prompt=True,
            prompt_default=lambda: self.OEBase.jumpmax,
        )
        self.actions.add(
            callback=self.setbinsize,
            name="Set bin size",
            prompt=True,
            n_args=1,
            prompt_default=lambda: self.OEBase.RateMap.binsize,
        )
        self.actions.add(
            callback=self.setNBins,
            name="Set number of bins",
            prompt=True,
            prompt_default=lambda: str(self.OEBase.RateMap.nBins)
            .strip(")")
            .strip("(")
            .replace(",", ""),
        )
        self.actions.add(
            callback=self.setSmoothSize,
            name="Set smoothing window",
            prompt=True,
            prompt_default=lambda: self.OEBase.RateMap.smooth_sz,
        )
        self.actions.add(
            callback=self.setXLims,
            name="Set x limits",
            prompt=True,
            n_args=2,
            prompt_default=lambda: str(self.OEBase.x_lims)
            .strip(")")
            .strip("(")
            .replace(",", ""),
        )
        self.actions.add(
            callback=self.setYLims,
            name="Set y limits",
            prompt=True,
            n_args=2,
            prompt_default=lambda: str(self.OEBase.y_lims)
            .strip(")")
            .strip("(")
            .replace(",", ""),
        )
        self.actions.add(
            callback=self.speedFilter,
            name="Filter speed (min max) cm/s",
            prompt=True,
            n_args=2,
        )
        self.actions.add(
            callback=self.directionFilter,
            name='Filter direction ("w", "e", "n" or "s")',
            prompt=True,
            n_args=1,
        )
        self.actions.add(
            callback=self.timeFilter,
            name="Filter times(s) (start -> stop)",
            prompt=True,
            n_args=2,
        )
        self.actions.add(
            callback=self.overlaySpikes,
            name="Overlay spikes",
            checkable=True,
            checked=False,
        )

    def replot(self, plot2do="ratemap"):
        if hasattr(self, "plot_type"):
            plot2do = getattr(self, "plot_type")
        if "ratemap" in plot2do:
            self.plotRateMap()
        elif "head_direction" in plot2do:
            self.plotHeadDirection()
        elif "spikes_on_path" in plot2do:
            self.plotSpikesOnPath()
        elif "SAC" in plot2do:
            self.plotSAC()

    def get_spike_times(self, id: int):
        """
        spike times are returned in seconds
        """
        b = self.features(id, load_all=True)
        return np.array(b.data)

    def setbinsize(self, binsz: int):
        setattr(self.OEBase.RateMap, "binsize", binsz)
        self.replot()

    def setNBins(self, bx: int, by: int):
        setattr(self.OEBase.RateMap, "nBins", (bx, by))
        self.replot()

    def setSmoothSize(self, val: int):
        setattr(self.OEBase.RateMap, "smooth_sz)", val)
        self.replot()

    def setPPM(self, ppm: int):
        self.OEBase.ppm = ppm
        setattr(self.OEBase.PosCalcs, "ppm", ppm)
        self.OEBase.x_lims = None
        self.OEBase.y_lims = None
        self.replot()

    def setJumpMax(self, val: int):
        self.OEBase.jumpmax = val
        self.OEBase.loadPos()  # reload pos
        self.replot()

    def setXLims(self, _min: int, _max: int):
        setattr(self.OEBase, "x_lims", (_min, _max))
        self.replot()

    def setYLims(self, _min: int, _max: int):
        setattr(self.OEBase, "y_lims", (_min, _max))
        self.replot()

    def overlaySpikes(self, checked: bool):
        self.overlay_spikes = checked

    def speedFilter(self, _min: int, _max: int):
        if not _min or not _max:
            d = None
        else:
            d = {"speed": [_min, _max]}
        self.OEBase.filterPosition(d)
        self.replot()

    def directionFilter(self, dir2filt: str):
        if not dir2filt:
            d = None
        else:
            d = {"dir": dir2filt}
        self.OEBase.PosCalcs.filterPos(d)
        self.replot()

    def timeFilter(self, start: int, stop: int):
        d = {"time": (start, stop)}
        self.OEBase.PosCalcs.filterPos(d)

    def plotSpikesOnPath(self):
        self.canvas.ax.clear()
        if self.overlay_spikes:
            clusters = self.cluster_ids
        else:
            clusters = [self.cluster_ids[0]]
        for idx, cluster in enumerate(clusters):
            col = selected_cluster_color(idx)[0:3]
            self.OEBase.plot_spike_path(cluster, 1, ax=self.canvas.ax, s=3, c=col)
            self.canvas.update()
        self.plot_type = "spikes_on_path"

    def plotHeadDirection(self):
        self.canvas.ax.clear()
        for cluster in self.cluster_ids:
            self.OEBase.plot_speed_v_hd(cluster, 1, ax=self.canvas.ax)
            self.canvas.ax.set_aspect(10)
            self.canvas.ax.set_xlabel("Heading")
            self.plot_type = "head_direction"
            self.canvas.update()

    def plotRateMap(self):
        self.canvas.ax.clear()
        for cluster in self.cluster_ids:
            self.OEBase.plot_rate_map(cluster, 1, ax=self.canvas.ax)
            self.plot_type = "ratemap"
            self.canvas.update()

    def plotSAC(self):
        self.canvas.ax.clear()
        for cluster in self.cluster_ids:
            self.OEBase.plot_sac(cluster, 1, ax=self.canvas.ax)
            from ephysiopy.common.fieldcalcs import gridness

            sac = self.OEBase.get_grid_map(cluster, 1)
            # sac is an instance of BinnedData
            gs, _, _ = gridness(sac.binned_data[0])
            if ~np.isnan(gs):
                gs = str(gs)[0:5]
            else:
                gs = "NaN"
            self.canvas.ax.text(
                0.95,
                0.05,
                gs,
                c="w",
                fontsize=12,
                ha="center",
                va="top",
                transform=self.canvas.ax.transAxes,
            )
            self.plot_type = "SAC"
            self.canvas.update()


class SpatialRateMapPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_ratemap_view():
            """A function that creates and returns a view."""
            return SpatialRateMap(features=controller._get_feature_view_spike_times)

        controller.view_creator["SpatialRateMap"] = create_ratemap_view
