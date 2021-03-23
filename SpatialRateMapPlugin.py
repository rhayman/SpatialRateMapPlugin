from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
import numpy as np
import matplotlib.pylab as plt
from matplotlib.cm import jet
from astropy import convolution # deals with nans unlike other convs
# Suppress warnings generated from doing the ffts for the spatial autocorrelogram
# see autoCorr2D and crossCorr2D
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in greater")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

class SpatialRateMap(ManualClusteringView):
	plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

	def __init__(self, features=None):
		"""features is a function (cluster_id => Bunch(spike_times, ...)) where data is a 3D array."""
		super(SpatialRateMap, self).__init__()
		self.features = features
		import os
		# do this for now - maybe give loading option in future
		assert os.path.exists(os.path.join(os.getcwd(), 'xy.npy'))
		import numpy as np
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
		spk_times = np.squeeze(np.load(os.path.join(os.getcwd(), 'spike_times.npy')))
		spk_times = spk_times / 3e4
		self.RateMapMaker = MapCalcsGeneric(xy, np.squeeze(hdir), posProcessor.speed, xyts, spk_times, 'map')
		self.RateMapMaker.good_clusters = None
		if os.path.exists('spike_clusters.npy'):
			self.RateMapMaker.spk_clusters = np.load('spike_clusters.npy')
		else:
			self.RateMapMaker.spk_clusters = None
		# start out with 2D ratemaps as the default plot type
		self.plot_type = "ratemap"

	def on_select(self, cluster_ids=(), **kwargs):
		self.cluster_ids = cluster_ids
		# We don't display anything if no clusters are selected.
		if not cluster_ids:
			return
		if 'ratemap' in self.plot_type:
			self.RateMapMaker.makeRateMap(cluster_ids[0], ax=self.canvas.ax)
		else if 'head_direction' in self.plot_type:
			self.RateMapMaker.makeHDPlot(cluster_ids[0],  ax=self.canvas.ax)
		else if 'spikes_on_path' in self.plot_type:
			self.RateMapMaker.makeSpikePathPlot(cluster_ids[0], ax=self.canvas.ax)

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
		self.actions.add(callback=self.plotHeadDirection, name="head_direction", menu="Test", view=self, show_shortcut=False)

	def plotSpikesOnPath(self):
		self.RateMapMaker.makeSpikePathPlot(self.cluster_ids[0], ax=self.canvas.ax)
		self.plot_type = "spikes_on_path"

	def plotHeadDirection(self):
		self.RateMapMaker.makeHDPlot(self.cluster_ids[0], ax=self.canvas.ax)
		self.plot_type = "head_direction"

	# def plotSpatialAutocorr(self):
	# 	self.RateMapMaker.makeSAC()


class SpatialRateMapPlugin(IPlugin):
	def attach_to_controller(self, controller):
		def create_feature_density_view():
			"""A function that creates and returns a view."""
			return SpatialRateMap(features=controller._get_features)
		controller.view_creator['SpatialRateMap'] = create_feature_density_view
	
class RateMap(object):
	"""
	Bins up positional data (xy, head direction etc) and produces rate maps
	of the relevant kind. This is a generic class meant to be independent of
	any particular recording format

	Parameters
	----------
	xy : array_like, optional
		The xy data, usually given as a 2 x n sample numpy array
	hdir : array_like, optional
		The head direction data, usualy a 1 x n sample numpy array
	speed : array_like, optional
		Similar to hdir
	pos_weights : array_like, optional
		A 1D numpy array n samples long which is used to weight a particular
		position sample when binning data. For example, if there were 5 positions
		recorded and a cell spiked once in position 2 and 5 times in position 3 and
		nothing anywhere else then pos_weights looks like: [0 0 1 5 0]
		In the case of binning up position this will be an array of mostly just 1's
		unless there are some positions you want excluded for some reason
	ppm : int, optional
		Pixels per metre. Specifies how many camera pixels per metre so this,
		in combination with cmsPerBin, will determine how many bins there are
		in the rate map
	xyInCms : bool, optional, default False
		Whether the positional data is in cms
	cmsPerBin : int, optional, default 3
		How many cms on a side each bin is in a rate map OR the number of degrees
		per bin in the case of directional binning
	smooth_sz : int, optional, default = 5
		The width of the smoothing kernel for smoothing rate maps

	Notes
	----
	There are several instance variables you can set, see below
	
	"""
	def __init__(self, xy=None, hdir=None, speed=None, pos_weights=None, ppm=430, xyInCms=False, cmsPerBin=3, smooth_sz=5):
		self.xy = xy
		self.dir = hdir
		self.speed = speed
		self.__pos_weights = pos_weights
		self.__ppm = ppm #pixels per metre
		self.__cmsPerBin = cmsPerBin
		self.__inCms = xyInCms
		self.__binsize__ = None # has setter and getter - see below
		self.__smooth_sz = smooth_sz
		self.__smoothingType = 'gaussian' # 'boxcar' or 'gaussian'
		self.whenToSmooth = 'before' # or 'after'

	@property
	def inCms(self):
		# Whether the units are in cms or not
		return self.__inCms

	@inCms.setter
	def inCms(self, value):
		self.__inCms = value

	@property
	def ppm(self):
		# Get the current pixels per metre (ppm)
		return self.__ppm

	@ppm.setter
	def ppm(self, value):
		self.__ppm = value
		self.__binsize__ = self.__calcBinSize(self.cmsPerBin)

	@property
	def binsize(self):
		# Returns binsize calculated in __calcBinSize and based on cmsPerBin
		if self.__binsize__ is None:
			try:
				self.__binsize__ = self.__calcBinSize(self.cmsPerBin)
			except AttributeError:
				self.__binsize__ = None
		return self.__binsize__

	@binsize.setter
	def binsize(self, value):
		self.__binsize__ = value

	@property
	def pos_weights(self):
		"""
		The 'weights' used as an argument to np.histogram* for binning up position
		Mostly this is just an array of 1's equal to the length of the pos
		data, but usefully can be adjusted when masking data in the trial
		by
		"""
		return self.__pos_weights

	@pos_weights.setter
	def pos_weights(self, value):
		self.__pos_weights = value

	@property
	def cmsPerBin(self):
		# The number of cms per bin of the binned up map
		return self.__cmsPerBin

	@cmsPerBin.setter
	def cmsPerBin(self, value):
		self.__cmsPerBin = value
		self.__binsize__ = self.__calcBinSize(self.cmsPerBin)

	@property
	def smooth_sz(self):
		# The size of the smoothing window applied to the binned data (1D or 2D)
		return self.__smooth_sz

	@smooth_sz.setter
	def smooth_sz(self, value):
		self.__smooth_sz = value

	@property
	def smoothingType(self):
		# The type of smoothing to do - legal values are 'boxcar' or 'gaussian'
		return self.__smoothingType

	@smoothingType.setter
	def smoothingType(self, value):
		self.__smoothingType = value

	@property
	def pixelsPerBin(self):
		# Calculates the number of camera pixels per bin of the binned data
		if getattr(self, 'inCms'):
			return getattr(self, 'cmsPerBin')
		else:
			return (getattr(self, 'ppm') / 100.) * getattr(self, 'cmsPerBin')

	def __calcBinSize(self, cmsPerBin=3):
		"""
		Aims to get the right number of bins for x and y dims given the ppm
		in the set header and the x and y extent

		Parameters
		----------
		cmsPerBin : int, optional, default = 3
			The number of cms per bin OR degrees in the case of directional binning
		"""
		x_lims = (np.min(self.xy[0]), np.max(self.xy[0]))
		y_lims = (np.min(self.xy[1]), np.max(self.xy[1]))
		ppb = getattr(self, 'pixelsPerBin')
		self.binsize = np.array((np.ceil(np.ptp(y_lims) / ppb)-1,
								 np.ceil(np.ptp(x_lims) / ppb)-1), dtype=np.int)
		return self.binsize

	def getMap(self, spkWeights, varType='xy', mapType='rate', smoothing=True):
		"""
		Bins up the variable type varType and returns a tuple of (rmap, binnedPositionDir) or
		(rmap, binnedPostionX, binnedPositionY)

		Parameters
		----------
		spkWeights : array_like
			Shape equal to number of positions samples captured and consists of
			position weights. For example, if there were 5 positions
			recorded and a cell spiked once in position 2 and 5 times in position 3 and
			nothing anywhere else then pos_weights looks like: [0 0 1 5 0]
		varType : str, optional, default 'xy'
			The variable to bin up. Legal values are: 'xy', 'dir', and 'speed'
		mapType : str, optional, default 'rate'
			If 'rate' then the binned up spikes are divided by varType. Otherwise return
			binned up position. Options are 'rate' or 'pos'
		smoothing : bool, optional, default True
			Whether to smooth the data or not

		Returns
		-------
		binned_data, binned_pos : tuple
			This is either a 2-tuple or a 3-tuple depening on whether binned pos
			(mapType is 'pos') or binned spikes (mapType is 'rate') is asked for,
			respectively

		"""
		sample = getattr(self, varType)
		assert(sample is not None) # might happen if head direction not supplied for example

		if 'xy' in varType:
			self.binsize = self.__calcBinSize(self.cmsPerBin)
		elif 'dir' in varType:
			self.binsize = np.arange(0, 360+self.cmsPerBin, self.cmsPerBin)
		elif 'speed' in varType:
			self.binsize = np.arange(0, 50, 1)

		binned_pos = self.__binData(sample, self.binsize, self.pos_weights)

		if binned_pos.ndim == 1: # directional binning
			binned_pos_edges = binned_pos[1]
			binned_pos = binned_pos[0]
		elif binned_pos.ndim == 2:
			binned_pos_edges = (binned_pos[1])
			binned_pos = binned_pos[0]
		elif len(binned_pos) == 3:
			binned_pos_edges = binned_pos[1:]
			binned_pos = binned_pos[0]
		nanIdx = binned_pos == 0

		if 'pos' in mapType: #return just binned up position
			if smoothing:
				if 'dir' in varType:
					binned_pos = self.__circPadSmooth(binned_pos, n=self.smooth_sz)
				else:
					binned_pos = self.blurImage(binned_pos, self.smooth_sz, ftype=self.smoothingType)
			return binned_pos, binned_pos_edges
		binned_spk = self.__binData(sample, self.binsize, spkWeights)[0]
		# binned_spk is returned as a tuple of the binned data and the bin
		# edges
		if 'after' in self.whenToSmooth:
			rmap = binned_spk[0] / binned_pos
			if 'dir' in varType:
				rmap = self.__circPadSmooth(rmap, self.smooth_sz)
			else:
				rmap = self.blurImage(rmap, self.smooth_sz, ftype=self.smoothingType)
		else: # default case
			if not smoothing:
				if len(binned_pos_edges) == 1: #directional map
					return binned_spk / binned_pos, binned_pos_edges
				elif len(binned_pos_edges) == 2:
					if binned_spk.ndim == 3:
						nClusters = spkWeights.shape[0]
						multi_binned_spks = np.zeros([self.binsize[0], self.binsize[1], nClusters])
						for i in range(nClusters):
							multi_binned_spks[:, :, i] = binned_spk[i]
						return multi_binned_spks / binned_pos[:, :, np.newaxis], binned_pos_edges[0], binned_pos_edges[1]
					else:
						return binned_spk / binned_pos, binned_pos_edges[0], binned_pos_edges[1]
			if 'dir' in varType:
				binned_pos = self.__circPadSmooth(binned_pos, self.smooth_sz)
				binned_spk = self.__circPadSmooth(binned_spk, self.smooth_sz)
				if spkWeights.ndim == 1:
					rmap = binned_spk / binned_pos
				elif spkWeights.ndim == 2:
					rmap = np.zeros([spkWeights.shape[0], binned_pos.shape[0]])
					for i in range(spkWeights.shape[0]):
						rmap[i, :] = binned_spk[i] / binned_pos
			else:
				if isinstance(binned_spk.dtype, np.object):
					binned_pos = self.blurImage(binned_pos, self.smooth_sz, ftype=self.smoothingType)
					if binned_spk.ndim == 2:
						pass
					elif (binned_spk.ndim == 3 or binned_spk.ndim == 1):
						binned_spk_tmp = np.zeros([binned_spk.shape[0], binned_spk[0].shape[0], binned_spk[0].shape[1]])
						for i in range(binned_spk.shape[0]):
							binned_spk_tmp[i, :, :] = binned_spk[i]
						binned_spk = binned_spk_tmp
					binned_spk = self.blurImage(binned_spk, self.smooth_sz, ftype=self.smoothingType)
					rmap = binned_spk / binned_pos
					if rmap.ndim <= 2:
						rmap[nanIdx] = np.nan
					elif rmap.ndim == 3:
						rmap[:,nanIdx] = np.nan

		return rmap, binned_pos_edges

	def blurImage(self, im, n, ny=None, ftype='boxcar'):
		"""
		Smooths a 2D image by convolving with a filter

		Parameters
		----------
		im : array_like
			The array to smooth
		n, ny : int
			The size of the smoothing kernel
		ftype : str
			The type of smoothing kernel. Either 'boxcar' or 'gaussian'

		Returns
		-------
		res: array_like
			The smoothed vector with shape the same as im
		"""
		n = int(n)
		if not ny:
			ny = n
		else:
			ny = int(ny)
		#  keep track of nans
		nan_idx = np.isnan(im)
		im[nan_idx] = 0
		from scipy import signal
		g = signal.boxcar(n) / float(n)
		if 'box' in ftype:
			if im.ndim == 1:
				g = signal.boxcar(n) / float(n)
			elif im.ndim == 2:
				g = signal.boxcar(n) / float(n)
				g = np.tile(g, (1, ny, 1))
			elif im.ndim == 3: # mutlidimensional binning
				g = signal.boxcar([n, ny]) / float(n)
				g = g[None, :, :]
		elif 'gaussian' in ftype:
			x, y = np.mgrid[-n:n+1, -ny:ny+1]
			g = np.exp(-(x**2/float(n) + y**2/float(ny)))
			g = g / g.sum()
			if np.ndim(im) == 1:
				g = g[n, :]
		improc = signal.convolve(im, g, mode='same')
		improc[nan_idx] = np.nan
		return improc

	def __binData(self, var, bin_edges, weights):
		"""
		Bins data taking account of possible multi-dimensionality

		Parameters
		----------
		var : array_like
			The variable to bin
		bin_edges : array_like
			The edges of the data - see numpys histogramdd for more
		weights : array_like
			The weights attributed to the samples in var
		
		Returns
		-------
		ndhist : 2-tuple
			Think this always returns a two-tuple of the binned variable and
			the bin edges - need to check to be sure...		

		Notes
		-----
		This breaks compatability with numpys histogramdd
		In the 2d histogram case below I swap the axes around so that x and y
		are binned in the 'normal' format i.e. so x appears horizontally and y
		vertically. 
		Multi-binning issue is dealt with awkwardly through checking
		the dimensionality of the weights array - 'normally' this would be 1 dim
		but when multiple clusters are being binned it will be 2 dim. In that case
		np.apply_along_axis functionality is applied. The spike weights in
		that case might be created like so:

		>>> spk_W = np.zeros(shape=[len(trial.nClusters), trial.npos])
		>>> for i, cluster in enumerate(trial.clusters):
		>>>		x1 = trial.getClusterIdx(cluster)
		>>>		spk_W[i, :] = np.bincount(x1, minlength=trial.npos)

		This can then be fed into this fcn something like so:

		>>> rng = np.array((np.ma.min(trial.POS.xy, 1).data, np.ma.max(rial.POS.xy, 1).data))
		>>> h = __binData(var=trial.POS.xy, bin_edges=np.array([64, 64]), weights=spk_W, rng=rng)

		Returned will be a tuple containing the binned up data and the bin edges for x and y (obv this will be the same for all
		entries of h)
		"""
		if weights is None:
			weights = np.ones_like(var)
		dims = weights.ndim
		orig_dims = weights.ndim
		if (dims == 1 and var.ndim == 1):
			var = var[np.newaxis, :]
			bin_edges = bin_edges[np.newaxis, :]
		elif (dims > 1 and var.ndim == 1):
			var = var[np.newaxis, :]
			bin_edges = bin_edges[np.newaxis, :]
		else:
			var = np.flipud(var)
		ndhist = np.apply_along_axis(lambda x: np.histogramdd(var.T, weights=x, bins=bin_edges), 0, weights.T)
		if ndhist.ndim == 1:
			if var.ndim == 2: # 1-dimenstional spike weights and xy
				return ndhist
		if ndhist.ndim == 2:
			# a single map has been asked for, pos, single map or dir
			return ndhist[0], ndhist[-1][0]
		elif ndhist.ndim == 1:
			if orig_dims == 1: # directional binning
				return ndhist
			# multi-dimensional binning
			result = np.zeros((len(ndhist[0]), ndhist[0][0].shape[0], ndhist[0][0].shape[1]))
			for i in range(len(ndhist)):
				result[i,:,:] = ndhist[0][i]
			return result, ndhist[::-1]


	def __circPadSmooth(self, var, n=3, ny=None):
		"""
		Smooths a vector by convolving with a gaussian
		Mirror reflects the start and end of the vector to
		deal with edge effects

		Parameters
		----------
		var : array_like
			The vector to smooth
		n, ny : int
			Size of the smoothing (sigma in gaussian)

		Returns
		-------
		res : array_like
			The smoothed vector with shape the same as var
		"""
		from scipy import signal
		tn = len(var)
		t2 = int(np.floor(tn / 2))
		var = np.concatenate((var[t2:tn], var, var[0:t2]))
		if ny is None:
			ny = n
		x, y = np.mgrid[-n:n+1, -ny:ny+1]
		g = np.exp(-(x**2/float(n) + y**2/float(ny)))
		if np.ndim(var) == 1:
			g = g[n, :]
		g = g / g.sum()
		improc = signal.convolve(var, g, mode='same')
		improc = improc[tn-t2:tn-t2+tn]
		return improc

	def autoCorr2D(self, A, nodwell, tol=1e-10):
		"""
		Performs a spatial autocorrelation on the array A

		Parameters
		----------
		A : array_like
			Either 2 or 3D. In the former it is simply the binned up ratemap 
			where the two dimensions correspond to x and y. 
			If 3D then the first two dimensions are x
			and y and the third (last dimension) is 'stack' of ratemaps
		nodwell : array_like
			A boolean array corresponding the bins in the ratemap that
			weren't visited. See Notes below.
		tol : float, optional
			Values below this are set to zero to deal with v small values
			thrown up by the fft. Default 1e-10

		Returns
		-------
		sac : array_like
			The spatial autocorrelation in the relevant dimensionality

		Notes
		-----
		The nodwell input can usually be generated by:

		>>> nodwell = ~np.isfinite(A)
		
		"""

		if np.ndim(A) == 2:
			m,n = np.shape(A)
			o = 1
			x = np.reshape(A, (m,n,o))
			nodwell = np.reshape(nodwell, (m,n,o))
		elif np.ndim(A) == 3:
			m,n,o = np.shape(A)
			x = A.copy()
		
		x[nodwell] = 0
		# [Step 1] Obtain FFTs of x, the sum of squares and bins visited
		Fx = np.fft.fft(np.fft.fft(x,2*m-1,axis=0),2*n-1,axis=1)
		FsumOfSquares_x = np.fft.fft(np.fft.fft(np.power(x,2),2*m-1,axis=0),2*n-1,axis=1)
		Fn = np.fft.fft(np.fft.fft(np.invert(nodwell).astype(int),2*m-1,axis=0),2*n-1,axis=1)
		# [Step 2] Multiply the relevant transforms and invert to obtain the
		# equivalent convolutions
		rawCorr = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fx * np.conj(Fx),axis=1),axis=0)),axes=(0,1))
		sums_x = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(np.conj(Fx) * Fn,axis=1),axis=0)),axes=(0,1))
		sumOfSquares_x = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(FsumOfSquares_x),axis=1),axis=0)),axes=(0,1))
		N = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn * np.conj(Fn),axis=1),axis=0)),axes=(0,1))
		# [Step 3] Account for rounding errors.
		rawCorr[np.abs(rawCorr) < tol] = 0
		sums_x[np.abs(sums_x) < tol] = 0
		sumOfSquares_x[np.abs(sumOfSquares_x) < tol] = 0
		N = np.round(N)
		N[N<=1] = np.nan
		# [Step 4] Compute correlation matrix
		mapStd = np.sqrt((sumOfSquares_x * N) - sums_x**2)
		mapCovar = (rawCorr * N) - sums_x * sums_x[::-1,:,:][:,::-1,:][:,:,:]

		return np.squeeze(mapCovar / mapStd / mapStd[::-1,:,:][:,::-1,:][:,:,:])

	def crossCorr2D(self, A, B, A_nodwell, B_nodwell, tol=1e-10):
		"""
		Performs a spatial crosscorrelation between the arrays A and B

		Parameters
		----------
		A, B : array_like
			Either 2 or 3D. In the former it is simply the binned up ratemap 
			where the two dimensions correspond to x and y. 
			If 3D then the first two dimensions are x
			and y and the third (last dimension) is 'stack' of ratemaps
		nodwell_A, nodwell_B : array_like
			A boolean array corresponding the bins in the ratemap that
			weren't visited. See Notes below.
		tol : float, optional
			Values below this are set to zero to deal with v small values
			thrown up by the fft. Default 1e-10

		Returns
		-------

		sac : array_like
			The spatial crosscorrelation in the relevant dimensionality

		Notes
		-----
		The nodwell input can usually be generated by:

		>>> nodwell = ~np.isfinite(A)
		"""
		if np.ndim(A) != np.ndim(B):
			raise ValueError('Both arrays must have the same dimensionality')
		if np.ndim(A) == 2:
			ma, na = np.shape(A)
			mb, nb = np.shape(B)
			oa = ob = 1
		elif np.ndim(A) == 3:
			[ma,na,oa] = np.shape(A)
			[mb,nb,ob] = np.shape(B)
		A = np.reshape(A, (ma, na, oa))
		B = np.reshape(B, (mb, nb, ob))
		A_nodwell = np.reshape(A_nodwell, (ma, na, oa))
		B_nodwell = np.reshape(B_nodwell, (mb, nb, ob))
		A[A_nodwell] = 0
		B[B_nodwell] = 0
		# [Step 1] Obtain FFTs of x, the sum of squares and bins visited
		Fa = np.fft.fft(np.fft.fft(A,2*mb-1,axis=0),2*nb-1,axis=1)
		FsumOfSquares_a = np.fft.fft(np.fft.fft(np.power(A,2),2*mb-1,axis=0),2*nb-1,axis=1)
		Fn_a = np.fft.fft(np.fft.fft(np.invert(A_nodwell).astype(int),2*mb-1,axis=0),2*nb-1,axis=1)

		Fb = np.fft.fft(np.fft.fft(B,2*ma-1,axis=0),2*na-1,axis=1)
		FsumOfSquares_b = np.fft.fft(np.fft.fft(np.power(B,2),2*ma-1,axis=0),2*na-1,axis=1)
		Fn_b = np.fft.fft(np.fft.fft(np.invert(B_nodwell).astype(int),2*ma-1,axis=0),2*na-1,axis=1)
		# [Step 2] Multiply the relevant transforms and invert to obtain the
		# equivalent convolutions
		rawCorr = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fa * np.conj(Fb),axis=1),axis=0)))
		sums_a = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fa * np.conj(Fn_b),axis=1),axis=0)))
		sums_b = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(Fb),axis=1),axis=0)))
		sumOfSquares_a = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(FsumOfSquares_a * np.conj(Fn_b),axis=1),axis=0)))
		sumOfSquares_b = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(FsumOfSquares_b),axis=1),axis=0)))
		N = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifft(Fn_a * np.conj(Fn_b),axis=1),axis=0)))
		# [Step 3] Account for rounding errors.
		rawCorr[np.abs(rawCorr) < tol] = 0
		sums_a[np.abs(sums_a) < tol] = 0
		sums_b[np.abs(sums_b) < tol] = 0
		sumOfSquares_a[np.abs(sumOfSquares_a) < tol] = 0
		sumOfSquares_b[np.abs(sumOfSquares_b) < tol] = 0
		N = np.round(N)
		N[N<=1] = np.nan
		# [Step 4] Compute correlation matrix
		mapStd_a = np.sqrt((sumOfSquares_a * N) - sums_a**2)
		mapStd_b = np.sqrt((sumOfSquares_b * N) - sums_b**2)
		mapCovar = (rawCorr * N) - sums_a * sums_b

		return np.squeeze(mapCovar / (mapStd_a * mapStd_b))

class MapCalcsGeneric(object):
	"""
	Produces graphical output including but not limited to spatial
	analysis of data.
	
	Parameters
	----------
	xy : array_like
		The positional data usually as a 2D numpy array
	hdir : array_like
		The head direction data usually a 1D numpy array
	pos_ts : array_like
		1D array of timestamps in seconds
	spk_ts : array_like
		1D array of timestamps in seconds
	plot_type : str or list
		Determines the plots produced. Legal values:
		['map','path','hdir','sac', 'speed']
	
	Notes
	-----
	Output possible: 
	* ratemaps (xy)
	* polar plots (heading direction)
	* grid cell spatial autocorrelograms
	* speed vs rate plots

	It is possible to iterate through instances of this class as it has a yield
	method defined
	"""
	def __init__(self, xy, hdir, speed, pos_ts, spk_ts, plot_type='map', **kwargs):
		if (np.argmin(np.shape(xy)) == 1):
			xy = xy.T
		self.xy = xy
		self.hdir = hdir
		self.speed = speed
		self.pos_ts = pos_ts
		if (spk_ts.ndim == 2):
			spk_ts = np.ravel(spk_ts)
		self.spk_ts = spk_ts
		if type(plot_type) is str:
			self.plot_type = [plot_type]
		else:
			self.plot_type = list(plot_type)
		self.spk_pos_idx = self.__interpSpkPosTimes()
		self.__good_clusters = None
		self.__spk_clusters = None
		self.save_grid_output_location = None
		if ( 'ppm' in kwargs.keys() ):
			self.__ppm = kwargs['ppm']
		else:
			self.__ppm = 400
		if 'pos_sample_rate' in kwargs.keys():
			self.pos_sample_rate = kwargs['pos_sample_rate']
		else:
			self.pos_sample_rate = 30
		if 'save_grid_summary_location' in kwargs.keys():
			self.save_grid_output_location = kwargs['save_grid_summary_location']

	@property
	def good_clusters(self):
		return self.__good_clusters

	@good_clusters.setter
	def good_clusters(self, value):
		self.__good_clusters = value

	@property
	def spk_clusters(self):
		return self.__spk_clusters

	@spk_clusters.setter
	def spk_clusters(self, value):
		self.__spk_clusters = value

	@property
	def ppm(self):
		return self.__ppm

	@ppm.setter
	def ppm(self, value):
		self.__ppm = value

	def __interpSpkPosTimes(self):
		"""
		Interpolates spike times into indices of position data
		NB Assumes pos times have been zeroed correctly - see comments in
		OEKiloPhy.OpenEphysNWB function __alignTimeStamps__()
		"""
		idx = np.searchsorted(self.pos_ts, self.spk_ts)
		idx[idx==len(self.pos_ts)] = len(self.pos_ts) - 1
		return idx

	def makeSAC(self, rmap, cluster, ax=None):
		nodwell = ~np.isfinite(rmap[0])
		R = RateMap()
		sac = R.autoCorr2D(rmap[0], nodwell)
		if ax is not None:
			pass
			# TODO : do the plotting!

	def makeRateMap(self, cluster, ax=None, **kwargs):
		pos_w = np.ones_like(self.pos_ts)
		mapMaker = RateMap(self.xy, None, None, pos_w, ppm=self.ppm)
		spk_w = np.bincount(self.spk_pos_idx, self.spk_clusters==cluster, minlength=self.pos_ts.shape[0])
		rmap = mapMaker.getMap(spk_w)
		ratemap = np.ma.MaskedArray(rmap[0], np.isnan(rmap[0]), copy=True)
		x, y = np.meshgrid(rmap[1][1][0:-1], rmap[1][0][0:-1][::-1])
		vmax = np.max(np.ravel(ratemap))
		
		ax.pcolormesh(x, y, ratemap, cmap=jet, edgecolors='face', vmax=vmax)
		ax.axis([x.min(), x.max(), y.min(), y.max()])
		ax.set_aspect('equal')
		plt.setp(ax.get_xticklabels(), visible=False)
		plt.setp(ax.get_yticklabels(), visible=False)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)

	def makeSpikePathPlot(self, cluster, ax, **kwargs):
		ax.plot(self.xy[0], self.xy[1])
		ax.set_aspect('equal')
		ax.invert_yaxis()
		idx = self.spk_pos_idx[self.spk_clusters==cluster]
		spk_colour = [0, 0, 0.7843]
		ms = 1
		if 'ms' in kwargs:
			ms = kwargs['ms']
		if 'markersize' in kwargs:
			ms = kwargs['markersize']
		ax.plot(self.xy[0,idx], self.xy[1,idx],'s',ms=ms, c=spk_colour,mec=spk_colour)
		plt.setp(ax.get_xticklabels(), visible=False)
		plt.setp(ax.get_yticklabels(), visible=False)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)

	def makeHDPlot(self, cluster, ax, **kwargs):
		pos_w = np.ones_like(self.pos_ts)
		mapMaker = RateMap(self.xy, self.hdir, None, pos_w, ppm=self.ppm)
		spk_w = np.bincount(self.spk_pos_idx, self.spk_clusters==cluster, minlength=self.pos_ts.shape[0])
		rmap = mapMaker.getMap(spk_w, 'dir', 'rate')
		if rmap[0].ndim == 1:
			# polar plot
			if ax is None:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='polar')
			theta = np.deg2rad(rmap[1][0])
			ax.clear()
			r = rmap[0]
			r = np.insert(r, -1, r[0])
			ax.plot(theta, r)
			if 'fill' in kwargs:
				ax.fill(theta, r, alpha=0.5)
			ax.set_aspect('equal')
			ax.tick_params(axis='both', which='both', bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False, labeltop=False, labelright=False)
			ax.set_rticks([])

class PosCalcsGeneric(object):
	"""
	Generic class for post-processing of position data
	Uses numpys masked arrays for dealing with bad positions, filtering etc

	Parameters
	----------
	x, y : array_like
		The x and y positions.
	ppm : int
		Pixels per metre
	cm : boolean
		Whether everything is converted into cms or not
	jumpmax : int
		Jumps in position (pixel coords) greater than this are bad

	Notes
	-----
	The positional data (x,y) is turned into a numpy masked array once this
	class is initialised - that mask is then modified through various
	functions (postprocesspos being the main one).
	"""
	def __init__(self, x, y, ppm, cm=True, jumpmax=100):
		assert np.shape(x) == np.shape(y)
		self.xy = np.ma.MaskedArray([x, y])
		self.dir = np.ma.MaskedArray(np.zeros_like(x))
		self.speed = None
		self.ppm = ppm
		self.cm = cm
		self.jumpmax = jumpmax
		self.nleds = np.ndim(x)
		self.npos = len(x)
		self.tracker_params = None
		self.sample_rate = None

	def postprocesspos(self, tracker_params, **kwargs)->tuple:
		"""
		Post-process position data

		Parameters
		----------
		tracker_params : dict
			Same dict as created in OEKiloPhy.Settings.parsePos
			(from module openephys2py)
			Contains key / values describing the spatial extent of the pos data
			keys = LeftBorder, RightBorder, TopBorder, BottomBorder and SampleRate

		Returns
		-------
		xy, hdir : np.ma.MaskedArray
			The post-processed position data

		Notes
		-----
		Several internal functions are called here: speefilter, interpnans, smoothPos
		and calcSpeed. Some internal state/ instance variables are set as well. The
		mask of the positional data (an instance of numpy masked array) is modified
		throughout this method.

		"""
		xy = self.xy
		xy = np.ma.MaskedArray(xy, dtype=np.int32)
		x_zero = xy[:, 0] < 0
		y_zero = xy[:, 1] < 0
		xy[np.logical_or(x_zero, y_zero), :] = np.ma.masked

		self.tracker_params = tracker_params
		if 'LeftBorder' in tracker_params.keys():
			min_x = tracker_params['LeftBorder']
		else:
			min_x = np.min(xy[:,0])
		xy[:, xy[0,:] <= min_x] = np.ma.masked
		if 'TopBorder' in tracker_params.keys():
			min_y = tracker_params['TopBorder'] # y origin at top
		else:
			min_y = np.min(xy[:,1])
		xy[:, xy[1,:] <= min_y] = np.ma.masked
		if 'RightBorder' in tracker_params.keys():
			max_x = tracker_params['RightBorder']
		else:
			max_x = np.max(xy[:,0])
		xy[:, xy[0,:] >= max_x] = np.ma.masked
		if 'BottomBorder' in tracker_params.keys():
			max_y = tracker_params['BottomBorder']
		else:
			max_y = np.max(xy[:,1])
		xy[:, xy[1,:] >= max_y] = np.ma.masked
		if 'SampleRate' in tracker_params.keys():
			self.sample_rate = int(tracker_params['SampleRate'])
		else:
			self.sample_rate = 30

		xy = xy.T
		xy = self.speedfilter(xy)
		xy = self.interpnans(xy) # ADJUST THIS SO NP.MASKED ARE INTERPOLATED OVER
		xy = self.smoothPos(xy)
		self.calcSpeed(xy)

		import math
		pos2 = np.arange(0, self.npos-1)
		xy_f = xy.astype(np.float)
		self.dir[pos2] = np.mod(((180/math.pi) * (np.arctan2(-xy_f[1, pos2+1] + xy_f[1,pos2],+xy_f[0,pos2+1]-xy_f[0,pos2]))), 360)
		self.dir[-1] = self.dir[-2]

		hdir = self.dir

		return xy, hdir

	def speedfilter(self, xy):
		"""
		Filters speed

		Parameters
		----------
		xy : np.ma.MaskedArray
			The xy data

		Returns
		-------
		xy : np.ma.MaskedArray
			The xy data with speeds > self.jumpmax masked
		"""
		
		disp = np.hypot(xy[:,0], xy[:,1])
		disp = np.diff(disp, axis=0)
		disp = np.insert(disp, -1, 0)
		xy[np.abs(disp) > self.jumpmax, :] = np.ma.masked
		return xy

	def interpnans(self, xy):
		for i in range(0,np.shape(xy)[-1],2):
			missing = xy.mask.any(axis=-1)
			ok = np.logical_not(missing)
			ok_idx = np.ravel(np.nonzero(np.ravel(ok))[0])#gets the indices of ok poses
			missing_idx = np.ravel(np.nonzero(np.ravel(missing))[0])#get the indices of missing poses
			if len(missing_idx) > 0:
				try:
					good_data = np.ravel(xy.data[ok_idx,i])
					good_data1 = np.ravel(xy.data[ok_idx,i+1])
					xy.data[missing_idx,i] = np.interp(missing_idx,ok_idx,good_data)#,left=np.min(good_data),right=np.max(good_data)
					xy.data[missing_idx,i+1] = np.interp(missing_idx,ok_idx,good_data1)
				except ValueError:
					pass
		xy.mask = 0
		print("{} bad/ jumpy positions were interpolated over".format(len(missing_idx)))#this is wrong i think
		return xy

	def __smooth(self, x, window_len=9, window='hanning'):
		"""
		Smooth the data using a window with requested size.
		
		This method is based on the convolution of a scaled window with the signal.
		The signal is prepared by introducing reflected copies of the signal 
		(with the window size) in both ends so that transient parts are minimized
		in the begining and end part of the output signal.
		
		Parameters
		----------
		x : array_like
			the input signal 
		window_len : int
			The length of the smoothing window
		window : str
			The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			'flat' window will produce a moving average smoothing.

		Returns
		-------
		out : The smoothed signal
			
		Example
		-------
		>>> t=linspace(-2,2,0.1)
		>>> x=sin(t)+randn(len(t))*0.1
		>>> y=smooth(x)
		
		See Also
		--------
		numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
		scipy.signal.lfilter
	
		TODO: the window parameter could be the window itself if an array instead of a string   
		"""

		if type(x) == type([]):
			x = np.array(x)

		if x.ndim != 1:
			raise ValueError("smooth only accepts 1 dimension arrays.")

		if x.size < window_len:
			raise ValueError("Input vector needs to be bigger than window size.")
		if window_len < 3:
			return x

		if (window_len % 2) == 0:
			window_len = window_len + 1

		if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
			raise ValueError(
				"Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

		if window == 'flat':  # moving average
			w = np.ones(window_len, 'd')
		else:
			w = eval('np.'+window+'(window_len)')
		from astropy.convolution import convolve
		y = convolve(x, w/w.sum(), normalize_kernel=False, boundary='extend')
		# return the smoothed signal
		return y

	def smoothPos(self, xy):
		"""
		Smooths position data

		Parameters
		----------
		xy : np.ma.MaskedArray
			The xy data

		Returns
		-------
		xy : array_like
			The smoothed positional data
		"""
		# Extract boundaries of window used in recording

		x = xy[:,0].astype(np.float64)
		y = xy[:,1].astype(np.float64)

		# TODO: calculate window_len from pos sampling rate
		# 11 is roughly equal to 400ms at 30Hz (window_len needs to be odd)
		sm_x = self.__smooth(x, window_len=11, window='flat')
		sm_y = self.__smooth(y, window_len=11, window='flat')
		return np.array([sm_x, sm_y])

	def calcSpeed(self, xy):
		"""
		Calculates speed

		Parameters
		---------
		xy : np.ma.MaskedArray
			The xy positional data

		Returns
		-------
		Nothing. Sets self.speed
		"""
		speed = np.sqrt(np.sum(np.power(np.diff(xy),2),0))
		speed = np.append(speed, speed[-1])
		if self.cm:
			self.speed = speed * (100 * self.sample_rate / self.ppm) # in cm/s now
		else:
			self.speed = speed

	def upsamplePos(self, xy, upsample_rate=50):
		"""
		Upsamples position data from 30 to upsample_rate

		Parameters
		---------
		
		xy : np.ma.MaskedArray
			The xy positional data

		upsample_rate : int
			The rate to upsample to

		Returns
		-------
		new_xy : np.ma.MaskedArray
			The upsampled xy positional data

		Notes
		-----
		This is mostly to get pos data recorded using PosTracker at 30Hz
		into Axona format 50Hz data
		"""
		from scipy import signal
		denom = np.gcd(upsample_rate, 30)
		new_xy = signal.resample_poly(xy, upsample_rate/denom, 30/denom)
		return new_xy