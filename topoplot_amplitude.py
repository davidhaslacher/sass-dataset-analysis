import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg
from scipy.io import savemat, loadmat
from mne.preprocessing import compute_current_source_density
from mne.channels import make_standard_montage
import seaborn as sns
import matplotlib
from mne.viz import plot_topomap
from mne.stats import spatio_temporal_cluster_test, permutation_cluster_test
from scipy.stats import zscore
from mne.channels import find_ch_connectivity
import scipy
from functools import reduce
from pycircstat.tests import watson_williams

def kruskal(*input):
    n_tests = input[0].shape[1]
    stats = []
    for ix in range(n_tests):
        stats.append(scipy.stats.kruskal(*[inp[:,ix] for inp in input])[0])
    return np.array(stats)

def mannwhitneyu(A,B):
    n_tests = A.shape[1]
    stats = []
    for ix in range(n_tests):
        stats.append(scipy.stats.mannwhitneyu(A[:,ix],B[:,ix])[0])
    return np.array(stats)

def f_stat(A,B):
    n_tests = A.shape[1]
    stats = []
    for ix in range(n_tests):
        stats.append(scipy.stats.f_oneway(A[:,ix],B[:,ix])[0])
    return np.array(stats)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_context('poster')
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def plv(phases):
    return np.abs(np.exp(1j*phases).mean())

def circ_detrend(phases):
    return wrap(phases-np.angle(np.exp(1j*phases).mean()))

def find_n_nulls(A,B,D,M):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        mses.append(np.mean((np.diag(B)-np.diag(P.dot(A).dot(P.T)))**2))
    return np.argmin(mses)

participants = ['p1','p2','p3','p4','p5','p6']
base_path = '../../SASS_data'
bads = np.load('bads.npy',allow_pickle=True).item()
# bads = dict(p1=['audio','stim','ECG','CPz','Pz','P2','CP2','C2'])
binmethod = 10
figsize = (18,6)
n_trials = 200

for participant in participants:

    path = base_path+'/'+participant+'/no_stim.vhdr'

    raw_no_stim = mne.io.read_raw_brainvision(path,preload=True)

    threshold = 3
    audio_events = np.where(np.diff((stats.zscore(raw_no_stim.copy().
        pick_channels(['audio'])._data.flatten()) > threshold).astype('int'))>0)[0]

    miniti = raw_no_stim.time_as_index(0.4)[0]
    audio_events_no_stim = [audio_events[ix] for ix in range(1,audio_events.size) 
                    if audio_events[ix]-audio_events[ix-1] > miniti]

    raw_no_stim.pick_channels([ch for ch in raw_no_stim.ch_names if ch not in bads[participant]])
    montage = make_standard_montage('easycap-M1')
    raw_no_stim.set_montage(montage,match_case=False)
    raw_no_stim.filter(9,11)
    hil = raw_no_stim.copy().apply_hilbert(envelope=False)._data.squeeze()
    amplitudes_brain = np.abs(hil)

    nsamp_trial = raw_no_stim.time_as_index(2)[0]
    amplitudes = []
    for ev in audio_events_no_stim:
        if ev < len(raw_no_stim.times)-nsamp_trial:
            amplitudes.append(np.mean(amplitudes_brain[:,ev:ev+nsamp_trial],axis=1))

    amplitudes_no_tacs = np.array(amplitudes)[:n_trials].T


    path = base_path+'/'+participant+'/open.vhdr'
    raw_open = mne.io.read_raw_brainvision(path,preload=True)
    threshold = 3
    audio_events = np.where(np.diff((stats.zscore(raw_open.copy().
        pick_channels(['audio'])._data.flatten()) > threshold).astype('int'))>0)[0]
    miniti = raw_open.time_as_index(0.4)[0]
    audio_events_open = [audio_events[ix] for ix in range(1,audio_events.size) 
                    if audio_events[ix]-audio_events[ix-1] > miniti]

    raw_open.pick_channels([ch for ch in raw_open.ch_names if ch not in bads[participant]])
    montage = make_standard_montage('easycap-M1')
    raw_open.set_montage(montage,match_case=False)
    raw_open.filter(9,11)
    hil = raw_open.copy().apply_hilbert(envelope=False)._data.squeeze()
    amplitudes_brain = np.abs(hil)

    nsamp_trial = raw_no_stim.time_as_index(2)[0]
    amplitudes = []
    for ev in audio_events_open:
        if ev < len(raw_open.times)-nsamp_trial:
            amplitudes.append(np.mean(amplitudes_brain[:,ev:ev+nsamp_trial],axis=1))

    amplitudes_tacs_without_sass = np.array(amplitudes)[:n_trials].T

    A = np.cov(raw_open._data)
    B = np.cov(raw_no_stim._data)
    eigen_values, eigen_vectors = linalg.eig(A,B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:,ix].T
    M = linalg.pinv2(D)
    DI = np.ones(M.shape[0])
    n_nulls = find_n_nulls(A,B,D,M)
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    raw_open._data = P.dot(raw_open._data)
    raw_no_stim._data = P.dot(raw_no_stim._data)


    hil = raw_no_stim.copy().apply_hilbert(envelope=False)._data.squeeze()
    amplitudes_brain = np.abs(hil)

    nsamp_trial = raw_no_stim.time_as_index(2)[0]
    amplitudes = []
    for ev in audio_events_no_stim:
        if ev < len(raw_no_stim.times)-nsamp_trial:
            amplitudes.append(np.mean(amplitudes_brain[:,ev:ev+nsamp_trial],axis=1))

    amplitudes_no_tacs_with_sass = np.array(amplitudes)[:n_trials].T


    hil = raw_open.copy().apply_hilbert(envelope=False)._data.squeeze()
    amplitudes_brain = np.abs(hil)

    nsamp_trial = raw_no_stim.time_as_index(2)[0]
    amplitudes = []
    amplitudes_trials = []
    for ev in audio_events_open:
        if ev < len(raw_open.times)-nsamp_trial:
            amplitudes.append(np.mean(amplitudes_brain[:,ev:ev+nsamp_trial],axis=1))

    amplitudes_tacs_with_sass = np.array(amplitudes)[:n_trials].T

    fig, axs = plt.subplots(1, 3,figsize=figsize)
    ax = axs[1]
    data = np.log10(np.median(amplitudes_tacs_without_sass,axis=1))
    vmin = data.min()
    vmax = data.max()
    im2,_ = plot_topomap(data, vmin=vmin,vmax=vmax, sensors=False, pos=raw_no_stim.info, 
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet', axes=ax)
    cb = fig.colorbar(im2, ax=ax)
    cb.set_label('Amplitude (log V)')
    # cb.formatter.set_powerlimits((-6,-6))
    cb.update_ticks()
    ax = axs[0]
    data = np.log10(np.median(amplitudes_no_tacs,axis=1))
    vmin_b = data.min()
    vmax_b = data.max()
    im1,_ = plot_topomap(data, vmin=vmin_b,vmax=vmax_b, sensors=False, pos=raw_no_stim.info, 
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet', axes=ax)
    vmin, vmax = im1.get_clim()
    cb = fig.colorbar(im1, ax=ax)
    cb.set_label('Amplitude (log V)')
    # cb.formatter.set_powerlimits((-6,-6))
    cb.update_ticks()
    ax = axs[2]
    data = np.log10(np.median(amplitudes_tacs_with_sass,axis=1))
    im3,_ = plot_topomap(data, vmin=vmin_b,vmax=vmax_b, sensors=False, pos=raw_no_stim.info,
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet', axes=ax)
    cb = fig.colorbar(im3, ax=ax)
    cb.set_label('Amplitude (log V)')
    # cb.formatter.set_powerlimits((-6,-6))
    cb.update_ticks()
    plt.savefig('figs/open/'+participant+'_amplitudes_topo.pdf')

    fig, axs = plt.subplots(1, 2,figsize=figsize)
    ax = axs[0]
    im1,_ = plot_topomap(np.median(amplitudes_no_tacs,axis=1), sensors=False, pos=raw_no_stim.info, 
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet', axes=ax)
    vmin, vmax = im1.get_clim()
    cb = fig.colorbar(im1, ax=ax)
    cb.set_label('Amplitude (uV)')
    cb.formatter.set_powerlimits((-6,-6))
    cb.update_ticks()
    ax = axs[1]
    im2,_ = plot_topomap(np.median(amplitudes_no_tacs_with_sass,axis=1), sensors=False, pos=raw_no_stim.info, 
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet', vmin=vmin,vmax=vmax,axes=ax)
    cb = fig.colorbar(im2, ax=ax)
    cb.set_label('Amplitude (uV)')
    cb.formatter.set_powerlimits((-6,-6))
    cb.update_ticks()
    plt.savefig('figs/open/'+participant+'_amplitudes_attenuation_topo.pdf')


        
    threshold = dict(start=0, step=-0.2)


    connectivity, ch_names = find_ch_connectivity(raw_no_stim.info, ch_type='eeg')
    _,clusters,pvals,_ = spatio_temporal_cluster_test([amplitudes_tacs_with_sass.T[:,np.newaxis,:],
                        amplitudes_no_tacs.T[:,np.newaxis,:]],connectivity=connectivity,threshold=threshold,tail=1)

    plt.figure()
    significant_clusters = [clusters[ix] for ix in np.where(pvals < 0.05)[0]]
    data = np.zeros(len(raw_no_stim.ch_names))
    for cluster in significant_clusters:
        data[cluster[1]] = 1
    im,_ = plot_topomap(data, sensors=False, pos=raw_no_stim.info, 
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet',vmin=0,vmax=1)
    plt.savefig('figs/open/'+participant+'_cluster_amplitudes.pdf')


    connectivity, ch_names = find_ch_connectivity(raw_no_stim.info, ch_type='eeg')
    _,clusters,pvals,_ = spatio_temporal_cluster_test([amplitudes_no_tacs_with_sass.T[:,np.newaxis,:],
                        amplitudes_no_tacs.T[:,np.newaxis,:]],connectivity=connectivity,threshold=threshold,tail=1)
    
    plt.figure()
    significant_clusters = [clusters[ix] for ix in np.where(pvals < 0.05)[0]]
    data = np.zeros(len(raw_no_stim.ch_names))
    for cluster in significant_clusters:
        data[cluster[1]] = 1
    im,_ = plot_topomap(data, sensors=False, pos=raw_no_stim.info, 
    contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet',vmin=0,vmax=1)
    plt.savefig('figs/open/'+participant+'_cluster_amplitudes_attenuation.pdf')

    # # plt.show()