import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg
from scipy.io import savemat, loadmat
from mne.preprocessing import compute_current_source_density
from mne.channels import make_standard_montage
from mne.time_frequency import psd_welch
import seaborn as sns
import matplotlib
from mne.viz import plot_topomap
from itertools import product
from matplotlib import rcParams

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_context('poster')
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def plv(phases):
    return np.abs(np.exp(1j*phases).mean())

def robust_z_score(ys):
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.array(modified_z_scores)

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

for participant in participants:

    plt.figure(figsize=(18,12))
    path = base_path+'/'+participant+'/no_stim.vhdr'
    raw_no_stim = mne.io.read_raw_brainvision(path,preload=True)
    raw_no_stim.pick_channels([ch for ch in raw_no_stim.ch_names if ch not in bads[participant]])
    montage = make_standard_montage('easycap-M1')
    raw_no_stim.set_montage(montage,match_case=False)


    path = base_path+'/'+participant+'/open.vhdr'
    raw_open = mne.io.read_raw_brainvision(path,preload=True)
    raw_open.pick_channels([ch for ch in raw_open.ch_names if ch not in bads[participant]])
    montage = make_standard_montage('easycap-M1')
    raw_open.set_montage(montage,match_case=False)
    

    A = np.cov(raw_open.copy().filter(9,11)._data)
    B = np.cov(raw_no_stim.copy().filter(9,11)._data)
    eigen_values, eigen_vectors = linalg.eig(A,B)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    ix = np.argsort(eigen_values)[::-1]
    D = eigen_vectors[:,ix].T
    M = linalg.pinv2(D)
    n_nulls = find_n_nulls(A,B,D,M)
    DI = np.ones(M.shape[0])
    DI[:n_nulls] = 0
    DI = np.diag(DI)
    P = M.dot(DI).dot(D)
    raw_open._data = P.dot(raw_open._data)
    sorted_eigen_values = eigen_values[ix]

    plt.figure(figsize=(12,8))
    plt.semilogy(eigen_values[ix],linewidth=5)
    plt.xlabel('Component #')
    plt.ylabel('Eigenvalue')
    plt.tight_layout()
    plt.axvline(n_nulls,c='royalblue',label='Threshold')
    plt.legend()
    sns.despine()
    plt.minorticks_off()
    plt.savefig('figs/open/'+participant+'_eigenspectrum.pdf')


    fig, axs = plt.subplots(2,3,figsize=(6.4*2,4.8*2))
    for ix1,ix2 in product(range(2),range(3)):
        ax = axs[ix1,ix2]
        flat_ix = ix1*3+ix2
        im1,_ = plot_topomap(M.T[flat_ix], sensors=False, pos=raw_no_stim.info, 
            contours=0, show=False, show_names=False, names=raw_no_stim.info['ch_names'], cmap='jet', axes=ax)
        ax.set_title("{:.0f}".format(sorted_eigen_values[flat_ix]))
    plt.tight_layout()
    plt.savefig('figs/open/'+participant+'_spatial_patterns.pdf')