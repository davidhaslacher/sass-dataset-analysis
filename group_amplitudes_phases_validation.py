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
from scipy.signal import hilbert
from scipy.stats import ttest_ind
from pycircstat.tests import watson_williams
from scipy.stats import wilcoxon

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_context('poster')
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def plv(phases):
    return np.abs(np.exp(1j*phases).mean())

def circ_mean(phases):
    return np.angle(np.exp(1j*phases).mean())

def robust_z_score(ys):
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.array(modified_z_scores)

def circ_detrend(phases):
    return wrap(phases-np.angle(np.exp(1j*phases).mean()))

def find_n_nulls(A,B,D,M):
    mses = []
    for n_nulls in range(A.shape[0]):
        DI = np.ones(M.shape[0])
        DI[:n_nulls] = 0
        DI = np.diag(DI)
        P = M.dot(DI).dot(D)
        mses.append(np.mean((np.diag(B)-np.diag(P.dot(A).dot(P.T)))*np.conj((np.diag(B)-np.diag(P.dot(A).dot(P.T))))))
    return np.argmin(mses)

participants = ['p1','p2','p3','p4','p5','p6']
base_path = '../../SASS_data'
bads = np.load('bads.npy',allow_pickle=True).item()
binmethod = 10
n_folds = 5
n_trials_cov = 50
n_trials_total = 100

all_amplitudes_no_stim = []
all_amplitudes_tacs_without_sass = []
all_amplitudes_tacs_with_sass = []

all_plvs_no_stim = []
all_plvs_tacs_without_sass = []
all_plvs_tacs_with_sass = []

for participant in participants:

    path = base_path+'/'+participant+'/no_stim.vhdr'

    raw_no_stim = mne.io.read_raw_brainvision(path,preload=True)
    if participant == 'p5':
        raw_no_stim.crop(0,420)

    threshold = 3
    audio_events = np.where(np.diff((stats.zscore(raw_no_stim.copy().
        pick_channels(['audio'])._data.flatten()) > threshold).astype('int'))>0)[0]

    miniti = raw_no_stim.time_as_index(0.4)[0]
    audio_events_no_stim = [audio_events[ix] for ix in range(1,audio_events.size) 
                    if audio_events[ix]-audio_events[ix-1] > miniti]

    # # Check if audio events are correct
    # plt.plot(raw_no_stim.times,raw_no_stim.copy().pick_channels(['audio'])._data.flatten())
    # for ev in audio_events:
    #     plt.axvline(raw_no_stim.times[ev],c='r')
    # # plt.xlim([0,1])
    # plt.show()

    raw_no_stim.pick_channels([ch for ch in raw_no_stim.ch_names if ch not in bads[participant]])
    montage = make_standard_montage('easycap-M1')
    raw_no_stim.set_montage(montage,match_case=False)
    raw_no_stim.filter(9,11)
    # chidxs = [raw_no_stim.ch_names.index(chname) for chname in raw_no_stim.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
    # hil = hilbert(raw_no_stim._data[chidxs].mean(0))
    hil = raw_no_stim.copy().pick_channels([ch for ch in raw_no_stim.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
    phases_brain = np.angle(hil)
    amplitudes_brain = np.abs(hil)

    t = np.arange(0,2,1/500)
    phases_flicker_trial = wrap(2*np.pi*10*t)
    nsamp_trial = phases_flicker_trial.size
    nsamp_start = 0
    phasediffs = []
    amplitudes = []
    for ev in audio_events_no_stim:
        if ev < len(raw_no_stim.times)-nsamp_trial:
            phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
            amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))

    phasediffs_no_tacs = np.array(phasediffs)[n_trials_cov:n_trials_total]
    amplitudes_no_tacs = np.array(amplitudes)[n_trials_cov:n_trials_total]

    all_amplitudes_no_stim.append(np.mean(amplitudes_no_tacs))
    all_plvs_no_stim.append(plv(phasediffs_no_tacs))


    path = base_path+'/'+participant+'/open.vhdr'
    raw_open = mne.io.read_raw_brainvision(path,preload=True)
    threshold = 3
    audio_events = np.where(np.diff((stats.zscore(raw_open.copy().
        pick_channels(['audio'])._data.flatten()) > threshold).astype('int'))>0)[0]
    miniti = raw_open.time_as_index(0.4)[0]
    audio_events_open = [audio_events[ix] for ix in range(1,audio_events.size) 
                    if audio_events[ix]-audio_events[ix-1] > miniti]

    # # Check if audio events are correct
    # plt.plot(raw_open.times,raw_open.copy().pick_channels(['audio'])._data.flatten())
    # for ev in audio_events:
    #     plt.axvline(raw_open.times[ev],c='r')
    # # plt.xlim([0,1])
    # plt.show()

    raw_open.pick_channels([ch for ch in raw_open.ch_names if ch not in bads[participant]])
    montage = make_standard_montage('easycap-M1')
    raw_open.set_montage(montage,match_case=False)
    raw_open.filter(9,11)
    # chidxs = [raw_open.ch_names.index(chname) for chname in raw_open.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
    # hil = hilbert(raw_open._data[chidxs].mean(0))
    hil = raw_open.copy().pick_channels([ch for ch in raw_open.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
    phases_brain = np.angle(hil)
    amplitudes_brain = np.abs(hil)

    t = np.arange(0,2,1/500)
    phases_flicker_trial = wrap(2*np.pi*10*t)
    nsamp_trial = phases_flicker_trial.size
    nsamp_start = 0
    phasediffs = []
    amplitudes = []
    for ev in audio_events_open:
        if ev < len(raw_open.times)-nsamp_trial:
            phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
            amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))

    phasediffs_tacs_without_sass = np.array(phasediffs)[n_trials_cov:n_trials_total]
    amplitudes_tacs_without_sass = np.array(amplitudes)[n_trials_cov:n_trials_total]

    # A = np.cov(raw_open._data[:,:audio_events_open[n_trials_cov]])
    A = np.cov(raw_open._data)
    B = np.cov(raw_no_stim._data)
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
    raw_no_stim._data = P.dot(raw_no_stim._data)

    hil = raw_no_stim.copy().pick_channels([ch for ch in raw_no_stim.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
    phases_brain = np.angle(hil)
    amplitudes_brain = np.abs(hil)

    t = np.arange(0,2,1/500)
    phases_flicker_trial = wrap(2*np.pi*10*t)
    nsamp_trial = phases_flicker_trial.size
    nsamp_start = 0
    phasediffs = []
    amplitudes = []
    for ev in audio_events_no_stim:
        if ev < len(raw_no_stim.times)-nsamp_trial:
            phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
            amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))

    phasediffs_no_tacs_with_sass = np.array(phasediffs)[n_trials_cov:n_trials_total]
    amplitudes_no_tacs_with_sass = np.array(amplitudes)[n_trials_cov:n_trials_total]

    hil = raw_open.copy().pick_channels([ch for ch in raw_open.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
    phases_brain = np.angle(hil)
    amplitudes_brain = np.abs(hil)

    t = np.arange(0,2,1/500)
    phases_flicker_trial = wrap(2*np.pi*10*t)
    nsamp_trial = phases_flicker_trial.size
    nsamp_start = 0
    phasediffs = []
    amplitudes = []
    for ev in audio_events_open:
        if ev < len(raw_open.times)-nsamp_trial:
            phasediffs.append(circ_mean(wrap(phases_brain[ev+nsamp_start:ev+nsamp_trial]-phases_flicker_trial[nsamp_start:])))
            amplitudes.append(np.mean(amplitudes_brain[ev+nsamp_start:ev+nsamp_trial]))

    phasediffs_tacs_with_sass = np.array(phasediffs)[n_trials_cov:n_trials_total]
    amplitudes_tacs_with_sass = np.array(amplitudes)[n_trials_cov:n_trials_total]

    all_amplitudes_tacs_without_sass.append(np.mean(amplitudes_tacs_without_sass))
    all_amplitudes_tacs_with_sass.append(np.mean(amplitudes_tacs_with_sass))
    all_plvs_tacs_without_sass.append(plv(phasediffs_tacs_without_sass))
    all_plvs_tacs_with_sass.append(plv(phasediffs_tacs_with_sass))

plt.figure(figsize=(12,8))
n_subjects = len(participants)
x = ['No tACS']*n_subjects+['tACS without SASS']*n_subjects+['tACS with SASS']*n_subjects
y = all_amplitudes_no_stim+all_amplitudes_tacs_without_sass+all_amplitudes_tacs_with_sass
x = np.array(x)
y = np.array(y)
sns.boxplot(x=x, y=y, boxprops=dict(alpha=.3), color='k')
sns.stripplot(x=x, y=y, jitter=False, color='k')
locs, labels = plt.xticks()
x = np.tile(locs[:,np.newaxis],(1,len(participants)))
y = np.array([all_amplitudes_no_stim,all_amplitudes_tacs_without_sass,all_amplitudes_tacs_with_sass])
plt.plot(x,y)
plt.ylabel('Amplitude (uv)')
plt.yscale('log')
plt.minorticks_off()
plt.tight_layout()
sns.despine()
plt.savefig('figs/open/group_amplitudes_validation.pdf')
p1 = wilcoxon(all_amplitudes_no_stim,all_amplitudes_tacs_without_sass,alternative='less')[1]
p2 = wilcoxon(all_amplitudes_tacs_without_sass,all_amplitudes_tacs_with_sass,alternative='greater')[1]
p3 = wilcoxon(all_amplitudes_no_stim,all_amplitudes_tacs_with_sass,alternative='two-sided')[1]
pvals = np.array([p1,p2,p3])
np.save('figs/open/pvals_group_amplitudes_validation.npy',pvals)

plt.figure(figsize=(12,8))
n_subjects = len(participants)
x = ['No tACS']*n_subjects+['tACS without SASS']*n_subjects+['tACS with SASS']*n_subjects
y = all_plvs_no_stim+all_plvs_tacs_without_sass+all_plvs_tacs_with_sass
x = np.array(x)
y = np.array(y)
sns.boxplot(x=x, y=y, boxprops=dict(alpha=.3), color='k')
sns.stripplot(x=x, y=y, jitter=False, color='k')
locs, labels = plt.xticks()
x = np.tile(locs[:,np.newaxis],(1,len(participants)))
y = np.array([all_plvs_no_stim,all_plvs_tacs_without_sass,all_plvs_tacs_with_sass])
plt.plot(x,y)
plt.ylabel('PLV')
plt.tight_layout()
sns.despine()
plt.savefig('figs/open/group_plvs_validation.pdf')
p1 = wilcoxon(all_plvs_no_stim,all_plvs_tacs_without_sass,alternative='greater')[1]
p2 = wilcoxon(all_plvs_tacs_without_sass,all_plvs_tacs_with_sass,alternative='less')[1]
p3 = wilcoxon(all_plvs_no_stim,all_plvs_tacs_with_sass,alternative='two-sided')[1]
pvals = np.array([p1,p2,p3])
np.save('figs/open/pvals_group_plvs_validation.npy',pvals)