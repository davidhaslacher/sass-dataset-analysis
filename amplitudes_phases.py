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
from scipy.stats import ttest_ind, ttest_rel, wilcoxon, mannwhitneyu
from pycircstat.tests import watson_williams
from scipy.stats import circstd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set_context('poster')
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

def wrap(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

def plv(phases):
    return np.abs(np.exp(1j*phases).mean())

def plv_unbiasedz(phases):
    N = phases.size
    return (1/(N-1))*((plv(phases)**2)*N-1)

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

def wallraff_dep(A,B,alternative):
    return wilcoxon(np.abs(circ_detrend(A)),np.abs(circ_detrend(B)),alternative=alternative)[1]

def wallraff_ind(A,B,alternative):
    return mannwhitneyu(np.abs(circ_detrend(A)),np.abs(circ_detrend(B)),alternative=alternative)[1]

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
binmethod = 10
n_trials = 200

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

    phasediffs_no_tacs = circ_detrend(np.array(phasediffs)[:n_trials])
    amplitudes_no_tacs = np.array(amplitudes)[:n_trials]


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
    # chidxs = [raw_open.ch_names.index(chname) for chname in raw_open.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
    # hil = hilbert(raw_open._data[chidxs].mean(0))
    hil = raw_open.copy().pick_channels([ch for ch in raw_no_stim.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
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

    phasediffs_tacs_without_sass = circ_detrend(np.array(phasediffs)[:n_trials])
    amplitudes_tacs_without_sass = np.array(amplitudes)[:n_trials]

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

    phasediffs_no_tacs_with_sass = circ_detrend(np.array(phasediffs)[:n_trials])
    amplitudes_no_tacs_with_sass = np.array(amplitudes)[:n_trials]

    # chidxs = [raw_open.ch_names.index(chname) for chname in raw_open.ch_names if chname[0] == 'O' or chname[:2] == 'PO']
    # hil = hilbert(raw_open._data[chidxs].mean(0))
    hil = raw_open.copy().pick_channels([ch for ch in raw_no_stim.ch_names if ch[:1]=='O' or ch[:2]=='PO']).apply_hilbert(envelope=False)._data.mean(0)
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

    phasediffs_tacs_with_sass = circ_detrend(np.array(phasediffs)[:n_trials])
    amplitudes_tacs_with_sass = np.array(amplitudes)[:n_trials]

    # fig = plt.figure(figsize=(12,12))
    # ax = plt.subplot2grid( (1,2), [0,0], 1, 1,projection='polar')
    # ax.tick_params(pad=10)
    # ax.yaxis.grid(False)
    # ax.xaxis.grid(False)
    # ax.get_yaxis().set_visible(False)
    # plt.hist(circ_detrend(phasediffs_no_tacs),edgecolor='black',bins=binmethod)
    # ax.tick_params(axis='x', labelsize=15)
    # plt.title('No tACS',y=1.2)
    # plv_open_no_sass = plv(phasediffs_no_tacs)
    # ax = plt.subplot2grid( (1,2), [0,1], 1, 1,projection='polar')
    # ax.tick_params(pad=10)
    # ax.yaxis.grid(False)
    # ax.xaxis.grid(False)
    # ax.get_yaxis().set_visible(False)
    # plt.hist(circ_detrend(phasediffs_no_tacs_with_sass),edgecolor='black',bins=binmethod)
    # ax.tick_params(axis='x', labelsize=15)
    # plt.title('No tACS with SASS',y=1.2)
    # plv_open_no_sass = plv(phasediffs_no_tacs_with_sass)
    # fig.tight_layout(pad=1)
    # plt.savefig('figs/open/'+participant+'_phases_attenuation.pdf')

    fig,axs = plt.subplots(1,3,subplot_kw=dict(polar=True),figsize=(18,6))
    ax = axs[0]
    ax.tick_params(pad=10)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.hist(phasediffs_no_tacs,edgecolor='black',bins=binmethod)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_title('No tACS',y=1.2)
    # plt.title(str(plv(phasediffs_no_tacs)))
    ax = axs[1]
    ax.tick_params(pad=10)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.hist(phasediffs_tacs_without_sass,edgecolor='black',bins=binmethod)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_title('tACS without SASS',y=1.2)
    # plt.title(str(plv(phasediffs_tacs_without_sass)))
    ax = axs[2]
    ax.tick_params(pad=10)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.hist(phasediffs_tacs_with_sass,edgecolor='black',bins=binmethod)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_title('tACS with SASS',y=1.2)
    # plt.title(str(plv(phasediffs_tacs_with_sass)))
    fig.tight_layout()
    plt.savefig('figs/open/'+participant+'_phases.pdf')
    pvals_phases = np.zeros(3)
    pvals_phases[0] = wallraff_ind(phasediffs_no_tacs,phasediffs_tacs_without_sass,alternative='less')
    pvals_phases[1] = wallraff_dep(phasediffs_tacs_without_sass,phasediffs_tacs_with_sass,alternative='greater')
    pvals_phases[2] = wallraff_ind(phasediffs_no_tacs,phasediffs_tacs_with_sass,alternative='two-sided')
    np.save('figs/open/'+participant+'_pvals_phases.npy',pvals_phases)

    phase_means = np.empty(3)
    phase_means[0] = circ_mean(phasediffs_no_tacs)
    phase_means[1] = circ_mean(phasediffs_tacs_without_sass)
    phase_means[2] = circ_mean(phasediffs_tacs_with_sass)
    np.save('figs/open/'+participant+'_phase_means.npy',phase_means)

    phase_stds = np.empty(3)
    phase_stds[0] = plv(phasediffs_no_tacs)
    phase_stds[1] = plv(phasediffs_tacs_without_sass)
    phase_stds[2] = plv(phasediffs_tacs_with_sass)
    np.save('figs/open/'+participant+'_phase_stds.npy',phase_stds)

    amplitude_means = np.empty(3)
    amplitude_means[0] = np.mean(amplitudes_no_tacs)
    amplitude_means[1] = np.mean(amplitudes_tacs_without_sass)
    amplitude_means[2] = np.mean(amplitudes_tacs_with_sass)
    np.save('figs/open/'+participant+'_amplitude_means.npy',amplitude_means)

    amplitude_stds = np.empty(3)
    amplitude_stds[0] = np.std(amplitudes_no_tacs)
    amplitude_stds[1] = np.std(amplitudes_tacs_without_sass)
    amplitude_stds[2] = np.std(amplitudes_tacs_with_sass)
    np.save('figs/open/'+participant+'_amplitude_stds.npy',amplitude_stds)


    # plt.figure(figsize=(18,12))
    # X = ['No tACS']*amplitudes_no_tacs.size+['No tACS with SASS']*amplitudes_no_tacs_with_sass.size
    # Y = np.concatenate([amplitudes_no_tacs,amplitudes_no_tacs_with_sass])
    # ax = sns.violinplot(X,Y)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(-6,-6))
    # plt.setp(ax.yaxis.get_offset_text(), visible=False)
    # plt.ylabel('Amplitude (uV)')
    # sns.despine()
    # plt.savefig('figs/open/'+participant+'_amplitudes_attenuation.pdf')


    plt.figure(figsize=(12,8))
    X = ['No tACS']*amplitudes_no_tacs.size+['tACS without SASS']*amplitudes_tacs_without_sass.size+['tACS with SASS']*amplitudes_tacs_with_sass.size
    Y = np.concatenate([amplitudes_no_tacs,amplitudes_tacs_without_sass,amplitudes_tacs_with_sass])
    plt.yscale('log')
    ax = sns.stripplot(X,Y)
    # sns.boxplot(X,Y, boxprops=dict(alpha=.3), color='k',ax=ax)
    plt.minorticks_off()
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(-6,-6))
    plt.setp(ax.yaxis.get_offset_text(), visible=False)
    plt.ylabel('Amplitude (V)')
    sns.despine()
    fig.tight_layout()
    plt.savefig('figs/open/'+participant+'_amplitudes.pdf')
    pvals_amplitudes = np.zeros(3)
    pvals_amplitudes[0] = ttest_ind(amplitudes_no_tacs,amplitudes_tacs_without_sass)[1]/2
    pvals_amplitudes[1] = ttest_rel(amplitudes_tacs_without_sass,amplitudes_tacs_with_sass)[1]/2
    pvals_amplitudes[2] = ttest_ind(amplitudes_no_tacs,amplitudes_tacs_with_sass)[1]
    np.save('figs/open/'+participant+'_pvals_amplitudes.npy',pvals_amplitudes)
    # plt.show()