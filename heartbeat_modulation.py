import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg
from scipy.io import savemat, loadmat
from mne.preprocessing import compute_current_source_density
from mne.channels import make_standard_montage
from mne.time_frequency import psd_welch, psd_multitaper
import seaborn as sns
import matplotlib
from mne.preprocessing import find_ecg_events
from numpy.random import randint
from scipy.stats import norm
from scipy import signal

mne.set_log_level('ERROR')
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

participants = ['p1','p2','p3','p4','p5','p6']
base_path = '../../SASS_data'
bads = np.load('bads.npy',allow_pickle=True).item()

for participant in participants:

    print(participant)
    path = base_path+'/'+participant+'/open.vhdr'
    raw_open = mne.io.read_raw_brainvision(path,preload=True)
    chname = 'O2'
    events_ecg = find_ecg_events(raw_open,ch_name='ECG')[0][10:-10][:,0]
    heartrate = 1/np.median(np.diff(events_ecg)*raw_open.times[1])
    raw_open.pick_channels([ch for ch in raw_open.ch_names if ch not in bads[participant]])
    raw_open.filter(5,15)
    # chidx = np.argmax(np.var(raw_open._data,axis=1))
    # chname = raw_open.ch_names[chidx]
    raw_open_envelope = raw_open.copy().apply_hilbert(envelope=True)
    events_open = mne.make_fixed_length_events(raw_open,duration=120)

    path = base_path+'/'+participant+'/no_stim.vhdr'
    raw_no_stim = mne.io.read_raw_brainvision(path,preload=True)
    raw_no_stim.pick_channels([ch for ch in raw_open.ch_names if ch not in bads[participant]])
    raw_no_stim.filter(5,15)
    events_no_stim = mne.make_fixed_length_events(raw_no_stim,duration=120)

    max_psd_open = None
    max_psd_no_stim = None
    max_zscores = None
    max_zscore = 0
    max_chname = None

    for chname in ['O2']:
        
        print('Channel number '+str(raw_open.ch_names.index(chname)), end='\r')
        raw_open_envelope_data = raw_open_envelope.copy().pick_channels([chname])._data.squeeze()
        ix_2sec = raw_open_envelope.time_as_index(2)[0]
        ix_t0 = raw_open_envelope.time_as_index(2)[0]
        epochs_ecg_envelope = signal.detrend(np.stack([raw_open_envelope_data[ev-ix_2sec:ev+ix_2sec] for ev in events_ecg],axis=0),type='constant')
        n_epochs = epochs_ecg_envelope.shape[0]
        average_ecg_envelope = np.mean(epochs_ecg_envelope,axis=0)
        averages = []
        for ix_iter in range(1000):
            events = randint(4.5*500,len(raw_open.times)-4.5*500,n_epochs)
            averages.append(np.mean(signal.detrend(np.stack([raw_open_envelope_data[ev-ix_2sec:ev+ix_2sec] for ev in events],axis=0),type='constant'),axis=0))
        averages = np.array(averages)

        pvalues = (averages<average_ecg_envelope[np.newaxis,:]).sum(0)/1000
        zscores = norm.ppf(pvalues)
        zscore = np.abs(zscores[ix_t0])
        if zscore > max_zscore:
            max_zscores = zscores
            max_zscore = zscore
            max_chname = chname

    epochs_no_stim = mne.Epochs(raw_no_stim,events_no_stim,tmin=0,tmax=120,baseline=None,preload=True).pick_channels([max_chname])
    epochs_open = mne.Epochs(raw_open,events_open,tmin=0,tmax=120,baseline=None,preload=True).pick_channels([max_chname])
    psd_no_stim,freqs = psd_multitaper(epochs_no_stim,bandwidth=0.05,fmin=8,fmax=12)
    psd_no_stim = psd_no_stim.mean((0,1))
    psd_open,freqs = psd_multitaper(epochs_open,bandwidth=0.05,fmin=8,fmax=12)
    psd_open = psd_open.mean((0,1))

    plt.figure(figsize=(12,8))
    plt.semilogy(freqs,psd_no_stim,label='No tACS')
    plt.semilogy(freqs,psd_open,label='tACS')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power ($V^2$/Hz)')
    ix_plus = np.argmin(np.abs(freqs-(10+heartrate)))
    ix_minus = np.argmin(np.abs(freqs-(10-heartrate)))
    y_plus = 2.5*np.min([psd_no_stim[ix_plus],psd_open[ix_plus]])
    y_minus = 2.5*np.min([psd_no_stim[ix_minus],psd_open[ix_minus]])
    plt.plot([10-heartrate,10+heartrate],[y_minus,y_plus],'v',c='g',label='Stimulation Frequency ± Heartrate',alpha=0.5,markersize=20)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.minorticks_off()
    plt.title(max_chname)
    plt.savefig('figs/open/'+participant+'_heartbeat_psd.pdf')
    plt.close()

    dt = 1/500
    times = np.arange(-2,2,dt)
    plt.figure(figsize=(12,8))
    plt.plot(times,max_zscores)
    plt.xlabel('Time (s)')
    plt.ylabel('Z-Score')
    sns.despine()
    plt.tight_layout()
    plt.title(max_chname)
    plt.savefig('figs/open/'+participant+'_heartbeat.pdf')