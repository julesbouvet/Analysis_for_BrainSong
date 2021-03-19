from librosa import load
from librosa.feature import rms
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
from statsmodels.stats.weightstats import ttest_ind


def rms_audio(filename, sampling_rate, frame_length, display=True, save=False, savename='XXX'):

    # loads signal
    signal, sr = load(filename, sr=sampling_rate)

    print('sr',sr)

    # performs the rms
    rms_signal = rms(y=signal, frame_length=frame_length)[0]

    # display waveform and rms on a same plot
    if display==True:
        t_wf = np.linspace(0, len(signal)/sr, len(signal))
        t_rms = np.linspace(0, len(signal)/sr, rms_signal.shape[0])
        plt.figure('plot')
        plt.plot(t_wf, signal, label='Waveform')
        plt.plot(t_rms, rms_signal, label='RMS')
        plt.title(f'Waveform and RMS of {filename}')
        plt.legend()
        plt.show()

    if save==True:
        np.savez(savename, x=rms_signal)
        print('RMS saved')

    return rms_signal


def group_rms(rms, str):

    rms = signal.resample(rms, str.shape[0])

    print('rms', rms.shape)
    print('str', str.shape)

    # get index where STR=0 or 1
    idx_0 = np.where(str == 0)[0]
    idx_1 = np.where(str == 1)[0]

    # print(idx_0.shape, idx_1.shape)

    # get rms for STR=0 and STR=1
    rms_0 = [rms[k] for k in idx_0]
    rms_1 = [rms[k] for k in idx_1]

    return rms, rms_0, rms_1


def plot_rms(rms_0, rms_1):

    # sort rms
    rms_0.sort()
    rms_1.sort()

    # x axis
    x_0 = [k for k in range(len(rms_0))]
    x_1 = [k for k in range(len(rms_1))]

    # print(x_0.shape, x_1.shape)

    # display
    plt.bar(x_0, rms_0, label='STR=0')
    plt.bar(x_1, rms_1, label='STR=1', alpha=0.5)
    plt.title('RMS values when STR=0')
    plt.legend()
    plt.show()

    # boxplot
    rms_str = {'0': rms_0, '1': rms_1}
    fig, ax = plt.subplots()
    ax.boxplot(rms_str.values())
    ax.set_xticklabels(rms_str.keys())
    plt.title('Distribution of RMS for STR=0 and STR=1')
    plt.show()


def welch_test_rms(rms_0, rms_1):
    return ttest_ind(rms_0, rms_1)

