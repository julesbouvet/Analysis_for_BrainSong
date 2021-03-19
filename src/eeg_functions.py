import numpy as np
from matplotlib.pyplot import show, subplots, title, savefig
from scipy import stats
from scipy import signal
from scikit_posthocs import posthoc_ttest
from rms import rms_audio


def alpha_analysis(alpha, idx_onsets):

    # get the alpha power for each bin
    dict_alpha = {}
    dict_alpha['silence'] = alpha[idx_onsets['silence']]
    dict_alpha['low density'] = alpha[idx_onsets['low density']]
    dict_alpha['medium density'] = alpha[idx_onsets['medium density']]
    dict_alpha['high density'] = alpha[idx_onsets['high density']]

    # boxplot
    fig, ax = subplots()
    ax.boxplot(dict_alpha.values())
    ax.set_xticklabels(dict_alpha.keys())
    title("Distribution of alpha's power for four rythms")
    # savefig('results_44100Hz/alpha_power_perf3')
    show()

    # kruskal_test
    print('Alpha Kruskal Test\n', kruskal_test(dict_alpha['silence'], dict_alpha['low density'], dict_alpha['medium density'], dict_alpha['high density']))

    # post hoc t test
    a = [dict_alpha['silence'], dict_alpha['low density'], dict_alpha['medium density'], dict_alpha['high density']]
    print('Alpha Posthoc t-test\n', posthoc_ttest(a))

    pass


def beta_analysis(beta, idx_onsets):
    # get the beta power for each bin
    dict_beta = {}
    dict_beta['silence'] = beta[idx_onsets['silence']]
    dict_beta['low density'] = beta[idx_onsets['low density']]
    dict_beta['medium density'] = beta[idx_onsets['medium density']]
    dict_beta['high density'] = beta[idx_onsets['high density']]

    # boxplot
    fig, ax = subplots()
    ax.boxplot(dict_beta.values())
    ax.set_xticklabels(dict_beta.keys())
    title("Distribution of beta's power for four rythms")
    # savefig('results_44100Hz/beta_power_perf3')
    show()

    # kruskal_test
    print('Beta Kruskal Test\n',
          kruskal_test(dict_beta['silence'], dict_beta['low density'], dict_beta['medium density'],
                       dict_beta['high density']))

    # post hoc t test
    a = [dict_beta['silence'], dict_beta['low density'], dict_beta['medium density'], dict_beta['high density']]
    print('Beta Posthoc t-test \n', posthoc_ttest(a))

    pass


def theta_analysis(theta, idx_onsets):
    # get the theta power for each bin
    dict_theta = {}
    dict_theta['silence'] = theta[idx_onsets['silence']]
    dict_theta['low density'] = theta[idx_onsets['low density']]
    dict_theta['medium density'] = theta[idx_onsets['medium density']]
    dict_theta['high density'] = theta[idx_onsets['high density']]

    # boxplot
    fig, ax = subplots()
    ax.boxplot(dict_theta.values())
    ax.set_xticklabels(dict_theta.keys())
    title("Distribution of theta's power for four rythms")
    # savefig('results_44100Hz/theta_power_perf3')
    show()

    # kruskal_test
    print('Theta Kruskal Test\n', kruskal_test(dict_theta['silence'], dict_theta['low density'], dict_theta['medium density'],dict_theta['high density']))

    # post hoc t test
    a = [dict_theta['silence'], dict_theta['low density'], dict_theta['medium density'], dict_theta['high density']]
    print('Theta Posthoc t-test\n', posthoc_ttest(a))

    pass


def kruskal_test(silence, low_den, med_den, high_dens):
    return stats.kruskal(silence, low_den, med_den, high_dens)


def eeg_note_density_analysis(eeg_npzfile, nb_onsets_npzfile):

    # load eeg
    eeg = np.load(eeg_npzfile)['x']

    # get power of alpha, beta and gamma
    alpha = eeg[:, 0]
    beta = eeg[:, 1]
    theta = eeg[:, 2]

    # load nb onsets
    n_onsets = np.load(nb_onsets_npzfile)['x']

    # onsets dictionnary {'rythm': idx w/ this rythm}
    max_onsets = max(n_onsets)
    threshold = [0, int(max_onsets+1)/3, 2*int(max_onsets+1)/3, max_onsets]
    print(threshold)
    idx_onsets = {}
    idx_onsets['silence'] = np.where(n_onsets == 0)[0].tolist()
    idx_onsets['low density'] = np.where(((n_onsets > 0) & (n_onsets < threshold[1])))[0].tolist()
    idx_onsets['medium density'] = np.where(((n_onsets > threshold[1]-1) & (n_onsets < threshold[2])))[0].tolist()
    idx_onsets['high density'] = np.where(n_onsets > threshold[2]-1)[0].tolist()

    print(idx_onsets)

    alpha_analysis(alpha, idx_onsets)
    beta_analysis(beta, idx_onsets)
    theta_analysis(theta, idx_onsets)

    pass


def eeg_rms_analysis(eeg_npzfile, rms_npzfile):

    # load eeg
    eeg = np.load(eeg_npzfile)['x']

    # get power of alpha, beta and gamma
    alpha = eeg[:, 0]
    beta = eeg[:, 1]
    theta = eeg[:, 2]

    # load rms
    rms = np.load(rms_npzfile)['x']

    # resample rms
    rms = signal.resample(rms, alpha.shape[0])

    # calculate Spearman correlation
    print('Spearman correlation for Alpha Band\n', stats.spearmanr(alpha, rms))
    print('\n \nSpearman correlation for Beta Band\n', stats.spearmanr(beta, rms))
    print('\n \nSpearman correlation for Theta Band\n', stats.spearmanr(theta, rms))

    pass

# rms_audio('audio_data/caroleeg_perf2.wav', sampling_rate=None, frame_length=2048, save=True, savename='rms/rms_every_instruments_perf2')
# nb_onsets_npzfile = 'nb_onsets/onsets_for_eeg_perf3_44100.npz'
# eeg_npzfile = 'EEG/eeg_perf3.npz'
# rms_npzfile = 'rms/rms_every_instruments_perf3_44100.npz'

# eeg_rms_analysis(eeg_npzfile, rms_npzfile)
# eeg_note_density_analysis(eeg_npzfile, nb_onsets_npzfile)