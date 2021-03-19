import librosa
import numpy as np
import matplotlib.pyplot as plt
import preprocessing
import soundfile as sf
from sklearn.preprocessing import QuantileTransformer

###
# Method:  Use librosa.load on 5 seconds samples
###


def note_density(filename, duration_wav_file, duration_sample, save=False, name_save_onsets='number_onset'):

    # calculate an offset list and duration
    len_wav_sec = duration_wav_file  # 855 s
    duration = duration_sample
    nb_samples = int(len_wav_sec/duration)
    offsets = [i*duration for i in range(nb_samples)]

    # create an empty list which will contain the number of onset detection for each sample
    nb_onsets = []

    # calculate the nb of onset detections
    for sample in range(len(offsets)):

        # load the signal and the sample rate
        y, sr = librosa.load(filename, sr=None, offset=offsets[sample], duration=duration)

        # detect if silence
        mean = np.mean(np.abs(y))

        if mean>0.0005:
            # detect onsets
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

            # add number of onset to the list nb_onsets
            nb_onsets.append(len(onset_frames))

        else:
            nb_onsets.append(0)

    if save == True:
        np.savez(name_save_onsets, x=nb_onsets)
        print('Note Density Saved')

    return nb_onsets, offsets


def histogram_note_density(filename, sampling_rate, duration_sample, nb_onsets, offsets, nb_bins, save_audio_file=False):

    # plot the histogram
    plt.figure()
    counts, bins, hist = plt.hist(nb_onsets, nb_bins)
    plt.title('Histogram Note Density')
    plt.show()

    # get the bin's value of each sample
    sample_bins_value = np.digitize(nb_onsets, bins)

    for bins in range(1, len(bins)):
        # use np.where() to find sample in a precise bin
        np_sample_bins_value = np.array(sample_bins_value)
        sample = np.where(np_sample_bins_value == bins)[0]

        # select 5 samples randomly
        np.random.shuffle(sample)
        sample = sample[:5]

        # save audio file:
        if save_audio_file == True:
            for k in (sample):
                signal, sr = librosa.load(filename, sr=sampling_rate, offset=offsets[k], duration=duration_sample)
                sf.write(f'test_{bins}_{k}.wav', signal, sr)

                # display the waveform with onset detection if plot_display_onset_detection=True
                display_onset_detection = False

                if display_onset_detection == True:
                    preprocessing.display_waveform((15, 10), signal, sr, name='clarinette')
                    onset_frames = librosa.onset.onset_detect(y=signal, sr=sr)
                    onset = librosa.frames_to_time(onset_frames, sr=sr)
                    for t in onset:
                        plt.axvline(x=t)
                    plt.show()

    return counts, bins


def quantile_transformer(nb_onsets, nb_quantiles, nb_bins):
    qt = QuantileTransformer(n_quantiles=nb_quantiles, output_distribution='normal')
    new_onsets = qt.fit_transform(np.array(nb_onsets).reshape(-1 ,1))
    print(new_onsets)
    plt.figure()
    counts, bins, hist = plt.hist(new_onsets.tolist())
    plt.title('New Histogram Note Density')
    plt.show()
    return nb_onsets


def barplot_note_density(nb_onset):

    # make the x-axis
    len_signal = len(nb_onset)
    x_barplot = [t for t in range(len_signal)]

    #plot barplot
    plt.figure('barplot')
    plt.bar(x_barplot, nb_onset)
    plt.xlabel('Samples')
    plt.ylabel('Number of onsets')
    plt.title('Number of onsets per sample')
    plt.show()

