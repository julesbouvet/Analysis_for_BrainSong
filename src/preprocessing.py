import numpy as np
import librosa, librosa.display
import librosa.core as core
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy import signal


def load_signal(filename):
    sig, sr = core.load(filename, sr=None, mono=False)
    return sig, sr


def make_sample(nb_sample, signal, sr, waveform=False):
    len_record = signal.shape[1]
    len_sample = int(len_record/nb_sample)
    print(len_sample)
    samples = np.zeros((nb_sample, 2,  len_sample))

    for k in range(nb_sample):
        samples[k][0] = signal[0][k*len_sample: (k+1)*len_sample]
        samples[k][1] = signal[1][k * len_sample: (k + 1) * len_sample]

    if waveform == True:
        sp = 1
        for sample in samples:
            fig_size = (15, 10)
            name = f"Waveform_{sp}"
            wv = display_waveform(fig_size, sample, sr, name)
    # save numpy array
    np.savez('all_sample', x=samples)

    return samples


def make_sample_with_len_sample(signal, sr, len_sample, nb_samples, display_wf):
    samples = []
    # make a loop of samples creation:
    for nb in range(nb_samples):
        idx_max = signal.shape[1]-(len_sample+1)
        idx_start = np.random.randint(0, idx_max)
        sample = signal[0][idx_start:idx_start+len_sample]
        samples.append(sample)

    if display_wf == True:
        fig, ax = plt.subplots(3, 3)
        for ii in range(3):
            for jj in range(3):
                idx_sample = ii * 3 + jj + 1
                ax[ii, jj].plot(samples[idx_sample][:200], color="tab:blue")
                ax[ii, jj].set_xticks([])
                ax[ii, jj].set_yticks([])
        plt.show()
    pass


def display_waveform (FIG_SIZE, signal, sample_rate, name, show_axis=True, save=False, display=False):
    figure = plt.figure(figsize=FIG_SIZE)
    librosa.display.waveplot(signal, sample_rate, alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(name)
    if show_axis == False:
        figure.gca().spines["right"].set_visible(False)
        figure.gca().spines["top"].set_visible(False)
        figure.gca().spines["left"].set_visible(False)
        figure.gca().spines["bottom"].set_visible(False)
        plt.yticks([])

    if save == True:
        plt.savefig('samples/'+name)
    if display == True:
        plt.show()
    return figure

def display_spectrum(signal, sample_rate, FIG_SIZE):
    # FFT -> power spectrum
    # perform Fourier transform
    fft = np.fft.fft(signal)

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # create frequency variable
    f = np.linspace(0, sample_rate, len(spectrum))

    # take half of the spectrum and frequency
    left_spectrum = spectrum[:int(len(spectrum)/2)]
    left_f = f[:int(len(spectrum)/2)]

    # plot spectrum
    plt.figure(figsize=FIG_SIZE)
    plt.plot(left_f, left_spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")
    plt.show()


def spectrogram(sample_rate, signal):
    # STFT -> spectrogram
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples

    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sample_rate
    n_fft_duration = float(n_fft)/sample_rate

    print("STFT hop length duration is: {}s".format(hop_length_duration))
    print("STFT window duration is: {}s".format(n_fft_duration))

    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)

    return spectrogram


def log_spectrogram(spectrogram):

    # apply logarithm to cast amplitude to Decibels
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    return log_spectrogram

def displayTime(startFrame, endFrame, sr):
    print(' start time: ' + str(startFrame/sr) + ', end time: ' + str(endFrame/sr))

def split_silence(signal, sr):
    nonMuteSections = librosa.effects.split(signal)
    for i in nonMuteSections:
        displayTime(i[0], i[1], sr)




def display_log_spectrogram(FIG_SIZE, log_spectrogram, sample_rate, hop_length=512):
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    plt.show()


def run(load=False, open=True,  sample=False, silence=False, display=False):

    if load == True:

        # define figure size and filename
        FIG_SIZE = (15, 10)
        filename = "../data/audio_data/caroleeg_perf1.wav"

        # load signal
        signal, sample_rate = load_signal(filename)
        print(signal.shape, sample_rate)

    if open == True:
        npzfile = np.load('../../Generating Sound with NN/Test with BrainSong/all_sample.npz')
        samples = npzfile['arr_0']
        print(samples.shape)

    if sample == True:
        samples = make_sample(100, signal, sample_rate)

        print(samples.shape)

        # save signal as txt file
        # np.savetxt('signal',signal, delimiter=';')

        # display waveform signal
        # wv = display_waveform(FIG_SIZE, signal, sample_rate)
    if silence == True:

        #
        # split silence
        #
        split_silence(signal, sr=sample_rate)


    if display == True:
        x = np.loadtxt('signal', delimiter=';')
        print(x.shape)
        print(type(x))
        image = imread('waveform.png')[:, :, 2]
        print(image.shape)
        plt.imshow(image)
        plt.show()

