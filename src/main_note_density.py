from note_density_functions import note_density, barplot_note_density, histogram_note_density
from rms import rms_audio, group_rms, plot_rms, welch_test_rms
from numpy import savez, load, array
from analyse_str_onset import analyse_onsets, distribution_note_density


def main(filename, sr, frame_length, duration_wav_file, duration_sample, str_file, display_barplot=True,
         save=False, save_filename='rms_note_density_brainsong', analyse=True):

    # calculate rms
    rms_signal = rms_audio(filename=filename, sampling_rate=sr, frame_length=frame_length, save=False, savename='rms_perf3')

    # calculate note density
    nb_onsets, offsets = note_density(filename=filename, duration_wav_file=duration_wav_file,
                                      duration_sample=duration_sample, save=False, name_save_onsets='nb_onsets/onsets_for_eeg_perf1_44100')

    # plot barplot
    if display_barplot == True:
        barplot_note_density(nb_onset=nb_onsets)

    # plot histogram note density
    counts, bins = histogram_note_density(filename=filename, sampling_rate=sr, duration_sample=duration_sample, nb_onsets=nb_onsets,
                           nb_bins=10, offsets=offsets)

    # save rms and note density in a .npz file
    if save ==True:
        savez(save_filename, x=rms_signal, y=nb_onsets)

    # display the distribution of str and onsets
    if analyse == True:

        # load STR states
        str = load(str_file)['x'][:duration_wav_file]
        print('str', str.shape)

        # nb onset list to numpy
        np_nb_onsets = array(nb_onsets)

        # display distribution note density when STR=0 or 1
        distribution_note_density(str, np_nb_onsets)

        # seperate onset=0 and display distribution
        analyse_onsets(str, np_nb_onsets)

        #
        # rms analysis for STR=0 and STR=1
        #

        # get rms when STR=0 and STR=1
        rms_to_save, rms_0, rms_1 = group_rms(rms_signal, str)

        # plot rms histogram and boxplot
        plot_rms(rms_0, rms_1)

        # Welch's t-test
        print("Welch's t-test between RMS(STR=0) and RMS(STR=1) : \n ", welch_test_rms(rms_0, rms_1))


duration_wav = 583  # 789 sec clarinette_perf1 (583 w/troncature)/ 1048 sec clarinette_perf2 / 1401 for clarinette_perf3
audio_file = '../data/audio_data/clarinette_perf1.wav'
str_file = '../data/states_STR/states_STR_perf1.npz'
duration_sample = 1  # 583/128 pour eeg_1 / 1402/256 pour eeg_3
main(audio_file, sr=None, frame_length=2048, duration_wav_file=duration_wav, duration_sample=duration_sample, str_file=str_file)