from numpy import where
from matplotlib.pyplot import bar, show, subplots, title, hist, legend, savefig
from scipy import stats

def analyse_onsets(str, onsets):

    # detect onset==0 and non zeros
    idx_0 = where(onsets==0)[0]
    idx_non_0 = where(onsets!=0)[0]

    #
    ### analyse zeros' onset
    #

    # find nb str = 0 and 1 for onset =0
    str_0_0 = where(str[idx_0]==0)[0]
    str_1_0 = where(str[idx_0] == 1)[0]

    count_zero_onset = [str_0_0.shape[0], str_1_0.shape[0]]
    x_bar = ['0', '1']

    # plot barplot
    bar(x_bar, count_zero_onset)
    title('Distribution of STR w/ zero onset detected ')
    show()

    #
    ### analyse non-zeros' onset
    #

    # find nb str = 0 and 1 for onset !=0
    str_0_1 = where(str[idx_non_0] == 0)[0]
    str_1_1 = where(str[idx_non_0] == 1)[0]

    count_nonzero_onset = [str_0_1.shape[0], str_1_1.shape[0]]

    # plot barplot
    bar(x_bar, count_nonzero_onset)
    title('Distribution of STR w/ at least one onset detected ')
    show()

    # boxplot
    str_onset_non_zero = {'0': [onsets[i] for i in str_0_1], '1': [onsets[i] for i in str_1_1]}

    # plot boxplot
    fig, ax = subplots()
    ax.boxplot(str_onset_non_zero.values())
    ax.set_xticklabels(str_onset_non_zero.keys())
    title('Distribution of non-zero onsets for STR=0 and 1')
    show()

    # t-test
    print('t-test', stats.ttest_ind(str[idx_non_0], onsets[idx_non_0]))


def distribution_note_density(str, onsets):

    # get idex where STR=0 or 1
    idx_0 = where(str == 0)[0]
    idx_1 = where(str == 1)[0]

    # get onsets values for STR=0 or 1
    onset_0 = [onsets[idx] for idx in idx_0]
    onset_1 = [onsets[idx] for idx in idx_1]

    onset_str = {'0': onset_0, '1': onset_1}

    # plot boxplot
    fig, ax = subplots()
    ax.boxplot(onset_str.values())
    ax.set_xticklabels(onset_str.keys())
    title('Distribution of onsets for STR=0 and STR=1')
    show()

    # plot histogram
    hist(onset_0, label='0')
    hist(onset_1, alpha=0.3, label='1')
    legend()
    title('Distribution of onsets for STR=0 and STR=1')
    # savefig('results_44100Hz/hist_note_density_str_perf1_new')
    show()