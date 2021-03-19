import numpy as np
from hmmlearn.hmm import GMMHMM, GaussianHMM


def from_npz_to_states (npzfile,name_savefile, res=1, measure='subjtime',delaytime=1):
    measr = np.load(npzfile)[measure]
    # Quantization
    subt = quantize(measr, res=res, delaytime=delaytime)
    y, states, mus, sigmas, P = fit_HMM(subt)
    np.savez('states_STR/'+name_savefile, x=states)
    return y, states, mus, sigmas, P


def quantize(vector, res=0.3, endtime=2, delaytime=1):
    vectimes = vector[:, 0]
    maxtime = max(vectimes)

    alltimes = np.arange(0, maxtime - endtime, res)
    qvec = []
    ## loop over all time segments between two instants of length "res"

    prev_val = vector[0, 1]

    for i, curtime in enumerate(alltimes[:-2]):

        ind = np.argwhere((vectimes > curtime) & (vectimes < alltimes[i + 1]))

        if len(ind) == 0:
            qvec.append([curtime, prev_val])
        else:
            qvec.append([curtime, np.mean(vector[ind, 1])])
            prev_val = np.mean(vector[ind, 1])
    return np.stack(qvec) + np.repeat(np.array([[delaytime], [0]]), len(qvec), axis=1).T


def fit_HMM(feedback, n_components=2, hmm='GaussianHMM'):
    assert hmm == 'GaussianHMM' or hmm == 'GMMHMM', "You have to choose between GaussianHMM or GMMHMM"

    y = feedback[:, 1]

    if hmm == 'GaussianHMM':
        model = GaussianHMM(n_components=n_components)
    else:
        model = GMMHMM(n_components=n_components)

    model.fit(y.reshape(len(y), 1))

    states = model.predict(y.reshape(len(y), 1))
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    return y, states, mus, sigmas, P


def run():
    y, states, mus, sigmas, P = from_npz_to_states('EEG/caroleeg_perf2.npz', 'states_STR_perf2')
    print(states.shape)


run()