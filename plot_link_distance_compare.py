# %%
import numpy as np
from pandas import to_datetime
import scipy.sparse as sp
import scipy.optimize as so
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = "Myriad Pro"
mpl.rcParams['font.size'] = 9


# %%
def output_figcase(figcase):
    datanm = "spimv2"

    th = 1.5
    figcase = figcase
    # path = '/home/climate/hmwang/PycharmProjects/StandardIndex_SPI1_temp'
    if figcase == 0:
        path2, sig = '', 0.005
        title = 'Distribution of event synchonization link distances'
    elif figcase == 1:
        path2, sig = '', 0.001
        title = r"$\theta = 0.001$"
    elif figcase == 2:
        path2, sig = '', 0.01
        title = r"$\theta = 0.01$"
    elif figcase == 3:
        path2, sig = "dT0/", 0.005
        title = r"$\Delta T = 0$"
    elif figcase == 4:
        path2, sig = 'dT2/', 0.005
        title = r"$\Delta T = \pm2$"

    fig = plt.figure(figsize=(4.5, 3), tight_layout=True)
    ax1 = fig.add_subplot(111)
    fig.show()
    clrs = {"00": "tab:red", "11": "tab:blue", "01": 'tab:purple'}
    lbls = {"00": "Drought-Drought", "11": "Pluvial-Pluvial", "01": "Drought-Pluvial"}
    mkrs = {"00": "d", "11": "X", "01": "o"}

    density = False
    for direc in ["00", "11", "01"]:
        print('Direction ', direc)
        tic = time.time()



        # %% normal histogram
        # hist = np.histogram(angdist0, bins=200)  # bins = np.linspace(angdist0.min(), angdist0.max(), 200)
        # hist = np.histogram(angdist0, bins=200, density=True)
        # binx = hist[1][:-1] + (hist[1][1:] - hist[1][:-1]) / 2.0
        # print("Histogram Ended: {:.2f}s".format(time.time() - tic))
        # ax1.plot(binx, hist[0], color=clrs[direc], ls='None', marker='o',
        #          label=lbls[direc])

        # %% log-scale histogram
        # examine file "3link/{}disthist{}_{}_glb_event{}_{}.npz".format(path2, sig, datanm, direc, th) exists:
        try:
            loghist = np.load("3link/{}disthist{}_{}_glb_event{}_{}.npz".format(path2, sig, datanm, direc, th))
            logx = loghist['logx']
            logp = loghist['logp']
            print("Histogram Loaded: {:.2f}s".format(time.time() - tic))
        except:
            # %% Distance loading
            dist = sp.load_npz("3link/{}linkdist{}_{}_glb_event{}_{}.npz".format(path2, sig, datanm, direc, th))
            angdist0 = dist.data[dist.data > 0]
            # new calculation
            logbins = np.logspace(np.log10(angdist0.min()), np.log10(angdist0.max()), 50)
            loghist = np.histogram(angdist0, bins=logbins, density=density)
            print("Histogram Ended: {:.2f}s".format(time.time() - tic))
            logx = loghist[1][:-1] + (loghist[1][1:] - loghist[1][:-1]) / 2.0
            if density:
                logp = loghist[0] * (loghist[1][1:] - loghist[1][:-1])
            else:
                logp = loghist[0]
            np.savez("3link/{}disthist{}_{}_glb_event{}_{}.npz".format(path2, sig, datanm, direc, th), logx=logx, logp=logp)
        print("Sum of p is ", logp.sum())
        ax1.loglog(logx, logp, color=clrs[direc], ls='None', marker=mkrs[direc], alpha=0.5, ms=5,
                   label=lbls[direc])

    ax1.axvline(x=2500, color='gray', linestyle="--", alpha=.6,
                label="Distance=2500 km")
    ax1.set_xlim(50, 30000)
    ax1.set_ylim(10**-5 if density else 10**4)
    ax1.set(xlabel='Distance [km]', ylabel='Probability [-]' if density else "Frequency of link distances [-]",
            title=title)
    # xtks, xtklbs = ax1.get_xticks(), ax1.get_xticklabels()
    # xtks = np.concatenate((xtks, [2500]))
    ax1.set(xticks=[100, 1000, 10000])
    ax1.set(xticklabels=["$\\mathdefault{100}$", "$\\mathdefault{1000}$", "$\\mathdefault{10000}$"])
    # xtklbs.append(mpl.text.Text(2500, 0, "$\\mathdefault{2500}$"))
    # ax1.set(xlim=xlim, xticks=xtks, xticklabels=xtklbs)

    fig.savefig('pics/dist/{}densities_angdist{}_{}_glb_compare_{}.pdf'.format(path2, sig, datanm, th), bbox_inches='tight')
    print()

if __name__ == '__main__':
    for i in [0]: #range(5):
        output_figcase(i)