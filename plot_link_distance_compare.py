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


def plot_loghist(dist, cos_lat, title, figname, sample=None):
    angdist = dist.data
    angdist = angdist[angdist > 0]
    tic = time.time()

    logbins = np.logspace(np.log10(1), np.log10(20015), 50)
    # sam = np.random.choice(np.arange(dist0.size), int(1e8))
    if sample is None:
        dist_sam = angdist
    else:
        dist_sam = np.random.choice(angdist, size=sample, replace=False)
    loghist = np.histogram(dist_sam, bins=logbins, density=True)
    print("Histogram Ended: {:.2f}s".format(time.time() - tic))

    x_min, x_max = 500, 2500
    x_cut = loghist[1][(loghist[1] > x_min) & (loghist[1] < x_max)]  # fit power-law # TODO: 这里xy对嘛？
    y_cut = loghist[0][(loghist[1][:-1] > x_min) & (loghist[1][:-1] < x_max)]  # 实测 histogram
    plfit = lambda params: np.sum(np.abs(params[0] * x_cut ** (-params[1]) - y_cut))  # a*x_cut^(-b) - y_cut
    pl_params = so.fmin(plfit, [1, 1], disp=False)
    logx = loghist[1][:-1] + (loghist[1][1:] - loghist[1][:-1]) / 2.0
    x_prd = logx[logx > x_min]

    fig = plt.figure(figsize=(6, 4), tight_layout=True)
    ax1 = fig.add_subplot(111)
    ax1.loglog(logx[logx <= 2500], loghist[0][logx <= 2500], color='r', ls='None', marker='o',
               fillstyle='none', label=r'No weights ($d\leq$ 2500km)')
    ax1.loglog(logx[logx > 2500], loghist[0][logx > 2500], color='b', ls='None', marker='o',
               fillstyle='none', label=r'No weights ($d>$ 2500km)')
    ax1.loglog(x_prd, pl_params[0] * x_prd ** (-pl_params[1]), '--', alpha=0.5, lw=1.5, color='r',
               label=r'Powerlaw fit, $\alpha$ = %.3f' % (pl_params[1]))
    ax1.axvline(x=2500, color='m', alpha=.6)
    ax1.set(xlabel='Distance [km]', ylabel='PDF', title=title)
    ax1.legend(loc="lower right")
    fig.savefig(figname, dpi=300, bbox_inches='tight')
    fig.show()
    return None

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
    mkrs = {"00": "o", "11": "o", "01": "o"}
    density = False
    for direc in ["00", "11", "01"]:
        print('Direction ', direc)
        tic = time.time()

        # %% Distance loading
        dist = sp.load_npz("3link/{}linkdist{}_{}_glb_event{}_{}.npz".format(path2, sig, datanm, direc, th))
        angdist0 = dist.data[dist.data > 0]

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
        ax1.loglog(logx, logp, color=clrs[direc], ls='None', marker=mkrs[direc], alpha=0.7, ms=6,
                   label=lbls[direc])
        # if direc in ["00", "11"]:
        #     x_min, x_max = 500, 2500
        #     x_cut = loghist[1][(loghist[1] > x_min) & (loghist[1] < x_max)]  # fit power-law # TODO: 这里xy对嘛？
        #     y_cut = loghist[0][(loghist[1][:-1] > x_min) & (loghist[1][:-1] < x_max)] / angdist0.size  # 实测 histogram
        #     plfit = lambda params: np.sum(np.abs(params[0] * x_cut ** (-params[1]) - y_cut))  # a*x_cut^(-b) - y_cut
        #     pl_params = so.fmin(plfit, [1, 1], disp=False)
        #     logx = loghist[1][:-1] + (loghist[1][1:] - loghist[1][:-1]) / 2.0
        #     x_prd = logx[logx > x_min]
        #     ax1.loglog(x_prd, (pl_params[0] * x_prd ** (-pl_params[1])) * angdist0.size, '--',
        #                alpha=0.5, lw=1.5, color=clrs[direc])  #,label=r'Powerlaw fit, $\alpha$ = %.3f' % (pl_params[1])

        # # %% Ref plotting
        # dist = sp.load_npz("3link/{}linkdistref{}_{}_glb_event{}_{}.npz".format(path2, sig, datanm, direc, th))
        # angdist0 = dist.data[dist.data > 0]
        # logbins = np.logspace(np.log10(angdist0.min()), np.log10(angdist0.max()), 50)
        # loghist = np.histogram(angdist0, bins=logbins, density=density)
        # print("Histogram Ended: {:.2f}s".format(time.time() - tic))
        #
        # logx = loghist[1][:-1] + (loghist[1][1:] - loghist[1][:-1]) / 2.0
        # if density:
        #     logp = loghist[0] * (loghist[1][1:] - loghist[1][:-1])
        # else:
        #     logp = loghist[0]
        # print("Sum of p is ", logp.sum())
        # ax1.loglog(logx, logp, "--", color=clrs[direc], alpha=0.5,
        #            label="reference")

    ax1.axvline(x=2500, color='gray', linestyle="--", alpha=.6,
                label="Distance=2500 km")
    ax1.set_xlim(50, 30000)
    ax1.set_ylim(10**-5 if density else 10**4)
    ax1.set(xlabel='Distance [km]', ylabel='Probability [-]' if density else "Frequency [-]",
            title=title)
    # xtks, xtklbs = ax1.get_xticks(), ax1.get_xticklabels()
    # xtks = np.concatenate((xtks, [2500]))
    ax1.set(xticks=[100, 1000, 2500, 10000])
    ax1.set(xticklabels=["$\\mathdefault{10^2}$", "$\\mathdefault{10^3}$", "$\\mathdefault{2500}$",
                            "$\\mathdefault{10^4}$"])
    # xtklbs.append(mpl.text.Text(2500, 0, "$\\mathdefault{2500}$"))
    # ax1.set(xlim=xlim, xticks=xtks, xticklabels=xtklbs)

    # if figcase == 0:
    # else:
        # pass
        # ax1.legend(loc="lower right")
    fig.savefig('pics/dist/{}densities_angdist{}_{}_glb_compare_{}.pdf'.format(path2, sig, datanm, th), bbox_inches='tight')
    print()

if __name__ == '__main__':
    for i in [0]: #range(5):
        output_figcase(i)