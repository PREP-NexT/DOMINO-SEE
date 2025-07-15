import time
import numpy as np
import scipy.sparse as sp

datanm = "spimv2"
# datanm = "spimv2fkt"  # For Fekete grid
vp = np.load("0data/prcp_validpoint_annual_100.npy")
# vp = np.load("0data/prcpfkt_validpoint_annual_100.npy")  # For Fekete grid
path = ''

th = 1.5
print("Start Time: ", time.asctime())
for direc in ["00", "01", "11"]:
    # direc = "01"
    sig = 0.005
    print("Direction ", direc)
    link = sp.load_npz("{}3link/link{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))
    print("Link Fraction: {:.2f}%".format(link.size / (vp.sum() ** 2) * 100))
    degree0 = np.array(link.sum(axis=1))  # 这里axis一定不要弄错了
    degree1 = np.array(link.sum(axis=0))
    np.savez("{}3link/linkdeg{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th),
            degree0=degree0, degree1=degree1)

    lnk_shr = sp.load_npz("{}3link/linkshr{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))
    print("Short Link Fraction: {:.2f}%".format(lnk_shr.size / (vp.sum() ** 2) * 100))
    degree_shr0 = np.array(lnk_shr.sum(axis=1))
    degree_shr1 = np.array(lnk_shr.sum(axis=0))
    np.savez("{}3link/linkdegshr{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th),
            degree0=degree_shr0, degree1=degree_shr1)

    lnk_tel = sp.load_npz("{}3link/linktel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th))
    print("Tele Link Fraction: {:.2f}%".format(lnk_tel.size / (vp.sum() ** 2) * 100))
    degree_tel0 = np.array(lnk_tel.sum(axis=1))
    degree_tel1 = np.array(lnk_tel.sum(axis=0))
    np.savez("{}3link/linkdegtel{}_{}_glb_event{}_{}.npz".format(path, sig, datanm, direc, th),
            degree0=degree_tel0, degree1=degree_tel1)
print("End Time: ", time.asctime())
