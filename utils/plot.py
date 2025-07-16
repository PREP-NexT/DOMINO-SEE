def inset_cbar_axes(ax, pad=0.01, width=0.03, length=1, loc="right"):
    if loc in ["right", "vertical"]:
        cax = ax.inset_axes([1+pad, 0, width, length])
    elif loc in ["bottom", "horizontal"]:
        cax = ax.inset_axes([0, -pad-width-0.05, length, width])
    return cax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    import numpy as np
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap