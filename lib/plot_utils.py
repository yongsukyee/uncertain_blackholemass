##################################################
# PLOT UTILITIES
# Author: Suk Yee Yong
##################################################


from collections import namedtuple
import numpy as np


def cb_qualitative_ptol():
    """Colorblind friendly muted qualitative color scheme from: https://personal.sron.nl/~pault/"""
    cset = namedtuple('qualitative_ptol', 'rose indigo sand green cyan wine teal olive purple pale_grey black')
    return cset('#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
                '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD',
                '#000000')


def ax_square(ax, identitystd=None, identitystd_shade='gray', plot_identity=True):
    """Square axis with same xy limit. Optional plot identity line with standard deviation from the line"""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    min_lim, max_lim = np.min((xlim, ylim)), np.max((xlim, ylim))
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_aspect('equal', adjustable='box')
    # Target within standard deviation
    if isinstance(identitystd, (int, float)):
        identitystd_range = np.linspace(min_lim, max_lim, 10)
        ax.fill_between(identitystd_range, identitystd_range-identitystd, identitystd_range+identitystd, color=identitystd_shade, edgecolor='none', alpha=0.15, label=rf"$\sigma$={identitystd}")
    if plot_identity:
        ax.axline([min_lim, min_lim], slope=1, ls='--', color='gray', alpha=0.6)
    return ax

