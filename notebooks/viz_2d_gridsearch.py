EPS = 0.001

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt

def do_gridsearch_and_visualize(
        calc_cost=None,
        cost_kwargs={},
        G=10,
        H=10,
        xvar_name='alpha',
        yvar_name='beta',
        xvar_lims=(EPS, 1.0-EPS),
        yvar_lims=(EPS, 1.0-EPS),
        xvar_fmtstr='{:.4f}',
        yvar_fmtstr='{:.4f}',
        n_levels=3,
        vmin=0.0,
        vmax=0.2,
        ):
    '''
    '''
    _, axgrid = plt.subplots(nrows=1, ncols=n_levels, figsize=(4 * n_levels, 4))
    axgrid = np.atleast_1d(axgrid)

    for zoom_level in range(n_levels):
        if zoom_level > 0:
            # Zoom in on most promising region found so far
            xvar_lims = np.percentile(
                xvar_G[np.maximum(minrow - 2,0):minrow + 2], [0, 100])
            yvar_lims = np.percentile(
                yvar_H[np.maximum(mincol - 2,0):mincol + 2], [0, 100])
            G = int(np.ceil(G*1.4))
            H = int(np.ceil(H*1.4))
    
        xvar_G = np.linspace(xvar_lims[0], xvar_lims[1], G)
        yvar_H = np.linspace(yvar_lims[0], yvar_lims[1], H)
        print("\nAt Level %d" % zoom_level)
        print("-----------")
        print("xvar %s grid has %d elements" % (xvar_name, G))
        print(xvar_G[:3].tolist(),)
        print("...",)
        print(xvar_G[-3:].tolist())
        print("yvar %s grid has %d elements" % (yvar_name, H))
        print(yvar_H[:3].tolist(),)
        print("...",)
        print(yvar_H[-3:].tolist())
    
        yvar_GH, xvar_GH = np.meshgrid(yvar_H, xvar_G);
        assert xvar_GH.shape == (G,H)

        cost_GH = np.zeros((G,H))
        for gg in range(G):
            for hh in range(H):
                cost_GH[gg,hh] = calc_cost(
                    xvar_GH[gg,hh], yvar_GH[gg,hh], **cost_kwargs)

        minrow, mincol = np.unravel_index(np.argmin(cost_GH, axis=None), cost_GH.shape)
        best_x = xvar_GH[minrow, mincol]
        best_y = yvar_GH[minrow, mincol]
        best_cost = cost_GH[minrow, mincol]
        ax = axgrid[zoom_level]
        ax.pcolor(xvar_GH, yvar_GH, cost_GH,
            shading='nearest', cmap='hot_r',
            vmin=vmin, vmax=vmax);
        ax.plot(best_x, best_y, '+', color='k', mew=2, markersize=20);
        ax.set_xlabel('%s' % xvar_name);
        ax.set_ylabel('%s' % yvar_name);
        ax.set_title("Level %d Cost %.5g \n %s=%s, %s=%s" % (
            zoom_level, best_cost,
            xvar_name, xvar_fmtstr.format(best_x),
            yvar_name, yvar_fmtstr.format(best_y)))

    return best_x, best_y, best_cost