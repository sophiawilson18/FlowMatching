import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def add_field_plot(obs, gen, row_gt, row_pred, channel, label, args, cmap='RdBu'): # (12, 5, 4, 256, 512)
    global f, axarr, ncol  # ensure these are accessible

    # Compute symmetric color range
    data_gt = obs[0, :, channel]
    data_pred = gen[0, :, channel]
    vmax = np.max(np.abs(data_gt))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for i in range(ncol):
        axarr[row_gt, i].imshow(obs[0, i, channel], cmap=cmap, norm=norm) # plot first batch, loops over all snapshots
        axarr[row_pred, i].imshow(gen[0, i, channel], cmap=cmap, norm=norm)

        for ax in [axarr[row_gt, i], axarr[row_pred, i]]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Only title on the top row
        if row_gt == 0:
            axarr[row_gt, i].set_title("Cond." if i < args.condition_snapshots else "Rollout", fontsize=12)

    # Set single ylabel on the far-left of each row pair
    axarr[row_gt, 0].set_ylabel(f"GT\n{label}", fontsize=12)
    axarr[row_pred, 0].set_ylabel(f"Pred\n{label}", fontsize=12)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Adjust colorbar vertical placement per row group
    bar_height = 0.18
    bar_top = 0.94 - 0.24 * (row_pred // 2)
    cbar_ax = f.add_axes([0.88, bar_top - bar_height, 0.015, bar_height])
    f.colorbar(sm, cax=cbar_ax)