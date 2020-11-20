import math
import string

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error as MSE

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ternary
from ternary.helpers import simplex_iterator

from models import ViscNet, ViscNetVFT
from data import (viscosity_df, composition_df, test_idx, train_val_idx,
                  train_idx, val_idx, compounds)


###############################################################################
#                                    Config                                   #
###############################################################################

plot_file_ext = 'png'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',
    'axes.formatter.limits': [-2, 5],
    'axes.formatter.useoffset': False,
    'axes.formatter.use_mathtext': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.framealpha': 1,
    'legend.edgecolor': 'k',
    'figure.figsize': [4 ,4],
    'figure.dpi': 150,
    'figure.subplot.wspace': 0.3,
    'figure.subplot.hspace': 0.3,
    'errorbar.capsize': 5,
    'mathtext.fontset': 'dejavuserif',
})

model_dict = {
    'ViscNet': f'files/viscnet.pt',
    'ViscNet-Huber': f'files/viscnet_huber.pt',
    'ViscNet-VFT': f'files/viscnet_vft.pt',
}

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


###############################################################################
#                                  Functions                                  #
###############################################################################

def learning_curve(x, y_train, y_val, xlim, model_name):

    fig, axe = plt.subplots(ncols=1, nrows=1)

    axe.plot(x, y_train, label='Training dataset')
    axe.plot(x, y_val, label='Validation dataset')

    axe.set_xlabel('Epoch')
    axe.set_ylabel('Average MSE Loss')

    axe.set_yscale('log', nonposy='clip')
    axe.set_xlim(xlim)
    axe.legend()

    fig.savefig(
        rf'plots/{model_name}/learning_curve'
        rf'.{plot_file_ext}',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)


def boxplot(residual, model_name):

    bins = np.arange(-1.5, 13.5, 1) 

    binplot = []
    dataplot = []

    binjump = 0  
    binplotjump = 2

    for b in range(len(bins) - 1):
        logic1 = y_true >= bins[b]
        logic2 = y_true < bins[b + 1]
        logic = np.logical_and(logic1, logic2)

        Y = y_true[logic]

        if binjump % binplotjump == 0:
            binplot.append(str(int((bins[b] + bins[b + 1]) / 2)))
        else:
            binplot.append('')

        binjump += 1

        dataplot.append(residual[logic])

    fig, axe = plt.subplots(ncols=1, nrows=1, figsize=(3.5, 3.5))

    axe.axhline(0, c='black', ls='--', alpha=0.5)

    axe.boxplot(
        dataplot,
        showfliers=False,
        whis=[16.5, 100 - 16.5],  
        notch=True,
        bootstrap=5000,
        patch_artist=True,
        boxprops={'facecolor': 'mediumaquamarine'},
    )

    axe.set_xticklabels(binplot)
    axe.set_xlabel('$\log_{10} (\eta)$  [$\eta$ in Pa.s]')
    axe.set_ylabel('Prediction residual')

    fig.savefig(
        rf'plots/{model_name}/boxplot'
        rf'.{plot_file_ext}',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )
    plt.close(fig)


def compound(residual, composition_df, model_name):

    compounds = composition_df.columns.values

    quantity = composition_df.astype(bool).sum(axis=0).values
    compounds = compounds[quantity.astype(bool)]
    quantity = \
        composition_df.reindex(compounds,axis=1).astype(bool).sum(axis=0).values

    quantity, compounds = zip(*sorted(zip(quantity, compounds)))

    means = []
    stds = []
    xticks = []
    xnumber = []

    n = False

    for q, c in zip(quantity, compounds):
        logic = composition_df[c].values > 0
        means.append(np.mean(residual[logic]))
        stds.append(np.std(residual[logic]))

        xnumber.append(q)
        if n:
            xticks.append(f'{c.translate(subscript)}')
        else:
            xticks.append(f'\n{c.translate(subscript)}')

        n = not n


    fig, axe = plt.subplots(ncols=1, nrows=1, figsize=(10/1.2,5/1.2))

    axe.axhline(0, c='black', ls='--', alpha=0.5)
    axe.errorbar(xticks, means, yerr=stds, marker='o', ls='none')

    axe.set_xticklabels(xticks, rotation=0, ha='center', fontsize=10)

    axe.set_ylabel('Prediction residual')

    axe.grid(color='black', ls=':', alpha=0.2)

    axe.set_xlim(-1, len(xticks))

    axt = axe.secondary_xaxis('top')
    axt.set_xticks(range(len(xticks)))
    axt.set_xticklabels(xnumber, rotation=90, ha='center', fontsize=10)

    fig.savefig(
        rf'plots/{model_name}/Error_compound'
        rf'.{plot_file_ext}',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)


def distance(compounds, viscosity_df, test_idx, train_val_idx, model,
             model_name):

    df = viscosity_df.loc[test_idx].drop_duplicates(subset=['ID'])
    IDs = df['ID'].values

    df_tst = df.reindex(compounds, axis=1).values
    df_tv = viscosity_df.loc[train_val_idx].drop_duplicates(subset=['ID'])
    df_tv = df_tv.reindex(compounds, axis=1).values

    min_dist = cdist(df_tv, df_tst, metric='canberra').min(axis=0)

    RMSE = []

    for n, ID in enumerate(IDs):

        comp = viscosity_df[compounds][viscosity_df.ID == ID]
        T = viscosity_df['T'][viscosity_df.ID == ID].values
        y_true = viscosity_df['log_visc'][viscosity_df.ID == ID]

        with torch.no_grad():
            y_pred = model.viscosity_from_composition(comp, T)

        RMSE.append(MSE(y_true, y_pred, squared=False))

    fig, axe = plt.subplots(ncols=1, nrows=1)

    axe.axhline(0.5, c='gray', alpha=0.5, ls='--')
    axe.plot(min_dist, RMSE, marker='o', ls='none', markeredgecolor='black',
             alpha=0.8)

    axe.set_xscale('log', nonposx='clip')

    axe.set_xlabel('Distance')
    axe.set_ylabel('RMSE')

    fig.savefig(
        rf'plots/{model_name}/distance'
        rf'.{plot_file_ext}',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)

    
def correlation(y_true, y_pred, model_name):
    
    fig, axe = plt.subplots(ncols=1, nrows=1)

    s = 0.3  # size of the square
    xedges = np.arange(
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
        s
    )
    yedges = xedges

    H, xedges, yedges = np.histogram2d(y_true, y_pred, bins=(xedges, yedges))
    H = H.T  
    X, Y = np.meshgrid(xedges, yedges)
    cm = axe.pcolormesh(X, Y, H, cmap='viridis_r', norm=LogNorm())
    cb = fig.colorbar(cm, ax=axe, fraction=0.046, pad=0.04)
    cb.set_label('Density')
    axe.plot(
        [min(y_true)-5, max(y_true)+5], [min(y_true)-5,max(y_true)+5],
        ls='-', c='k', lw=0.7, alpha=0.7
    )

    axe.set_xlim(-3, 16)
    axe.set_ylim(-3, 16)

    label= '$\log_{10} (\eta)$'

    axe.set_xlabel(rf'Reported {label}')
    axe.set_ylabel(rf'Predicted {label}')

    size = '33%'
    ax2 = inset_axes(
        axe,
        width=size,
        height=size,
        loc=2,
    )

    ax2.set_xlabel('Pred. residual')
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_label_position('right')

    cols = np.linspace(-5, 5, 100)
    residuals = y_true - y_pred
    n, bins, patches = ax2.hist(
        residuals,
        cols,
        fc='gray',
        weights=None,
        align='mid',
        log=False,
        ec='none',
        fill=True,
        histtype='step',
        alpha=0.7,
    )

    ax2.set_xlim([-2.5, 2.5])
    ax2.xaxis.set_ticks([-2, -1, 0, 1, 2])

    fig.savefig(
        rf'plots/{model_name}/2D_histogram'
        rf'.{plot_file_ext}',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)


def individual(compounds, viscosity_df, model, num_samples, model_name,
               folder_name='individual', confidence=95):

    q = [(100-confidence)/2, 100 - (100-confidence)/2]

    for ID in viscosity_df.ID.unique():

        df = viscosity_df[viscosity_df.ID == ID]
        x = df['T'].values
        y = df['log_visc'].values
        composition_dict = df.reindex(compounds, axis=1).iloc[0].to_dict()
        composition_dict = {k:v for k,v in composition_dict.items() if v > 0}

        title = '.'.join([f'{v*100:.0f}{c.translate(subscript)}'
                 for c,v in composition_dict.items()])

        T = torch.linspace(min(x), max(x), 50, requires_grad=False)
        bands = \
            model.viscosity_bands_single(composition_dict, T, q, num_samples)
            
        fig, axe = plt.subplots(ncols=1, nrows=1)

        axe.plot(x, y, marker='o', ls='none', markeredgecolor='black',)
        for band in bands:
            axe.plot(T, band, ls='--', c='tab:red')

        axe.set_title(title)
        axe.set_xlabel('$T$  [K]')
        axe.set_ylabel(r'$\log_{10} (\eta)$  [$\eta$ in Pa.s]')

        fig.savefig(
            rf'plots/{model_name}/{folder_name}/{str(ID).zfill(3)}'
            rf'.{plot_file_ext}',
            dpi=150,
            bbox_inches='tight',
            pad_inches=2e-2,
        )

        plt.close(fig)


def outofband(compounds, viscosity_df, model, num_samples, model_name,
              folder_name='individual', confidence=95):

    q = [(100-confidence)/2, 100 - (100-confidence)/2]

    x = viscosity_df['T'].values
    y = viscosity_df['log_visc'].values
    composition_df = viscosity_df.reindex(compounds, axis=1)

    xt = torch.from_numpy(x)
    bands = model.viscosity_bands_batch(composition_df, xt, q, num_samples)

    above_lower = np.greater_equal(y, bands[0])
    below_upper = np.less_equal(y, bands[1])
    between_bands = np.logical_and(above_lower, below_upper)

    num_between = sum(between_bands)
    num_total = len(x)
    fraction_between = num_between / num_total

    oob_residual = np.zeros(len(x))

    oob_residual[~between_bands] = \
        (~above_lower * (y - bands[0]))[~between_bands] + \
        (~below_upper * (y - bands[1]))[~between_bands]

    fig, axe = plt.subplots(ncols=1, nrows=1)

    n, bins, patches = axe.hist(
        oob_residual,
        np.arange(-1.125, 1.125, 0.25),
        align='mid',
        ec='black',
    )

    axe.set_xlabel('Out of band residual')
    axe.set_ylabel(r'Number of data points')

    fig.savefig(
        rf'plots/{model_name}/{folder_name}/out_of_band'
        rf'.{plot_file_ext}',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)

    return oob_residual


def distance_between_data_and_train(data, train, metric='canberra', neighbor=1):
    if neighbor == 1:
        dist = cdist(train, data, metric).min(axis=0)
    else:
        dist = cdist(data, train, metric)
        dist = np.partition(dist, neighbor)[:, neighbor-1]

    return dist
        

def ternaryplot(comp1, comp2, comp3, compounds, train_val_viscosity_df,
                model, model_name, max_distance=0.5):

    scale = 100
    neighbor = 1
    distance_to_plot = max_distance
    metric = 'canberra'

    parameters_dict = {
        'Tg': {
            'index': 1,
            'label': '$T_{g}$  [K]',
            'round': 50,
            'bad_val': 400,
            'format': lambda x, pos: x if x != 400 else 'OoD',
        },

        'm': {
            'index': 2,
            'label': 'Fragility index',
            'round': 2,
            'bad_val': 10,
            'format': lambda x, pos: x if x != 10 else 'OoD',
        },
    }

    chemtrans = {
        'SiO2': r'$\mathrm{SiO_2}$',
        'Na2O': r'$\mathrm{Na_2O}$',
        'Li2O': r'$\mathrm{Li_2O}$',
        'CaO': r'$\mathrm{CaO}$',
        'Al2O3': r'$\mathrm{Al_2O_3}$',
    }

    base_ternary = [(i, j, k) for i, j, k in simplex_iterator(scale)]

    df_ternary = pd.DataFrame(base_ternary, columns=[comp1, comp2, comp3])
    df_ternary = df_ternary.reindex(compounds, axis=1).fillna(0)
    data_ternary = df_ternary.div(df_ternary.sum(axis=1), axis=0).values

    df_train_val = train_val_viscosity_df.drop_duplicates(subset=['ID'])
    data_train_val = df_train_val.reindex(compounds, axis=1).values

    min_dist = distance_between_data_and_train(
        data_ternary,
        data_train_val,
        metric=metric,
        neighbor=neighbor,
    )

    with torch.no_grad():
        params_pred = model.viscosity_params_from_composition(df_ternary)

    for param_name, param_info in parameters_dict.items():

        param_values = params_pred[param_name]

        round_base = param_info['round']
        label = param_info['label']
        bad_val = param_info['bad_val']

        heatmap = \
            {k: round_base * round(float(v)/round_base) if d <= max_distance \
             else bad_val \
             for k, v, d in zip(base_ternary, param_values, min_dist)}

        fig, tax = ternary.figure(scale=scale)
        fig.set_size_inches(8/1.5, 10/1.5)

        cb_kwargs = {
            "shrink" : 0.9,
            "pad" : 0.10,
            "aspect" : 30,
            "orientation" : "horizontal",
            'format': param_info['format'],
        }

        tax.heatmap(
            heatmap,
            style="hexagonal",
            use_rgba=False,
            colorbar=True,
            cbarlabel=label,
            cmap='pink_r',
            cb_kwargs=cb_kwargs,
        )

        tax.clear_matplotlib_ticks()
        tax.get_axes().axis('off')
        tax.boundary()

        tax.left_axis_label(
            chemtrans.get(comp3, comp3) + ' mol%',
            fontsize=10,
            offset=0.14,
        )
        tax.right_axis_label(
            chemtrans.get(comp2, comp2) + ' mol%',
            fontsize=10,
            offset=0.14,
        )
        tax.bottom_axis_label(
            chemtrans.get(comp1, comp1) + ' mol%',
            fontsize=10,
            offset=0.14,
        )

        tax.ticks(axis='lbr', linewidth=1, multiple=scale/10, offset=0.03)

        fig.savefig(
            rf'plots/{model_name}/ternary_{comp1}_{comp2}_{comp3}'
            f'_{param_name}.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=2e-2,
        )

        plt.close(fig)


def allliquids(compounds, viscosity_df, model, num_samples, model_name,
               ignored_IDs=[], folder_name='allliquids', confidence=95):

    q = [(100-confidence)/2, 100 - (100-confidence)/2]

    IDs = viscosity_df.ID.unique().tolist()

    for iid in ignored_IDs:
        IDs.pop(IDs.index(iid))

    base_size = 3
    num_lines = 5
    num_cols = 4
    filename_idx = 1

    for n, ID in enumerate(IDs):

        if n % (num_lines * num_cols) == 0:
            if len(IDs) - n < num_lines * num_cols:
                num_lines = \
                    math.ceil((len(IDs) % (num_lines * num_cols)) / num_cols)

            fig, axes = plt.subplots(
                num_lines,
                num_cols,
                figsize=(base_size*num_cols, base_size*num_lines),
            )
            filename = f'all_liquids_{filename_idx}'
            filename_idx += 1
            idx = 0

        df = viscosity_df[viscosity_df.ID == ID]
        x = df['T'].values
        y = df['log_visc'].values
        composition_dict = df.reindex(compounds, axis=1).iloc[0].to_dict()
        composition_dict = {k:v for k,v in composition_dict.items() if v > 0}

        title = '.'.join([f'{v*100:.0f}{c.translate(subscript)}'
                 for c,v in composition_dict.items()])

        T = torch.linspace(min(x), max(x), 50, requires_grad=False)
        bands = \
            model.viscosity_bands_single(composition_dict, T, q, num_samples)

        axe = axes.flatten()[idx]
        letter = string.ascii_lowercase[idx]
        idx += 1

        axe.plot(x, y, marker='o', ls='none', markeredgecolor='black',)
        for band in bands:
            axe.plot(T, band, ls='--', c='tab:red')

        axe.set_title(title, fontsize=8)

        textbox = f'({letter})'
        axe.text(
            0.95,
            0.95,
            textbox,
            transform=axe.transAxes,
            fontsize=10,
            va='top',
            ha='right',
        )

        if (n + 1) % (num_lines * num_cols) == 0 or n == len(IDs) - 1:

            if len(axes.shape) == 2:
                for i, row in enumerate(axes):
                    for j, cell in enumerate(row):
                        if i == len(axes) - 1:
                            cell.set_xlabel('$T$  [K]')
                        if j == 0:
                            cell.set_ylabel(
                                r'$\log_{10} (\eta)$  [$\eta$ in Pa.s]'
                            )
            else:
                for i, cell in enumerate(axes):
                    cell.set_xlabel('$T$  [K]')
                    if i == 0:
                        cell.set_ylabel(r'$\log_{10} (\eta)$  [$\eta$ in Pa.s]')

            fig.savefig(
                rf'plots/{model_name}/{folder_name}/{filename}'
                rf'.{plot_file_ext}',
                dpi=150,
                bbox_inches='tight',
                pad_inches=2e-2,
            )

            plt.close(fig)


###############################################################################
#                                   Plotting                                  #
###############################################################################

if __name__ == '__main__':
    for model_name, model_path in model_dict.items():

        model = torch.load(model_path)
        model.eval()

        comp = composition_df.loc[test_idx]
        T = viscosity_df.loc[test_idx]['T'].values
        y_true = viscosity_df.loc[test_idx]['log_visc'].values

        with torch.no_grad():
            y_pred = model.viscosity_from_composition(comp, T)

        residual = y_true - y_pred

        boxplot(residual, model_name)
        compound(residual, composition_df.loc[test_idx], model_name)
        distance(compounds, viscosity_df, test_idx, train_val_idx, model,
                 model_name)
        correlation(y_true, y_pred, model_name)

        individual(compounds, viscosity_df.loc[test_idx], model, 1000,
                   model_name, folder_name='test_dataset', confidence=95)

        individual(compounds, viscosity_df.loc[train_idx], model, 1000,
                   model_name, folder_name='train_dataset', confidence=95)

        individual(compounds, viscosity_df.loc[val_idx], model, 1000,
                   model_name, folder_name='val_dataset', confidence=95)

        allliquids(compounds, viscosity_df.loc[test_idx], model, 1000,
                   model_name, [501], folder_name='test_dataset', confidence=95)

        oob_residual = outofband(compounds, viscosity_df.loc[test_idx], model,
                                 1000, model_name, folder_name='test_dataset',
                                 confidence=95)
        print(len(oob_residual[oob_residual == 0]))

        ternaryplot('Al2O3', 'SiO2', 'Na2O', compounds,
                    viscosity_df.loc[train_val_idx], model, model_name,
                    max_distance=0.5)
