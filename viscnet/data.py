import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset


###############################################################################
#                                    Config                                   #
###############################################################################

base_path = Path(r'files')
path_idx = base_path / 'train_test_indices.p'
path_data = base_path / 'data.csv.xz'

subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


###############################################################################
#                                  Functions                                  #
###############################################################################

def train_val_ds(feats, viscosity_df, train_idx, val_idx):

    x_train = torch.from_numpy(
        feats.assign(T=viscosity_df['T']).loc[train_idx].values
    ).float()
    y_train = torch.from_numpy(
        viscosity_df['log_visc'].loc[train_idx].values
    ).float()

    x_val = torch.from_numpy(
        feats.assign(T=viscosity_df['T']).loc[val_idx].values
    ).float()
    y_val = torch.from_numpy(
        viscosity_df['log_visc'].loc[val_idx].values
    ).float()

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    return train_ds, val_ds


###############################################################################
#                                     Data                                    #
###############################################################################

viscosity_df = pd.read_csv(path_data, index_col=0)
compounds = viscosity_df.columns[:-3]
composition_df = viscosity_df[compounds]


###############################################################################
#                                  Data split                                 #
###############################################################################

train_val_idx, test_idx = pickle.load(open(path_idx, "rb"))

IDs = viscosity_df.loc[train_val_idx]['ID'].unique()

train, val = train_test_split(
    IDs,
    test_size=len(IDs) // 10,
    shuffle=True,
    random_state=72,
)

train_idx = viscosity_df[viscosity_df.ID.isin(train)].index
val_idx = viscosity_df[viscosity_df.ID.isin(val)].index


