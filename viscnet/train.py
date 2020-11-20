from copy import deepcopy as copy
import numpy as np

import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl

from data import (viscosity_df, composition_df, train_idx, val_idx,
                  train_val_idx, train_val_ds)
from models import ViscNet, ViscNetVFT, train_model, hparams
from plots import learning_curve


###############################################################################
#                       Creating and training the model                       #
###############################################################################


pl.seed_everything(3258)                # for reproducibility

model = ViscNet(hparams)
feats = model.composition_to_features(composition_df)
model.update_x_norm(feats.loc[train_val_idx].values)

train_ds, val_ds = train_val_ds(feats, viscosity_df, train_idx, val_idx)

train_model(
    model,
    train_ds,
    val_ds,
    num_workers=4,
    deterministic=True,
)

torch.save(model, r'files/viscnet.pt')

state_dict = model.state_dict()

# Learning curve
lc_train = model.learning_curve_train
lc_val = model.learning_curve_val[:-1]
lc_x = np.arange(1, len(lc_train)+1)
learning_curve(lc_x, lc_train, lc_val, [1, len(lc_train)], 'ViscNet')


###############################################################################
#                 Training ViscNet-Huber via transfer learning                #
###############################################################################

hparams_huber = copy(hparams)
hparams_huber['loss'] = 'huber'

pl.seed_everything(3258)                # for reproducibility

model = ViscNet(hparams_huber)
model.load_state_dict(state_dict)
model.update_x_norm(feats.loc[train_val_idx].values)

train_model(
    model,
    train_ds,
    val_ds,
    num_workers=4,
    deterministic=True,
)

torch.save(model, r'files/viscnet_huber.pt')


###############################################################################
#                  Training ViscNet-VFT via transfer learning                 #
###############################################################################

pl.seed_everything(3258)                # for reproducibility

model = ViscNetVFT(hparams)
model.load_state_dict(state_dict)
model.update_x_norm(feats.loc[train_val_idx].values)

train_model(
    model,
    train_ds,
    val_ds,
    num_workers=4,
    deterministic=True,
)

torch.save(model, r'files/viscnet_vft.pt')
