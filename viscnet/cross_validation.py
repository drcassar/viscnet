import pandas as pd
from sklearn.model_selection import KFold

import torch
import pytorch_lightning as pl

from data import IDs as IDs_train_val
from data import viscosity_df, train_val_ds, composition_df
from metrics import metric_dict
from models import ViscNet, train_model, hparams


###############################################################################
#                                    Config                                   #
###############################################################################

num_folds = 10


###############################################################################
#                                  Functions                                  #
###############################################################################

def cv_fold_idx_gen(IDs, viscosity_df, num_folds):

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for train_ID_idx, val_ID_idx in kf.split(IDs):

        train = IDs[train_ID_idx]
        val = IDs[val_ID_idx]

        train_idx = viscosity_df[viscosity_df.ID.isin(train)].index
        val_idx = viscosity_df[viscosity_df.ID.isin(val)].index

        yield train_idx, val_idx


###############################################################################
#                               Cross-validation                              #
###############################################################################

if __name__ == '__main__':

    generator = cv_fold_idx_gen(IDs_train_val, viscosity_df, num_folds)

    metric_df = pd.DataFrame([], columns=range(1, num_folds+1), index=metric_dict)

    for i, (train_idx, val_idx) in enumerate(generator, start=1):

        try:
            model = torch.load(rf'files/crossval/viscnet_cv_fold_{i}.pt')

        except FileNotFoundError:
            pl.seed_everything(i)                # for reproducibility

            model = ViscNet(hparams)
            feats = model.composition_to_features(composition_df)
            train_val_idx = train_idx.union(val_idx)

            model.update_x_norm(feats.loc[train_val_idx].values)

            train_ds, val_ds = train_val_ds(feats, viscosity_df, train_idx, val_idx)

            train_model(
                model,
                train_ds,
                val_ds,
                num_workers=4,
                deterministic=True,
            )

            torch.save(model, rf'files/crossval/viscnet_cv_fold_{i}.pt')

        model.eval()

        with torch.no_grad():

            comp = composition_df.loc[val_idx]
            T = viscosity_df.loc[val_idx]['T'].values
            y_true = viscosity_df.loc[val_idx]['log_visc'].values
            y_pred = model.viscosity_from_composition(comp, T)

            for metric_name, metric_fun in metric_dict.items():
                metric_df[i][metric_name] = metric_fun(y_true, y_pred)

    metric_df = metric_df.assign(
        mean=metric_df.mean(axis=1),
        std=metric_df.std(axis=1),
    )

    print()
    print(metric_df)
