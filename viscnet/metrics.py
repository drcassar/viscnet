from functools import partial
import pandas as pd

import torch

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import median_absolute_error as MedAE
from sklearn.metrics import mean_absolute_error as MAE

from data import viscosity_df, composition_df, train_idx, val_idx, test_idx
from models import ViscNet, ViscNetVFT


def R2(y_true, y_pred):
    return 1 - sum((y_true - y_pred)**2) / sum(y_true**2)


###############################################################################
#                                    Config                                   #
###############################################################################

datasets = {
    'train': train_idx,
    'val': val_idx,
    'test': test_idx,
}

metric_dict = {
    'R2': R2,
    'RMSE': partial(MSE, squared=False),
    'MAE': MAE,
    'MedAE': MedAE,
}

model_dict = {
    'ViscNet': f'files/viscnet.pt',
    'ViscNet-Huber': f'files/viscnet_huber.pt',
    'ViscNet-VFT': f'files/viscnet_vft.pt',
}


###############################################################################
#                                 Calculations                                #
###############################################################################

if __name__ == '__main__':
    for model_name, model_path in model_dict.items():

        metric_df = pd.DataFrame([], columns=datasets, index=metric_dict)

        model = torch.load(model_path)
        model.eval()

        with torch.no_grad():

            for dataset_name, idx in datasets.items():
                comp = composition_df.loc[idx]
                T = viscosity_df.loc[idx]['T'].values
                y_true = viscosity_df.loc[idx]['log_visc'].values
                y_pred = model.viscosity_from_composition(comp, T)

                for metric_name, metric_fun in metric_dict.items():
                    metric_df[dataset_name][metric_name] = metric_fun(y_true, y_pred)

        print(model_name)
        print(metric_df)
