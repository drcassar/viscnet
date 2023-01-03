from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd
from chemparse import parse_formula

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


chem_prop_path = r'files/chemical_properties.csv.xz'

hparams = {
    'batch_size': 64,
    'layer_1_activation': 'ReLU',
    'layer_1_batchnorm': False,
    'layer_1_dropout': 0.07942161101271952,
    'layer_1_size': 192,
    'layer_2_activation': 'Tanh',
    'layer_2_batchnorm': False,
    'layer_2_dropout': 0.05371454289414608,
    'layer_2_size': 48,
    'loss': 'mse',
    'lr': 0.0011695226458761677,
    'max_epochs': 500,
    'n_features': 35,
    'num_layers': 2,
    'optimizer': 'AdamW',
    'patience': 9,
}

class BaseViscosityModel(pl.LightningModule, ABC):

    learning_curve_train = []
    learning_curve_val = []

    def __init__(self, hparams, x_mean=0, x_std=1):
        super().__init__()

        layers = []
        input_dim = hparams['n_features']

        for n in range(1, hparams['num_layers'] + 1):

            l = [
                nn.Linear(
                    input_dim, int(hparams[f'layer_{n}_size']),
                    bias=False if hparams[f'layer_{n}_batchnorm'] else True)
            ]

            if hparams[f'layer_{n}_batchnorm']:
                l.append(nn.BatchNorm1d(int(hparams[f'layer_{n}_size'])))

            if hparams[f'layer_{n}_dropout']:
                l.append(nn.Dropout(hparams[f'layer_{n}_dropout']))

            if hparams[f'layer_{n}_activation'] == 'Tanh':
                l.append(nn.Tanh())
            elif hparams[f'layer_{n}_activation'] == 'ReLU':
                l.append(nn.ReLU())
            else:
                raise NotImplementedError(
                    'Please add this activation to the model class.'
                )

            layers.append(nn.Sequential(*l))
            input_dim = int(hparams[f'layer_{n}_size'])

        self.hidden_layers = nn.Sequential(*layers)

        if hparams['loss'] == 'mse':
            self.loss_fun = F.mse_loss
        elif hparams['loss'] == 'huber':
            self.loss_fun = F.smooth_l1_loss
        else:
            raise NotImplementedError(
                'Please add this loss function to the model class.'
            )

        self.chemical_properties = pd.read_csv(chem_prop_path, index_col=0).T

        self.x_mean = x_mean
        self.x_std = x_std
        self.hparams.update(hparams)

    def gen_atomic_df(self, composition_df):
        compound_lst = composition_df.columns.tolist()
        all_elements = self.chemical_properties.columns.tolist()

        element_guide = np.zeros((len(all_elements), len(compound_lst)))
        for j in range(len(compound_lst)):
            c = compound_lst[j]
            cdic = parse_formula(c)
            for el in cdic:
                i = all_elements.index(el)
                element_guide[i,j] += cdic[el]

        atomic_df = np.zeros((len(composition_df), len(all_elements)))
        for i in range(len(compound_lst)):
            c = compound_lst[i]
            cdic = parse_formula(c)
            for el in cdic:
                j = all_elements.index(el)
                atomic_df[:,j] += composition_df[c].values*element_guide[j,i]

        atomic_df = pd.DataFrame(
            atomic_df,
            columns=all_elements,
            index=composition_df.index,
        )
        atomic_df = atomic_df.div(atomic_df.sum(axis=1), axis=0)

        return atomic_df

    @abstractmethod
    def composition_to_features(self, composition_df):
        pass

    @property
    @abstractmethod
    def parameters_range(self):
        pass

    @property
    @abstractmethod
    def absolute_features(self):
        pass

    @property
    @abstractmethod
    def weighted_features(self):
        pass

    @abstractmethod
    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def viscosity_from_composition(self, composition_df, T):

        feats = self.composition_to_features(composition_df)
        x = torch.from_numpy(feats.assign(T=T).values).float()
        log_viscosity = self(x)

        return log_viscosity.detach().numpy()

    def update_x_norm(self, feature_array):

        feature_tensor = torch.from_numpy(feature_array).float()

        x_mean = feature_tensor.mean(0, keepdim=True)
        x_std = feature_tensor.std(0, unbiased=False, keepdim=True)

        self.x_mean = x_mean
        self.x_std = x_std
        
    def configure_optimizers(self):

        if 'optimizer' not in self.hparams:
            optimizer = SGD(self.parameters())

        elif self.hparams['optimizer'] == 'SGD':
            optimizer = SGD( 
                self.parameters(),
                lr=self.hparams['lr'],
                momentum=self.hparams['momentum'],
            )

        elif self.hparams['optimizer'] == 'Adam':
            optimizer = Adam( 
                self.parameters(),
                lr=self.hparams['lr'],
            )

        elif self.hparams['optimizer'] == 'AdamW':
            optimizer = AdamW( 
                self.parameters(),
                lr=self.hparams['lr'],
            )

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        return {'loss': loss,}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.learning_curve_train.append(float(avg_loss))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fun(self(x), y)
        return {'val_loss_step': loss,}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss_step"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.learning_curve_val.append(float(avg_loss))


class ViscNet(BaseViscosityModel):

    parameters_range = {
        'log_eta_inf': [-18, 5],
        'Tg': [400, 1400],
        'm': [10, 130],
    }

    absolute_features = [
        ('ElectronAffinity', 'std'),
        ('FusionEnthalpy', 'std'),
        ('GSenergy_pa', 'std'),
        ('GSmagmom', 'std'),
        ('NdUnfilled', 'std'),
        ('NfValence', 'std'),
        ('NpUnfilled', 'std'),
        ('atomic_radius_rahm', 'std'),
        ('c6_gb', 'std'),
        ('lattice_constant', 'std'),
        ('mendeleev_number', 'std'),
        ('num_oxistates', 'std'),
        ('nvalence', 'std'),
        ('vdw_radius_alvarez', 'std'),
        ('vdw_radius_uff', 'std'),
        ('zeff', 'std'),
    ]

    weighted_features = [
        ('FusionEnthalpy', 'min'),
        ('GSbandgap', 'max'),
        ('GSmagmom', 'mean'),
        ('GSvolume_pa', 'max'),
        ('MiracleRadius', 'std'),
        ('NValence', 'max'),
        ('NValence', 'min'),
        ('NdUnfilled', 'max'),
        ('NdValence', 'max'),
        ('NsUnfilled', 'max'),
        ('SpaceGroupNumber', 'max'),
        ('SpaceGroupNumber', 'min'),
        ('atomic_radius', 'max'),
        ('atomic_volume', 'max'),
        ('c6_gb', 'max'),
        ('c6_gb', 'min'),
        ('max_ionenergy', 'min'),
        ('num_oxistates', 'max'),
        ('nvalence', 'min'),
    ]

    def __init__(self, hparams, x_mean=0, x_std=1):
        super().__init__(hparams, x_mean, x_std)

        input_dim = int(hparams[f'layer_{hparams["num_layers"]}_size'])

        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, len(self.parameters_range)),
            nn.Sigmoid(),
        )

    def composition_to_features(self, composition_df):

        atomic_df = self.gen_atomic_df(composition_df)

        # weighted data
        wei_data = atomic_df.copy()
        logic = wei_data > 0
        wei_data[~logic] = pd.NA

        # absolute data
        abs_data = atomic_df.copy().astype(bool)
        abs_data[abs_data == False] = pd.NA

        features = pd.DataFrame([], index=atomic_df.index)

        for attr, stat in self.absolute_features:
            attr_vec = self.chemical_properties.loc[attr]
            ab = attr_vec * abs_data
            features[f'abs|{attr}|{stat}'] = \
                getattr(ab, stat)(axis=1, skipna=True)

        for attr, stat in self.weighted_features:
            attr_vec = self.chemical_properties.loc[attr]
            wei = attr_vec * wei_data
            features[f'wei|{attr}|{stat}'] = \
                getattr(wei, stat)(axis=1, skipna=True)

        return features

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):

        log_viscosity = log_eta_inf + (12 - log_eta_inf)*(Tg / T) * \
            ((m / (12 - log_eta_inf) - 1) * (Tg / T - 1)).exp()

        return log_viscosity

    def forward(self, x):

        T = x[:, -1].detach().clone()

        # Neural net
        xf = self.hidden_layers((x[:, :-1] - self.x_mean) / self.x_std)
        xf = self.output_layer(xf)

        parameters = {}

        for i, (p_name, p_range) in enumerate(self.parameters_range.items()):

            # Scaling the viscosity parameters to be within the parameter range
            parameters[p_name] = torch.add(
                torch.ones(xf.shape[0]).mul(p_range[0]),
                xf[:,i],
                alpha=p_range[1] - p_range[0],
            )

        log_viscosity = self.log_viscosity_fun(T, **parameters)

        return log_viscosity

    def viscosity_params_from_features(self, feature_tensor,
                                       return_tensor=False):

        xf = self.hidden_layers((feature_tensor - self.x_mean) / self.x_std)
        xf = self.output_layer(xf)

        parameters = {}

        if xf.shape[0] == 1:
            for i, (p_name, p_range) in enumerate(self.parameters_range.items()):
                parameters[p_name] = float(torch.add(
                    torch.ones(xf.shape[0]).mul(p_range[0]),
                    xf[:,i],
                    alpha=p_range[1] - p_range[0],
                ).detach().numpy())

        elif return_tensor:
            for i, (p_name, p_range) in enumerate(self.parameters_range.items()):
                parameters[p_name] = torch.add(
                    torch.ones(xf.shape[0]).mul(p_range[0]),
                    xf[:,i],
                    alpha=p_range[1] - p_range[0],
                )

        else:
            for i, (p_name, p_range) in enumerate(self.parameters_range.items()):
                parameters[p_name] = torch.add(
                    torch.ones(xf.shape[0]).mul(p_range[0]),
                    xf[:,i],
                    alpha=p_range[1] - p_range[0],
                ).detach().numpy()

        return parameters

    def viscosity_params_from_composition(self, composition_df):

        feats = self.composition_to_features(composition_df)
        feats = torch.from_numpy(feats.values).float()

        return self.viscosity_params_from_features(feats)

    def viscosity_bands_batch(self, composition_df, T, q, num_samples=100):
        
        feats = self.composition_to_features(composition_df)
        feats = torch.from_numpy(feats.values).float()

        is_training = self.training

        if not is_training:
            self.train()

        all_curves = []

        with torch.no_grad():
            for _ in range(num_samples):
                parameters = \
                    self.viscosity_params_from_features(feats,
                                                        return_tensor=True)

                all_curves.append(
                    self.log_viscosity_fun(T, **parameters).numpy()
                )

        if not is_training:
            self.eval()

        bands = np.percentile(all_curves, q, axis=0)

        return bands

    def viscosity_bands_single(self, composition_dict, T, q, num_samples=100):
        
        composition_df = \
            pd.DataFrame.from_dict(composition_dict, orient='index').T
        feats = self.composition_to_features(composition_df)
        feats = torch.from_numpy(feats.values).float()

        is_training = self.training

        if not is_training:
            self.train()

        all_curves = []

        with torch.no_grad():
            for _ in range(num_samples):
                parameters = self.viscosity_params_from_features(feats)

                all_curves.append(
                    self.log_viscosity_fun(T, **parameters).numpy()
                )

        if not is_training:
            self.eval()

        bands = np.percentile(all_curves, q, axis=0)

        return bands


class ViscNetVFT(ViscNet):

    def __init__(self, hparams, x_mean=0, x_std=1):
        super().__init__(hparams, x_mean, x_std)

    def log_viscosity_fun(self, T, log_eta_inf, Tg, m):

        log_viscosity = log_eta_inf + (12 - log_eta_inf)**2 / \
            (m * (T / Tg - 1) + (12 - log_eta_inf))

        return log_viscosity


def train_model(model, train_ds, val_ds, num_workers=4, deterministic=True):

    train_dl = DataLoader(
        train_ds,
        batch_size=int(model.hparams['batch_size']),
        shuffle=True,
        num_workers=num_workers,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=256,
        num_workers=num_workers,
    )

    earlystop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=int(model.hparams['patience']),
        verbose=False,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=model.hparams['max_epochs'],
        callbacks=[earlystop],
        deterministic=deterministic,
    )

    trainer.fit(model, train_dl, val_dl)

    return trainer




