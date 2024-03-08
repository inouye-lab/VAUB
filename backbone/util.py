import os
import torch

from backbone.lenet import LeNet, LeNet_big
from backbone.svhn import SVHN
from backbone.minimalnet import MinimalNet32
from backbone.pz import FixedGaussian, MoGNN

def get_backbone(backbone, num_domain, load=None, save_path=None):
    if load is not None:
        assert save_path is not None

    if backbone == "minimal32":
        model = MinimalNet32(num_domain)

    if backbone == "lenet":
        model = LeNet(num_domain)

    if backbone == "lenet_big":
        model = LeNet_big(num_domain)

    if backbone == "svhn":
        model = SVHN(num_domain)

    if load:
        encoder_path = os.path.join(save_path, "encoder.pth")
        model.encoder.load_state_dict(torch.load(encoder_path))

        classifier_path = os.path.join(save_path, "classifier.pth")
        model.classifier.load_state_dict(torch.load(classifier_path))

    return model

def get_pz(pz, latent_dim, n_components=None, load=None, save_path=None):
    if load is not None:
        assert save_path is not None

    if pz == "fixed":
        model = FixedGaussian(latent_dim)

    if pz == "MoG":
        model = MoGNN(latent_dim, n_components)

    if load:
        pz_path = os.path.join(save_path, "pz.pth")
        model.pz.load_state_dict(torch.load(pz_path))

    return model