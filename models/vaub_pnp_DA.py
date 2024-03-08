from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from itertools import chain, cycle
from backbone.util import get_backbone, get_pz
from dataset.util import get_train_dataset, get_test_dataset
import torchmetrics


class VAUB_pnp_DA(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        model = get_backbone(cfg.training.backbone, cfg.num_domain)

        self.encoder = model.encoder
        self.classifier = model.classifier
        self.decoder_arr = model.decoder_arr

        self.init_pz(cfg.training.pz, cfg.training.latent_dim, cfg.training.n_components)
        self.init_dataset(cfg.dataset.src, cfg.dataset.tgt, cfg.dataset.img_size,
                          cfg.dataset.root)

        self.log_scale_qz_arr = nn.ParameterList([nn.Parameter(torch.zeros(cfg.training.latent_dim))
                                                  for _ in range(cfg.num_domain)])
        self.log_scale_px_arr = nn.ParameterList([nn.Parameter(torch.Tensor([0.0]))
                                                  for _ in range(cfg.num_domain)])

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)

        self.inject_noise = cfg.training.inject_noise
        self.p_base_noise = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def get_noise(self, shape):
        if self.cfg.training.strategy == 'fixed':
            return self.p_base_noise.sample(shape).squeeze(-1)*self.cfg.training.noise_scale
        if self.cfg.training.strategy == 'linear':
            return self.p_base_noise.sample(shape).squeeze(-1)*self.cfg.training.noise_scale*(1-self.get_p())

    def init_dataset(self, src, tgt, img_size, root):
        self.train_set_src, self.val_set_src = get_train_dataset(src, img_size, root)
        self.train_set_tgt, self.val_set_tgt = get_train_dataset(tgt, img_size, root)
        self.test_set_tgt = get_test_dataset(tgt, img_size, root)

    def init_pz(self, pz, latent_dim, n_components):
        self.pz = get_pz(pz, latent_dim, n_components=n_components)


    def train_dataloader(self):
        src_loader = DataLoader(self.train_set_src, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.cfg.training.num_workers, pin_memory=True,
                                sampler=None, drop_last=True)
        tgt_loader = DataLoader(self.train_set_tgt, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.cfg.training.num_workers, pin_memory=True,
                                sampler=None, drop_last=True)
        self.len_dataloader = min(len(src_loader), len(tgt_loader))
        return zip(range(self.len_dataloader), cycle(src_loader), cycle(tgt_loader))


    def val_dataloader(self):
        return DataLoader(self.val_set_tgt, batch_size=self.batch_size,
                          num_workers=self.cfg.training.num_workers)


    def test_dataloader(self):
        return DataLoader(self.test_set_tgt, batch_size=self.batch_size,
                          num_workers=self.cfg.training.num_workers)


    def training_step(self, batch, batch_idx):
        _, (x_src, y_src), (x_tgt, _) = batch

        p = self.get_p()
        self.lr_schedule_step(p)

        # forward the encoder and decoder
        encoded_src = self.encoder(x_src)
        encoded_tgt = self.encoder(x_tgt)
        qz_arr = self.get_qzx([encoded_src, encoded_tgt])
        self.z_arr = [qz.rsample() for qz in qz_arr]
        self.x_hat_arr = [decoder(z) for decoder,z in zip(self.decoder_arr, self.z_arr)]

        # get log_pz
        if self.inject_noise and (self.current_epoch > self.cfg.training.lr_qz_freeze):
            z_noisy_arr = [z + self.get_noise(z.size()).to(z.device) for z in self.z_arr]
            self.log_pz_arr = [self.pz.log_prob(z) for z in z_noisy_arr]
        else:
            self.log_pz_arr = [self.pz.log_prob(z) for z in self.z_arr]

        # get log_qzx
        self.log_qzx_arr = [qz.log_prob(z) for qz,z in zip(qz_arr, self.z_arr)]

        # get log_pxz
        self.log_pxz_arr = self.get_pxz([x_src, x_tgt])

        # calculate vaub loss
        loss_vaub = self.get_vaub_loss()

        # get cls loss
        outputs_src = self.classifier(encoded_src)
        loss_cls = F.cross_entropy(outputs_src, y_src)

        # get final loss
        if self.inject_noise:
            loss = loss_cls + self.cfg.training.lambda_vaub*loss_vaub + \
                   self.cfg.training.lambda_norm*sum([z.std() for z in self.z_arr])
        else:
            loss = loss_cls + self.cfg.training.lambda_vaub*loss_vaub

        # Record logs
        train_acc = self.train_accuracy(outputs_src, y_src)
        self.log("acc/train_acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("loss_detail/loss_cls", loss_cls, on_step=True, on_epoch=False, sync_dist=True)
        self.log("loss_detail/loss_vaub", loss_vaub, on_step=True, on_epoch=False, sync_dist=True)

        kl = sum([-1*(pz-qzx).mean() for (pz, qzx) in zip(self.log_pz_arr, self.log_qzx_arr)])
        recon = sum([-1*pxz.mean() for pxz in self.log_pxz_arr])
        jacobian = sum([(pxz-qzx).mean() for (pxz, qzx) in zip(self.log_pxz_arr, self.log_qzx_arr)])

        self.log("loss_detail/kl", kl, on_step=True, on_epoch=False, sync_dist=True)
        self.log("loss_detail/recon", recon, on_step=True, on_epoch=False, sync_dist=True)
        self.log("loss_detail/jacobian", jacobian, on_step=True, on_epoch=False, sync_dist=True)

        self.log("log_prob/log_qzx_src", self.log_qzx_arr[0].mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("log_prob/log_qzx_tgt", self.log_qzx_arr[1].mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("log_prob/log_pz_src", self.log_pz_arr[0].mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("log_prob/log_pz_tgt", self.log_pz_arr[1].mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("log_prob/log_pxz_src", self.log_pxz_arr[0].mean(), on_step=True, on_epoch=False, sync_dist=True)
        self.log("log_prob/log_pxz_tgt", self.log_pxz_arr[1].mean(), on_step=True, on_epoch=False, sync_dist=True)

        # z_mix = torch.cat(self.z_arr)
        self.log("latent_domain/mean_src", self.z_arr[0].mean(dim=0).mean(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log("latent_domain/mean_tgt", self.z_arr[1].mean(dim=0).mean(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log("latent_domain/mean_diff", (self.z_arr[0]-self.z_arr[1]).mean(dim=0).mean(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log("latent_domain/std_src", self.z_arr[0].std(dim=0).mean(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log("latent_domain/std_tgt", self.z_arr[1].std(dim=0).mean(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log("latent_domain/norm_src", torch.norm(self.z_arr[0]),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log("latent_domain/norm_tgt", torch.norm(self.z_arr[1]),
                 on_step=True, on_epoch=False, sync_dist=True)

        self.log("lr/p", p, on_step=True, on_epoch=False, sync_dist=True)
        for idx, param_group in enumerate(self.optimizers().param_groups):
            self.log(f"lr/lr_{idx}", param_group["lr"], on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def lr_schedule_step(self, p):
        for param_group in self.optimizers().param_groups:
            param_group["lr"] = param_group["lr_mult"] * self.cfg.training.lr * p
            if (self.current_epoch<self.cfg.training.lr_qz_freeze) and (param_group["label"]=="pz"):
                param_group["lr"] = 0.0

    def training_epoch_end(self, outs):
        self.log("acc_epoch/train_acc_epoch", self.train_accuracy.compute(),
                 prog_bar=True, logger=True, sync_dist=True)

        # print(f"Finished Epoch {self.current_epoch}")

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.encoder(inputs)
        outputs = self.classifier(features)

        loss = F.cross_entropy(outputs, targets)
        val_acc = self.val_accuracy(outputs, targets)

        self.log("acc/val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        tensorboard =self.logger.experiment
        tensorboard.add_images("input", inputs[:self.cfg.training.num_visual],
                               self.global_step)
        tensorboard.add_images("recon",
                               self.unnormalize_images(self.decoder_arr[1](features))[:self.cfg.training.num_visual],
                               self.global_step)
        tensorboard.add_images("flipped",
                               self.unnormalize_images(self.decoder_arr[0](features))[:self.cfg.training.num_visual],
                               self.global_step)

        return loss

    def unnormalize_images(self, x):
        return (x-x.min())/(x.max()-x.min())

    def validation_epoch_end(self, outs):
        self.log("acc_epoch/val_acc_epoch", self.val_accuracy.compute(),
                 prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        features = self.encoder(inputs)
        outputs = self.classifier(features)

        loss = F.cross_entropy(outputs, targets)
        test_acc = self.test_accuracy(outputs, targets)

        self.log("acc/test_acc", test_acc, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)
        self.log("loss/test_loss", loss, on_step=False, on_epoch=True, logger=True,
                 sync_dist=True)

        return loss

    def test_epoch_end(self, outs):
        test_acc = self.test_accuracy.compute()
        self.log("test_acc_epoch", test_acc, logger=True, sync_dist=True)

    def configure_optimizers(self):
        model_parameter = [
            {
                "params": self.encoder.parameters(),
                "lr_mult": 1.0,
                "label": "encoder",
            },
            {
                "params": chain(*[decoder.parameters() for decoder in self.decoder_arr]),
                "lr_mult": 1.0,
                "label": "decoders",
            },
            {
                "params": self.pz.parameters(),
                "lr_mult": 1.0,
                "label": "pz",
            },
            {
                "params": self.classifier.parameters(),
                "lr_mult": 1.0,
                "label": "cls",
            },
        ]
        optimizer = torch.optim.Adam(
            model_parameter,
            lr=self.cfg.training.lr,
            betas=(0.9, 0.999),
            weight_decay=self.cfg.training.weight_decay,
        )

        return optimizer

    def get_p(self):
        current_iterations = self.global_step
        current_epoch = self.current_epoch
        len_dataloader = self.len_dataloader

        p = 1 - float(max(0, current_iterations-self.cfg.training.lr_warmup_epochs*len_dataloader))\
            / (self.cfg.training.lr_decay_epochs*len_dataloader)

        return p

    def get_pxz(self, x_arr):

        p_dist = [torch.distributions.Normal(mean, std.exp())
                  for mean, std in zip(self.x_hat_arr, self.log_scale_px_arr)]
        log_pxz_arr = [p.log_prob(x) for p,x in zip(p_dist, x_arr)]
        return [log_pxz.sum(dim=(1, 2, 3)) for log_pxz in log_pxz_arr]

    def get_qzx(self, mu_arr):

        std_arr = [torch.exp(log_scale_qz).float() for log_scale_qz in self.log_scale_qz_arr]
        qzx_arr = [torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(std))
                  for mu, std in zip(mu_arr,std_arr)]

        return qzx_arr

    def get_vaub_loss(self):

        return -1*sum([(pz-qzx+pxz).mean() for pz,qzx,pxz in zip(self.log_pz_arr, self.log_qzx_arr, self.log_pxz_arr)])


