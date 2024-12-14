import datetime
import os.path as osp
import time
from tqdm import tqdm
import numpy as np
import sys
from math import sqrt 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.nn.functional import normalize
import cv2

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import VisionTransformer
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import load_clip_to_cpu

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}

noiseNumList = []  # the number of drawn noise images in each class
heatmapNumList = []  # the number of drawn heatmap in each class

# draw noise distribution for image
def noise_show(images, labels, noises, image_dir, class_names):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    reverse_normalize_image = transforms.Normalize([-m / s for m, s in zip(mean, std)], [1 / s for s in std])
    reverse_normalize_noise = transforms.Normalize([0, 0, 0], [1 / s for s in std])
    images = reverse_normalize_image(images)
    noises = reverse_normalize_noise(noises)
    labels = labels.unsqueeze(1)
    for (image, label, noise) in zip(images, labels, noises):
        class_idx = int(label)
        if noiseNumList[class_idx] > 100:
            continue
        noise_image = image + noise
        image_concat = torch.cat([image, noise, noise_image], dim=2)
        dir = image_dir + r'/' + class_names[class_idx]
        save_image(image_concat, dir + r'/' + str(noiseNumList[class_idx]) + r'_concat.jpg')
        save_image(image, dir + r'/' + str(noiseNumList[class_idx]) + r'_image.jpg')
        save_image(noise, dir + r'/' + str(noiseNumList[class_idx]) + r'_noise.jpg')
        save_image(noise_image, dir + r'/' + str(noiseNumList[class_idx]) + r'_noise-image.jpg')
        noiseNumList[class_idx] += 1


def heatmap_show(images, mus, variances, labels, heatmap_dir, class_names):
    """draw heatmap for variance"""
    labels = labels.unsqueeze(1)
    for (image, mu, variance, label) in zip(images, mus, variances, labels):
        class_idx = int(label)
        if noiseNumList[class_idx] > 100:
            continue
        variance = variance.cpu().detach().numpy()
        mu = mu.cpu().detach().numpy()
        # one channel
        if image.size(0) == 1:
            variance = np.uint8(variance[0] / max(variance[0].max(), sys.float_info.epsilon) * 256)
            heatmap = cv2.applyColorMap(variance, cv2.COLORMAP_JET)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'.jpg'
            cv2.imwrite(file_name, heatmap)
        # three channel
        else:
            varianceR = np.uint8(variance[0] / max(variance[0].max(), sys.float_info.epsilon) * 256)
            heatmapR = cv2.applyColorMap(varianceR, cv2.COLORMAP_JET)
            varianceG = np.uint8(variance[1] / max(variance[1].max(), sys.float_info.epsilon) * 256)
            heatmapG = cv2.applyColorMap(varianceG, cv2.COLORMAP_JET)
            varianceB = np.uint8(variance[2] / max(variance[2].max(), sys.float_info.epsilon) * 256)
            heatmapB = cv2.applyColorMap(varianceB, cv2.COLORMAP_JET)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'var_red.jpg'
            cv2.imwrite(file_name, heatmapR)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'var_green.jpg'
            cv2.imwrite(file_name, heatmapG)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'var_blue.jpg'
            cv2.imwrite(file_name, heatmapB)

            varianceR = np.uint8(mu[0] / max(mu[0].max(), sys.float_info.epsilon) * 256)
            heatmapR = cv2.applyColorMap(varianceR, cv2.COLORMAP_JET)
            varianceG = np.uint8(mu[1] / max(mu[1].max(), sys.float_info.epsilon) * 256)
            heatmapG = cv2.applyColorMap(varianceG, cv2.COLORMAP_JET)
            varianceB = np.uint8(mu[2] / max(mu[2].max(), sys.float_info.epsilon) * 256)
            heatmapB = cv2.applyColorMap(varianceB, cv2.COLORMAP_JET)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'mu_red.jpg'
            cv2.imwrite(file_name, heatmapR)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'mu_green.jpg'
            cv2.imwrite(file_name, heatmapG)
            file_name = heatmap_dir + r'/' + class_names[class_idx] + r'/' + str(heatmapNumList[class_idx]) + r'mu_blue.jpg'
            cv2.imwrite(file_name, heatmapB)
        heatmapNumList[class_idx] += 1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class VPNGenerator(torch.nn.Module):
    def __init__(self, clip_dim, n_channel, image_size, patch_size, patch_dim):
        super(VPNGenerator, self).__init__()
        self.clip_dim = clip_dim
        self.n_channel = n_channel
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.patch_dim = patch_dim
        self.block = BasicBlock
        self.num_blocks = [2, 1, 1]
        self._build_up()
        self.level = 0.1

    def _build_up(self):
        self.conv1 = nn.Conv2d(self.n_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #for layers
        self.in_planes = 64
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=1)
        self.layer3 = self._make_layer(self.block, 64, self.num_blocks[2], stride=1)
        self.conv_variance = nn.Conv2d(64, self.n_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv_mean = nn.Conv2d(64, self.n_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, images):
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        variance = self.conv_variance(out)
        variance = variance.reshape(variance.size(0), variance.size(1), -1)
        variance = normalize(variance, dim=2) * sqrt(variance.size(2)) * self.level
        variance = variance.reshape(images.shape)
        variance = variance.abs()
        mu = self.conv_mean(out)
        mu = mu.reshape(mu.size(0), mu.size(1), -1)
        mu = normalize(mu, dim=2) * sqrt(mu.size(2)) * self.level
        mu = mu.reshape(images.shape)
        return mu, variance

    def sample(self, mu, variance, num=1):
        # noise = noise.reshape(batch_size, num, dim)
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise


class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.clip = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.input_resolution = clip_model.visual.input_resolution
        self.embed_dim = clip_model.visual.output_dim
        if isinstance(clip_model.visual, VisionTransformer):
            self.patch_size = clip_model.visual.conv1.weight.shape[-1]
        else:  # Mo
            self.patch_size = 32
        self.vpnGenerator = VPNGenerator(
            self.embed_dim, 3, self.input_resolution, self.patch_size, self.clip.visual.conv1.out_channels
        ).to(clip_model.dtype)

    def forward(self, images, labels):
        # n*D
        text_feat = self.text_encoder()
        # stop gradient propagation
        text_feat = text_feat.detach()
        # N*D
        text_feat_batch = text_feat[labels]
        # generate noise
        mu, variance = self.vpnGenerator(images.type(self.dtype))
        noises = self.vpnGenerator.sample(mu, variance)
        noises = noises.squeeze(1)
        noise_images = images + noises
        #noise_images = images
        visual_feat = self.image_encoder(noise_images.type(self.dtype))
        visual_feat = visual_feat / visual_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_feat @ text_feat.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_VPN_cnn_image(TrainerX):
    """ CLIP-VPN """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CLIP_VPN.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIP_VPN.PREC == "fp32" or cfg.TRAINER.CLIP_VPN.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        self.scaler = GradScaler(
        ) if cfg.TRAINER.CLIP_VPN.PREC == "amp" else None

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'vpnGenerator' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.vpnGenerator, cfg.MODEL.INIT_WEIGHTS
            )

        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.vpnGenerator, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model(
            'vpnGenerator', self.model.vpnGenerator, self.optim, self.sched
        )

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f'Multiple GPUs detected (n_gpus={device_count}), use all of them!'
            )
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.CLIP_VPN.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
                self.test()
        self.plot()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def plot(self, split=None):
        self.set_model_mode("eval")
        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        print(f"Plot on the *{split}* set")
        plot_dir = osp.join(self.output_dir, "images")
        mkdir_if_missing(plot_dir)
        for i in range(self.num_classes):
            mkdir_if_missing(plot_dir + "/" + self.dm.dataset.classnames[i])

        global noiseNumList
        global heatmapNumList
        noiseNumList = [
            1 for i in range(self.num_classes)
        ]  # the number of drawn noise images in each class
        heatmapNumList = [
            1 for i in range(self.num_classes)
        ]  # the number of drawn heatmap in each class

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images, labels = self.parse_batch_test(batch)
            # generate noise
            mu, variance = self.model.vpnGenerator(images.type(self.model.dtype))
            noises = self.model.vpnGenerator.sample(mu, variance)
            noises = noises.squeeze(1)
            noise_show(
                images, labels, noises, plot_dir, self.dm.dataset.classnames
            )
            heatmap_show(
                images, mu.abs(), variance, labels, plot_dir, self.dm.dataset.classnames
            )
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model(input, label)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
