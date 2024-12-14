import datetime
import os.path as osp
import time
from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import load_clip_to_cpu, heatmap_show

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a type of pet, a photo of a',
    'OxfordFlowers': 'a type of flower, a photo of a',
    'FGVCAircraft': 'a type of aircraft, a photo of a',
    'DescribableTextures': 'texture',
    'EuroSAT': 'a centered satellite photo of',
    'StanfordCars': 'a photo of a',
    'Food101': 'a type of food, a photo of',
    'SUN397': 'a photo of a',
    'Caltech101': 'a photo of a',
    'UCF101': 'a photo of a person doing',
    'ImageNet': 'a photo of a',
    'ImageNetSketch': 'a photo of a',
    'ImageNetV2': 'a photo of a',
    'ImageNetA': 'a photo of a',
    'ImageNetR': 'a photo of a',
    "FER": "a photo of a face looking",
    "CLEVRCounts": "the number of objects in the photo is",
    "Resisc45": "satellite imagery of",
    "SST2": "the review of movie is",
    "GTSRB": "a zoomed in photo of a traffic sign",
    "HatefulMemes": "a",
    "PatchCameLyon": "this is a photo of",
    "Country211": "a photo i took in",
    "CLEVR": "the number of objects in photo is",
}


class CrossAttention(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_q = nn.Linear(input_dim, output_dim)
        self.fc_k = nn.Linear(input_dim, output_dim)
        self.fc_v = nn.Linear(input_dim, output_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = self.softmax(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class ImageVpnGenerator(torch.nn.Module):
    def __init__(self, clip_dim, reduction=1):
        super(ImageVpnGenerator, self).__init__()
        self.clip_dim = clip_dim
        self.reduction = reduction
        self.hidden_dim = self.clip_dim // self.reduction
        self._build_up()

    def _build_up(self):
        self.crossAttention = CrossAttention(self.clip_dim, self.hidden_dim)
        self.fc_variance = torch.nn.Linear(self.hidden_dim, self.clip_dim)
        self.fc_mean = torch.nn.Linear(self.hidden_dim, self.clip_dim)

    def forward(self, spatial_feat, text_feat):
        # batch (grid*grid) dim
        attn_feat, attn_weights = self.crossAttention(
            spatial_feat, text_feat, text_feat
        )
        # batch * patch_num * feat_dim
        variance = self.fc_variance(attn_feat).abs()
        mu = self.fc_mean(attn_feat)
        return mu, variance

    def sample(self, mu, variance, num=1):
        var = variance.expand(num, *variance.size()).transpose(0, 1)
        m = mu.expand(num, *mu.size()).transpose(0, 1)
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise


class PromptVpnGeneraotor(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        template = "x x x x x x x x x x x x x x x x " + template
        dtype = clip_model.dtype
        embedding_dim = clip_model.ln_final.weight.shape[0]
        
        # use given words to initialize context vectors
        template = template.replace("_", " ")
        template_lens = len(_tokenizer.encode(template))
        template_token = clip.tokenize(template)
        with torch.no_grad():
            template_embedding = clip_model.token_embedding(template_token
                                                            ).type(dtype)
        embedding_template = template_embedding[0, 1:1 + template_lens, :]
        self.register_buffer(
            "embedding_template", embedding_template
        )  # template

        print(f'Template: "{template}"')
        print(f"Number of context words (tokens): {template_lens}")

        template_mu = torch.empty(template_lens, embedding_dim, dtype=dtype)
        template_var = torch.empty(template_lens, embedding_dim, dtype=dtype)
        nn.init.normal_(template_mu, std=0.02)
        nn.init.normal_(template_var, std=0.02)
        self.template_mu = nn.Parameter(template_mu)  # to be optimized
        self.template_var = nn.Parameter(template_var)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [template + " " + name + "." for name in classnames]

        # prompt = template + classname
        prompts_token = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            prompts_embedding = clip_model.token_embedding(prompts_token
                                                           ).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer(
            "embedding_prefix", prompts_embedding[:, :1, :]
        )  # SOS
        self.register_buffer(
            "embedding_suffix", prompts_embedding[:, 1 + template_lens:, :]
        )  # CLS, EOS, BLANK

        self.n_cls = n_cls
        self.template_lens = template_lens
        self.tokenized_prompts = prompts_token  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        template = self.embedding_template
        template = template.unsqueeze(0).expand(self.n_cls, -1, -1)
        noise = self.sample(self.template_mu, self.template_var)
        noise = noise.expand(self.n_cls, -1, -1)
        noise_template = template + noise

        prefix = self.embedding_prefix
        suffix = self.embedding_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                noise_template,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

    def sample(self, mu, variance, num=1):
        """
        Args:
            mu:
            variance:
            num:

        Returns:
            Tensor: num * (n_cls) * template_lens * embedding_dim
        """
        # noise = noise.reshape(batch_size, num, dim)
        var = variance.expand(num, *variance.size())
        m = mu.expand(num, *mu.size())
        noise = torch.randn_like(var).to(var.device)
        noise = var*noise + m
        return noise


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.embed_dim = clip_model.visual.output_dim
        self.imageVpnGenerator = ImageVpnGenerator(
            self.embed_dim, cfg.TRAINER.CLIP_VPN.REDUCTION
        ).to(clip_model.dtype)
        self.promptVpnGenerator = PromptVpnGeneraotor(
            cfg, classnames, clip_model
        ).to(clip_model.dtype)
        self.tokenized_prompts = self.promptVpnGenerator.tokenized_prompts

    def forward(self, images, num=1):
        prompts = self.promptVpnGenerator()
        tokenized_prompts = self.tokenized_prompts
        text_feat = self.text_encoder(prompts, tokenized_prompts)
        visual_feat = self.image_encoder(images.type(self.dtype))
        # stop gradient propagation
        visual_feat = visual_feat.detach()
        # generate noise
        mu, variance = self.imageVpnGenerator(visual_feat, text_feat)
        noises = self.imageVpnGenerator.sample(mu, variance, num)
        if num == 1:
            noises = noises.squeeze(1)
        else:
            noises = noises.reshape(-1, *noises.size()[2:])
            visual_feat = visual_feat.expand(num, *visual_feat.size()
                                             ).transpose(0, 1)
            visual_feat = visual_feat.reshape(-1, *visual_feat.size()[2:])
        noise_visual_feat = visual_feat + noises
        noise_visual_feat = noise_visual_feat / noise_visual_feat.norm(
            dim=-1, keepdim=True
        )
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * noise_visual_feat @ text_feat.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_VPN(TrainerX):
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
            if 'VpnGenerator' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(
                self.model.promptVpnGenerator, cfg.MODEL.INIT_WEIGHTS[0]
            )
            load_pretrained_weights(
                self.model.imageVpnGenerator, cfg.MODEL.INIT_WEIGHTS[1]
            )

        self.model.to(self.device)

        self.optim = build_optimizer(
            None, cfg.OPTIM, [
                {
                    'params': self.model.promptVpnGenerator.parameters()
                }, {
                    'params': self.model.imageVpnGenerator.parameters()
                }
            ]
        )
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model(
            'imageVpnGenerator', self.model.imageVpnGenerator, self.optim,
            self.sched
        )
        self.register_model(
            'promptVpnGenerator', self.model.promptVpnGenerator, None, None
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
        noise_num = self.cfg.TRAINER.CLIP_VPN.NOISE_NUM
        if not noise_num == 1:
            label = label.expand(noise_num, *label.size()).transpose(0, 1)
            label = label.reshape(-1, *label.size()[2:])
        if prec == "amp":
            with autocast():
                output = self.model(image, noise_num)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, noise_num)
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
            if 'embedding_prefix' in state_dict:
                del state_dict['embedding_prefix']

            if 'embedding_suffix' in state_dict:
                del state_dict['embedding_suffix']

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

        heatmapNumList = [
            1 for i in range(self.num_classes)
        ]  # the number of drawn heatmap in each class

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images, labels = self.parse_batch_test(batch)
            prompts = self.model.promptVpnGenerator()
            tokenized_prompts = self.model.tokenized_prompts
            text_feat = self.model.text_encoder(prompts, tokenized_prompts)
            visual_feat = self.model.image_encoder(images.type(self.model.dtype))
            # generate noise
            mu, variance = self.model.imageVpnGenerator(visual_feat, text_feat)
            mu = mu.reshape(mu.shape[0], -1, 32)
            variance = variance.reshape(variance.shape[0], -1, 32)
            heatmap_show(
                images, mu, variance, labels, plot_dir, self.dm.dataset.classnames,
                heatmapNumList
            )
