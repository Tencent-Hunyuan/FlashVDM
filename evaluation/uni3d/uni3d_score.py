from collections import OrderedDict
import os

import torch
import open_clip

import models.uni3d as models
from utils.tokenizer import SimpleTokenizer
import munch
import timm

dtype = torch.float32


class Uni3DScore:
    def __init__(self):
        args = munch.munchify(dict(
            npoints=10000,
            num_group=512,
            group_size=64,
            pc_encoder_dim=512,
            clip_model='EVA02-E-14-plus',
            pretrained='uni3d/open_clip_pytorch_model.bin',
            pretrained_pc='',  # 'model.safetensors',
            drop_path_rate=0,
            pc_model='eva_giant_patch14_560.m30m_ft_in22k_in1k',  # 'eva_giant_patch14_560',
            pc_feat_dim=1408,
            embed_dim=1024,
            ckpt_path='uni3d/model.pt',
            patch_dropout=0,
        ))
        device = torch.device('cuda')

        self.clip_model, _, self.transforms = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained)
        self.clip_model.to(device).to(dtype)
        print('clip loaded')

        self.model = models.create_uni3d(args)
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        sd = checkpoint['module']
        self.model.load_state_dict(sd)
        self.model.to(device).to(dtype)
        self.tokenizer = SimpleTokenizer()

    @torch.no_grad()
    def __call__(self, pc, texts):
        text_features = []
        texts = self.tokenizer(texts).cuda()
        if len(texts.shape) == 1:
            texts = texts.unsqueeze(0)
        class_embeddings = self.clip_model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = class_embeddings

        pc = pc.cuda().to(dtype)
        rgb = torch.ones_like(pc) * (100.0 / 255.0)
        pc = torch.cat([pc, rgb], dim=-1)

        # print(pc.shape)
        # encode pc
        pc_features = self.model.encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_pc = pc_features.float() @ text_features.float().t()
        # similarity = torch.nn.functional.cosine_similarity(pc_features, text_features)

        return logits_per_pc

    @torch.no_grad()
    def sim_img(self, pc, images):
        image_list = [self.transforms(image).unsqueeze(0) for image in images]
        images = torch.cat(image_list, dim=0).cuda().to(dtype)

        class_embeddings = self.clip_model.encode_image(images)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        image_features = class_embeddings

        pc = pc.cuda().to(dtype)
        rgb = torch.ones_like(pc) * (100.0 / 255.0)
        pc = torch.cat([pc, rgb], dim=-1)

        # encode pc
        pc_features = self.model.encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_pc = pc_features.float() @ image_features.float().t()
        # similarity = torch.nn.functional.cosine_similarity(pc_features, image_features)
        # print(logits_per_pc.shape)
        # print(pc_features.shape)
        # print(image_features.shape)
        return logits_per_pc


if __name__ == '__main__':
    uni3d = Uni3DScore()
    import trimesh
    from PIL import Image

    mesh = trimesh.load('1.obj')
    samples, _ = trimesh.sample.sample_surface(mesh, 10000)
    pc = torch.from_numpy(samples).float().unsqueeze(0)

    texts = ['a point cloud model of a hat.', 'a point cloud model of a car.']
    outputs = uni3d(pc, texts)
    print(outputs)

    images = [Image.open('000.png').convert('RGB'), Image.open('car.png').convert('RGB')]
    outputs = uni3d.sim_img(pc, images)
    print(outputs)
