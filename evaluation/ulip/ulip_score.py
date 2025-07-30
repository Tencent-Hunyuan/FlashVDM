import torch
from ULIP.models.ULIP_models import ULIP2_PointBERT_Colored, ULIP_PointBERT
from collections import OrderedDict
import os
from ULIP.utils.tokenizer import SimpleTokenizer
import timm


dtype = torch.float32


class ULIP:
    def __init__(self):
        ckpt_path = 'ULIP/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        model = ULIP_PointBERT(args=None)
        model.cuda()
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        data_config = timm.data.resolve_model_data_config(model.visual)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.model = model.to(dtype)
        self.tokenizer = SimpleTokenizer()

    @torch.no_grad()
    def __call__(self, pc, texts):
        text_features = []
        texts = self.tokenizer(texts).cuda()
        if len(texts.shape) == 1:
            texts = texts.unsqueeze(0)
        class_embeddings = self.model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = class_embeddings
        
        pc = pc.cuda().to(dtype)
        # encode pc
        pc_features = self.model.encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_pc = pc_features @ text_features.t()
        return logits_per_pc

    @torch.no_grad()
    def sim_img(self, pc, images):
        image_list = [self.transforms(image).unsqueeze(0) for image in images]
        images = torch.cat(image_list, dim=0).cuda().to(dtype)
        
        class_embeddings = self.model.encode_image(images)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        text_features = class_embeddings
        
        pc = pc.cuda().to(dtype)
        # encode pc
        pc_features = self.model.encode_pc(pc)
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_pc = pc_features @ text_features.t()
        return logits_per_pc

if __name__ == '__main__':
    ulip = ULIP()
    import trimesh
    from PIL import Image
    
    mesh = trimesh.load('0_out_mesh.obj')
    samples, _ = trimesh.sample.sample_surface(mesh, 8192)
    pc = torch.from_numpy(samples).float().unsqueeze(0)

    texts = ['a point cloud model of a plane.', 'a point cloud model of a car.']
    outputs = ulip(pc, texts)
    print(outputs)
    
    images = [Image.open('plane.png').convert('RGB'), Image.open('car.png').convert('RGB')]
    outputs = ulip.sim_img(pc, images)
    print(outputs)