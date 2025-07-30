import glob
import trimesh
from uni3d_score import Uni3DScore
import json
import os
import torch

def main(
    data_dir='example_data/testset_v1_1_12_merged_catetory',
    output_dir='testset_v1_1_12_merged_catetory'
):
    files = glob.glob(data_dir+'/**', recursive=True)
    ulip = Uni3DScore()
    
    captions = {}
    with open('caption.json', 'r') as f:
        for line in f.readlines():
            d = json.loads(line)
            captions[d['id']] = d['caption']
    
    scores = []
    
    for file in files:
        if not file.endswith('.png'):
            continue
        mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '0.glb'))
        caption = captions[file.replace(data_dir+'/', '')]
    
        mesh = trimesh.load(mesh_path, force='mesh')
        samples, _ = trimesh.sample.sample_surface(mesh, 8192)
        pc = torch.from_numpy(samples).float().unsqueeze(0)
        outputs = ulip(pc, [caption])
        
        scores.append(outputs.item())
        
    print('avg ulip score:', sum(scores)/len(scores))
        

if __name__ == '__main__':
    main()