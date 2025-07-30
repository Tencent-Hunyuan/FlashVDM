import glob
import trimesh
from uni3d_score import Uni3DScore
import json
import os
import torch
from PIL import Image
import argparse
import json

def load_mesh(input_path):
    try:
        raw_mesh = trimesh.load(input_path, force="mesh", merge_primitives=True)
    except ValueError as e:
        scene = trimesh.load(input_path)

        geometries = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            # get a copy of the geometry
            current = scene.geometry[geometry_name]

            if not isinstance(current, trimesh.Trimesh):
                continue
            # move the geometry vertices into the requested frame
            current.apply_transform(transform)

            current.metadata["name"] = node_name

            # save to our list of meshes
            geometries.append(current)

        raw_mesh = trimesh.util.concatenate(geometries)
    except IndexError as e:
        return None
    return raw_mesh

def compute_text(
    data_dir=None,
    output_dir=None
):
    files = glob.glob(data_dir+'/**', recursive=True)
    uni3d = Uni3DScore()
    
    captions = {}
    with open('ulip/caption.json', 'r') as f:
        for line in f.readlines():
            d = json.loads(line)
            captions[d['id']] = d['caption']
    
    scores = []
    
    for file in files:
        if not file.endswith('.png'):
            continue
        mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '0.glb'))
        # if 'cratsman' in output_dir:
        #     mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '.obj'))
        # else:
        #     mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '0.glb'))
        
        caption = captions[file.replace(data_dir+'/', '')]

        if os.path.exists(mesh_path) == False:
            print(f"The file {mesh_path} does not exist. Skipping...")
            continue
    
        #mesh = trimesh.load(mesh_path, force='mesh')
        mesh = load_mesh(mesh_path)
        if mesh == None:
            print('Error glb:' + mesh_path)
            continue

        samples, _ = trimesh.sample.sample_surface(mesh, 8192)
        pc = torch.from_numpy(samples).float().unsqueeze(0)
        outputs = uni3d(pc, [caption])
        
        scores.append(outputs.item())
        
    print('avg uni3d text score:', sum(scores)/len(scores))
    print('count: ' + str(len(scores)))
    avg_uni3d = sum(scores)/len(scores)
    return avg_uni3d, len(scores)

def compute_image(
    data_dir=None,
    output_dir=None
):
    files = glob.glob(data_dir+'/**', recursive=True)
    uni3d = Uni3DScore()
    
    captions = {}
    with open('ulip/caption.json', 'r') as f:
        for line in f.readlines():
            d = json.loads(line)
            captions[d['id']] = d['caption']
    
    scores = []
    
    for file in files:
        if not file.endswith('.png'):
            continue
        mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '0.glb'))
        # if 'cratsman' in output_dir:
        #     mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '.obj'))
        # else:
        #     mesh_path = os.path.join(output_dir, file.replace(data_dir+'/', '').replace('.png', '0.glb'))
        
        caption = captions[file.replace(data_dir+'/', '')]

        if os.path.exists(mesh_path) == False:
            print(f"The file {mesh_path} does not exist. Skipping...")
            continue
    
        #mesh = trimesh.load(mesh_path, force='mesh')
        mesh = load_mesh(mesh_path)
        if mesh == None:
            print('Error glb:' + mesh_path)
            continue
        
        samples, _ = trimesh.sample.sample_surface(mesh, 8192)
        pc = torch.from_numpy(samples).float().unsqueeze(0)
        images = [Image.open(file).convert('RGB')]
        outputs = uni3d.sim_img(pc, images)
        scores.append(outputs.item())
        
    print('avg uni3d image score:', sum(scores)/len(scores))
    print('count: ' + str(len(scores)))
    avg_uni3d = sum(scores)/len(scores)
    return avg_uni3d, len(scores)


def test_text(path_list, output_path):
    data_dir='testset/testset_v1_1_12_merged_catetory'
    avg_uni3d_list = []
    for path in path_list:
        avg_uni3d, sum_num = compute_text(data_dir=data_dir, output_dir=path)
        avg_uni3d_list.append([avg_uni3d, sum_num])
    res_dict = dict(zip(path_list, avg_uni3d_list))

    import json
    # 打开一个文件以写入模式
    with open(os.path.join(output_path,'avg_uni3d_text.json'), 'w', encoding='utf-8') as file:
    # 使用json.dump()方法将字典写入文件
        json.dump(res_dict, file, ensure_ascii=False, indent=4)
    print('Finished!')

def test_image(path_list, output_path):
    data_dir='testset/testset_v1_1_12_merged_catetory'
    avg_uni3d_list = []
    for path in path_list:
        avg_uni3d, sum_num = compute_image(data_dir=data_dir, output_dir=path)
        avg_uni3d_list.append([avg_uni3d, sum_num])
    res_dict = dict(zip(path_list, avg_uni3d_list))

    import json
    # 打开一个文件以写入模式
    with open(os.path.join(output_path,'avg_uni3d_image.json'), 'w', encoding='utf-8') as file:
    # 使用json.dump()方法将字典写入文件
        json.dump(res_dict, file, ensure_ascii=False, indent=4)
    print('Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test_all", action='store_true', default=False)
    parser.add_argument("--is_test_image", action='store_true', default=False)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--metric_save_path", type=str, default=None)
    args = parser.parse_args()

    input_paths = args.input_path.split(',')
    input_path = []
    for item in input_paths:
        if item == '':
            continue
        input_path.append(item)
    input_paths = input_path

    if args.is_test_all:
        print('*'*20)
        print('eval 3d-text')
        test_text(input_paths, args.metric_save_path)
        print('*'*20)

        print('-'*20)
        print('eval 3d-image')
        test_image(input_paths, args.metric_save_path)
        print('-'*20)
    else:
        if args.is_test_image:
            print('-'*20)
            print('eval 3d-image')
            test_image(input_paths, args.metric_save_path)
            print('-'*20)
        else:
            print('*'*20)
            print('eval 3d-text')
            test_text(input_paths, args.metric_save_path)
            print('*'*20)
    