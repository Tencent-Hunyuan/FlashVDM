# export PYTHONPATH="${PYTHONPATH}:ulip/ULIP:uni3d"
# cd ulip/pointnet2_ops-main
# python3 setup.py install

import glob
import json
import os
import time
import traceback
from datetime import datetime
from functools import partial

import gradio as gr
import torch
import trimesh
from PIL import Image
from ulip_score import ULIP
from uni3d_score import Uni3DScore

start_time = time.time()
ulip = ULIP()
uni3d = Uni3DScore()
print('model loaded, time used', time.time() - start_time)

CAPTION_PATHS = [
    'eval/ulip/caption.json',
]
IMAGE_PATHS = [
    'example_data/testset',
]
SAVE_PATH = 'example_data/metric.jsonl'


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


def compute_all(pred_path, caption_path, gt_path, progress=gr.Progress()):
    progress(0, desc="Starting...")

    captions = {}
    with open(caption_path, 'r') as f:
        for line in f.readlines():
            d = json.loads(line)
            captions[d['id']] = d['caption']

    uni3d_t_scores = []
    uni3d_i_scores = []
    ulip_t_scores = []
    files = glob.glob(gt_path + '/**', recursive=True)
    files = [f for f in files if f.endswith('.png')]
    files = [path.replace('_preprocess.png', 'png') for path in files]
    files = [path for path in files if 'QF13' not in path]
    files = list(set(files))

    for file in progress.tqdm(files):
        mesh_path = os.path.join(pred_path, file.replace(gt_path + '/', '').replace('.png', '0.glb'))
        caption = captions[file.replace(gt_path + '/', '')]

        if os.path.exists(mesh_path) == False:
            mesh_path = mesh_path.replace('0.glb', '.glb')

        if os.path.exists(mesh_path) == False:
            mesh_path = mesh_path.replace('.glb', '0.glb')
            if os.path.exists(mesh_path.replace('.glb', '_simplified.glb')):
                print(f"the file {mesh_path} is not exist, try to find the simplified one")
                mesh_path = mesh_path.replace('.glb', '_simplified.glb')
            else:
                print(f"The file {mesh_path} does not exist. Skipping...")
                continue

        try:
            mesh = load_mesh(mesh_path)

            samples, _ = trimesh.sample.sample_surface(mesh, 8192)
            pc = torch.from_numpy(samples).float().unsqueeze(0)
            images = [Image.open(file).convert('RGB')]

            outputs = uni3d(pc, [caption])
            uni3d_t_scores.append(outputs.item())

            outputs = uni3d.sim_img(pc, images)
            uni3d_i_scores.append(outputs.item())

            outputs = ulip(pc, [caption])
            ulip_t_scores.append(outputs.item())
        except Exception as e:
            traceback.print_exc()
            print('Error glb:' + mesh_path)
            continue

    avg_uni3d_t = sum(uni3d_t_scores) / len(uni3d_t_scores)
    avg_uni3d_i = sum(uni3d_i_scores) / len(uni3d_i_scores)
    avg_ulip_t = sum(ulip_t_scores) / len(ulip_t_scores)

    with open(SAVE_PATH, 'a') as f:
        date = datetime.today().strftime('%Y-%m-%d')
        f.write(json.dumps({'name': pred_path, 'date': date, 'ulip_t': avg_ulip_t, 'uni3d_t': avg_uni3d_t,
                            'uni3d_i': avg_uni3d_i}) + '\n')

    print(pred_path, avg_ulip_t, avg_uni3d_t, avg_uni3d_i)

    return load_metric_cache(SAVE_PATH)


def load_metric_cache(path):
    metrics = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            metrics.append([data['name'], data['date'], data['ulip_t'], data['uni3d_t'], data['uni3d_i']])
    return metrics


def compute(path):
    return [['abc', '2024/7/12', 1, 2, 3]]


def main():
    print('running metric app')
    pred_path = 'xx'
    compute_all(pred_path)


def app(port=443):
    with gr.Blocks() as demo:
        gr.Markdown('# Metric Computer')

        with gr.Group():
            caption_path = gr.Dropdown(label='Caption Path', choices=CAPTION_PATHS, allow_custom_value=True,
                                       value=CAPTION_PATHS[0])
            image_path = gr.Dropdown(label='Image Path', choices=IMAGE_PATHS, allow_custom_value=True,
                                     value=IMAGE_PATHS[0])
            path = gr.Textbox(label='Input Path')
            with gr.Column():
                btn = gr.Button()
                reload_btn = gr.Button(value='Reload')

        data_frame = gr.Dataframe(
            value=load_metric_cache(SAVE_PATH),
            headers=["name", "date", "ulip-t", "uni3d-t", "uni3d_i"],
            datatype=["str", "date", "number", "number", "number"],
            col_count=(5, "fixed"),
        )

        btn.click(compute_all, [path, caption_path, image_path], data_frame)
        reload_btn.click(partial(load_metric_cache, path=SAVE_PATH), outputs=data_frame)

    demo.launch(server_name='0.0.0.0', server_port=port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=443)
    args = parser.parse_args()
    app(args.port)
    # main()
