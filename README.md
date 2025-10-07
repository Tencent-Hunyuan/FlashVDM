<div align="center">
    <img src="https://github.com/user-attachments/assets/96a9c182-6ac9-4744-89d2-9f95aa1e7b67"  height=120>
</div>
</br>

⚡️⚡️ Try it Now with **[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)** for super fast high-quality shape generation within 1 second on 4090.




https://github.com/user-attachments/assets/a2cbc5b8-be22-49d7-b1c3-7aa2b20ba460





<div align="center">
  <a href=https://huggingface.co/spaces/tencent/Hunyuan3D-2mini-Turbo  target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/tencent/Hunyuan3D-2mini target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href="https://arxiv.org/abs/2503.16302" target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
</div>

## News

- **[2025-07-29]**: Released ULIP and Uni3D evaluation [code](evaluation/).
- **[2025-06-27]**: FlashVDM is accepted by ICCV 2025 as highlight paper.
- **[2025-03-19]**: FlashVDM is released and integrated into [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2).


## What is FlashVDM?
FlashVDM is a general framework for accelerating shape generation Vecset Diffusion Model (VDM), such as [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), [Michelangelo](https://github.com/NeuralCarver/Michelangelo), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D), [CLAY](https://github.com/CLAY-3D/OpenCLAY), [TripoSG](https://arxiv.org/abs/2502.06608), [Dora](https://github.com/Seed3D/Dora) and etc.

It features two techniques for both VAE and DiT acceleration: 

1. ***Lightning Vecset Decoder*** that drastically lowers decoding FLOPs without any loss in decoding quality, achieving over **45x speedup**.
2. ***Progressive Flow Distillation*** that enables flexible diffusion sampling with as few as **5 inference steps** and comparable quality.

<img src="https://github.com/user-attachments/assets/bcc1f43e-4cfa-47f3-9a45-421f75cf5138"  height=250>


#### Official Supported Model

- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2?tab=readme-ov-file#-models-zoo) 
- [Hunyuan3D-2mini](https://github.com/Tencent-Hunyuan/Hunyuan3D-2?tab=readme-ov-file#-models-zoo) 
- [Hunyuan3D-2mv](https://github.com/Tencent-Hunyuan/Hunyuan3D-2?tab=readme-ov-file#-models-zoo) 
- [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Hunyuan3D-Omni](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni)


#### Community Supported Model

- [HoloPart](https://github.com/VAST-AI-Research/HoloPart): Generative 3D Part Amodal Segmentation
- [TripoSG](https://github.com/VAST-AI-Research/TripoSG): High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models
- [DetailGen3D](https://github.com/VAST-AI-Research/DetailGen3D): Generative 3D Geometry Enhancement via Data-Dependent Flow
- [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D): High-fidelity Mesh Generation
with 3D Native Generation and Interactive Geometry Refiner
- [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D): Towards High-Fidelity and Controllable Generation of Textured 3D Assets
- [PartPacker](https://github.com/NVlabs/PartPacker): Efficient Part-level 3D Object Generation via Dual Volume Packing


## How to Use?

Visit **[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)** to access the integration of FlashVDM with Hunyuan3D-2.

```diff
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
-   subfolder='hunyuan3d-dit-v2-0',
+   subfolder='hunyuan3d-dit-v2-0-turbo',
    use_safetensors=True,
)
+pipeline.enable_flashvdm()

pipeline(
    image=image,
-   num_inference_steps=50,
+   num_inference_steps=5,
)[0]
```

## Citation

If you found this repository helpful, please cite our report:

```bibtex
@misc{lai2025unleashingvecsetdiffusionmodel,
      title={Unleashing Vecset Diffusion Model for Fast Shape Generation}, 
      author={Zeqiang Lai and Yunfei Zhao and Zibo Zhao and Haolin Liu and Fuyun Wang and Huiwen Shi and Xianghui Yang and Qingxiang Lin and Jingwei Huang and Yuhong Liu and Jie Jiang and Chunchao Guo and Xiangyu Yue},
      year={2025},
      eprint={2503.16302},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16302}, 
}
```
