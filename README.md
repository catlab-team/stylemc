
# StyleMC: Multi-Channel Based Fast Text-Guided Image Generation and Manipulation
![stylemc](https://github.com/catlab-team/stylemc/blob/main/images/examples.png)

Paper: https://arxiv.org/abs/2112.08493<br>
Video: https://www.youtube.com/watch?v=ILm_5tvtzPI<br>


## Installation and Usage
The easiest way to get started with StyleMC is to run our [Colab notebook](https://colab.research.google.com/drive/1ByojSKlTozy_DSxnZ5dPjXu7om5W1J5v?usp=sharing). 

Alternatively, you can run StyleMC on your local as follows:
- Install CLIP as a Python package following the official [CLIP repository](https://github.com/openai/CLIP) instructions
- Clone this repo and install the dependencies
```
git clone git@github.com:catlab-team/stylemc.git
cd stylemc
pip install -r requirements.txt
```

**Finding a global manipulation direction with a target text prompt:**  
Running find_direction.py finds and save a manipulation direction. You can then use the saved style direction to edit a randomly generated image or a real image.
```
python find_direction.py --text_prompt=[TEXT_PROMPT] --resolution=[RESOLUTION] --batch_size=[BATCH_SIZE] --identity_power=[ID_COEFF] --outdir=out --trunc=0.7 --seeds=1-129 --network=[PATH_TO_MODEL]

python find_direction.py --text_prompt="A man with mohawk hair" --resolution=256 --batch_size=4 --identity_power=high --outdir=out --trunc=0.7 --seeds=1-129 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

**(a) Manipulate a randomly generated image**    
Retrieve and save the style code of the image to be manipulated.
```
python generate_w.py --trunc=0.7 --seeds=8 --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
python w_s_converter.py --outdir=out --projected-w=encoder4editing/projected_w.npz --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

Perform the manipulation and save the resulting image using the original text prompt:
```
python generate_fromS.py --text_prompt="A man with mohawk hair" --change_power=50 --outdir=out --s_input=out/input.npz --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

**(b) Manipulate a real image**  
Retrieve and save the style code of the image to be manipulated.
```
python encoder4editing/infer.py --input_image [PATH_TO_INPUT_IMAGE]
python w_s_converter.py --outdir=out --projected-w=encoder4editing/projected_w.npz --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

Perform the manipulation and save the resulting image using the original text prompt:
```
python generate_fromS.py --text_prompt="A man with mohawk hair" --change_power=50 --outdir=out --s_input=out/input.npz --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

**Generation a video of the manipulation**  
Simply use the *from_video* flag to create video of the manipulation steps.
```
python generate_fromS.py --from_video --text_prompt="A man with mohawk hair" --change_power=50 --outdir=out --s_input=out/input.npz --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

## Citation
This repo is built on the official StyleGAN2 repo, please refer to NVIDIA's [repo](https://github.com/NVlabs/stylegan2-ada-pytorch) for further details.

If you use this code for your research, please cite our paper:
```
@inproceedings{kocasari2022stylemc,
  title     = {StyleMC: Multi-Channel Based Fast Text-Guided Image Generationand Manipulation},
  author    = {Umut Kocasari and Alara Dirik and Mert Tiftikci and Pinar Yanardag},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2022}
}
```
