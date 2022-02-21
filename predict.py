"""
download the model and directions before hand
"""
import tempfile
from pathlib import Path
import subprocess
import os
import glob
import shutil
import cv2
import cog

CALCULATED_DIRECTIONS = ["a photo of a face with beard",
                         "a photo of a face with long hair",
                         "a photo of a happy face",
                         "a photo of a face with curly hair",
                         "a photo of a frowning face",
                         "a photo of a face with blonde hair",
                         "a photo of a tanned skin face",
                         "a photo of a relieved face",
                         "a photo of an excited face"]


class Predictor(cog.Predictor):
    def setup(self):
        pre_calculated = [f'direction_f{prompt.replace(" ", "_")}.npz' for prompt in CALCULATED_DIRECTIONS]
        for file in os.listdir('out'):
            if file not in pre_calculated:
                os.remove(os.path.join('out', file))

    @cog.input("image", type=Path, help="Input facial image. The demo supports FFHQ.")
    @cog.input(
        "prompt",
        type=str,
        default="a photo of a face with beard",
        options=CALCULATED_DIRECTIONS,
        help="Choose a prompt with pre-calculated direction or specify the customized_prompt below",
    )
    @cog.input(
        "customized_prompt",
        type=str,
        default=None,
        help="Specify your customized prompt. If set, the prompt above will be ignored. "
             "More descriptive texts produce better results such as using \"a photo of a face with blonde hair\" "
             "instead of \"blonde hair\".",
    )
    @cog.input(
        "change_power",
        type=int,
        default=10,
        min=1,
        max=50,
        help="Choose the style change power.",
    )
    def predict(self, image, prompt, customized_prompt, change_power):
        text_prompt = customized_prompt if customized_prompt else prompt
        if f'direction_f{text_prompt.replace(" ", "_")}.npz' not in os.listdir('out'):
            # calculate new directions
            resolution = 256
            batch_size = 4
            identity_power = "high"
            subprocess.call(["python", "find_direction.py", f"--text_prompt=f{text_prompt}",
                             f"--resolution={resolution}", f"--batch_size={batch_size}",
                             f"--identity_power={identity_power}", "--outdir=out",
                             "--trunc=0.7", "--seeds=1-129", "--network=ffhq.pkl"])
        # generating encoder4editing/projected_w.npz
        subprocess.call(["python", "encoder4editing/infer.py", "--input_image", f"{str(image)}"])
        # generating out/input.npz
        subprocess.call(["python", "w_s_converter.py", "--outdir=out", "--projected-w=encoder4editing/projected_w.npz",
                        "--network=ffhq.pkl"])

        # generating output image
        subprocess.call(["python", "generate_fromS.py", f"--text_prompt=f{text_prompt}",
                         f"--change_power={change_power}", "--outdir=out",
                         "--s_input=out/input.npz", "--network=ffhq.pkl"])

        res = os.path.join("out", f'f{text_prompt.replace(" ", "_")}_{str(change_power)}.jpeg')
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        shutil.copyfile(res, str(out_path))
        return out_path
