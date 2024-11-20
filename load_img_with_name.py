import os
from PIL import Image, ImageOps
import folder_paths
import torch
import numpy as np
import requests
from io import BytesIO


class LoadImageWithName:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"
    CATEGORY = ".monks"

    def load_image(self, url, image):
        print(f"""Your input contains:
                url: {url}
                image: {image}
            """)

        if url != "" and url.startswith("http"):
            i = open_image_from_url(url)
            # Extract filename from URL and remove extension
            filename = url.split("/")[-1].rsplit(".", 1)[0]
        else:
            i = open_image_from_input(image)
            # Remove extension from local filename
            filename = image.rsplit(".", 1)[0]

        image_output, mask = back_image(i)
        return (image_output, mask, filename)


def open_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)
    return image


def open_image_from_input(file_name):
    image_path = folder_paths.get_annotated_filepath(file_name)
    image = Image.open(image_path)
    return image


def back_image(i):
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))


NODE_CLASS_MAPPINGS = {
    "LoadImageWithName": LoadImageWithName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithName": "Load Image With Name",
}
