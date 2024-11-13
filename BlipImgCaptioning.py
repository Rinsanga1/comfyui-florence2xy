import torch
import numpy as np
from PIL import Image


class BlipImgCaptioning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blip_model": ("BLIP_MODEL",),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe this image in detail",
                        "placeholder": "Enter your instruction here...",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "WAS Suite/Text"

    def tensor2pil(self, image):
        # Convert tensor to PIL Image
        image = (255.0 * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
        if len(image.shape) == 3:
            image = Image.fromarray(image)
        else:
            image = Image.fromarray(image[0])
        return image

    def describe_image(self, image, blip_model, instruction):
        try:
            # Convert tensor to PIL image
            pil_image = self.tensor2pil(image)

            # Convert instruction to a question format if it's not already
            if not instruction.endswith("?"):
                question = instruction + "?"
            else:
                question = instruction

            # Use answer_question method
            description = blip_model.answer_question(pil_image, question)

            print(f"\033[33mBLIP Description:\033[0m {description}")
            print(f"\033[33mInstruction used:\033[0m {question}")

            return (description,)

        except Exception as e:
            try:
                # Fallback to regular caption if question answering fails
                description = blip_model.generate_caption(pil_image)
                print(f"\033[33mFallback BLIP Description:\033[0m {
                      description}")
                print(f"\033[31mOriginal error:\033[0m {str(e)}")
                return (description,)
            except Exception as e2:
                print(f"\033[31mError in BLIP Image Description:\033[0m {str(e2)}")
                return ("Error generating description.",)
