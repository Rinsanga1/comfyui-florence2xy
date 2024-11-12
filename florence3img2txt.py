# florence3img2txt.py
import torch
import torch.nn.functional as F
import comfy.utils as mm

class Florence3img2txt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face": ("IMAGE",),
                "clothing": ("IMAGE",),
                "background": ("IMAGE",),
                "florence2_model": ("FL2MODEL",),  # Changed to FL2MODEL to match the loader
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "encode"
    CATEGORY = "rinsanga"

    def encode(self, face, clothing, background, florence2_model):
        print("Florence3img2txt encode called")
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model = florence2_model["model"]
        processor = florence2_model["processor"]
        dtype = florence2_model.get("dtype", torch.float32)
        model.to(device)

        prompts = [
            "<CAPTION> Describe the face of the person:",
            "<CAPTION> Describe the clothing of the person:",
            "<CAPTION> Describe the background of the person:",
        ]

        images = [face, clothing, background]
        out_results = []

        for img, prompt in zip(images, prompts):
            img_tensor = img.permute(0, 3, 1, 2)[0]
            image_pil = F.to_pil_image(img_tensor)

            inputs = processor(
                text=prompt,
                images=image_pil,
                return_tensors="pt",
                do_rescale=False
            ).to(dtype).to(device)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=True,
                num_beams=3,
            )

            results = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            out_results.append(results)

        # Offload model after processing
        model.to(offload_device)
        mm.soft_empty_cache()

        concatenated_captions = ", ".join(out_results)
        return (concatenated_captions,)
