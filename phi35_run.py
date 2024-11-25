from PIL import Image
import comfy.model_management as mm


class Phi35VisionRun:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "phi35_model": ("PHI35MODEL",),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe the image in detail",
                    },
                ),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 2000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
                "do_sample": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "analyze"
    CATEGORY = "Phi-Vision"

    def analyze(
        self,
        image,
        phi35_model,
        instruction,
        keep_model_loaded=False,
        max_new_tokens=2000,
        temperature=0.0,
        do_sample=False,
    ):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model = phi35_model["model"]
        processor = phi35_model["processor"]

        model.to(device)

        # Convert tensor image to PIL
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image = Image.fromarray((image * 255).numpy().astype("uint8"))

        # Prepare messages
        messages = [
            {"role": "user", "content": "<|image_1|>\n" + instruction},
        ]

        # Process prompt
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = processor(prompt, [image], return_tensors="pt").to(device)

        # Generate response
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }

        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # Post-process response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if not keep_model_loaded:
            print("Offloading model...")
            model.to(offload_device)
            mm.soft_empty_cache()

        return (response,)


NODE_CLASS_MAPPINGS = {
    "Phi35VisionRun": Phi35VisionRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Phi35VisionRun": "Phi-3.5 Vision Run",
}

