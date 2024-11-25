from transformers import AutoModelForCausalLM, AutoProcessor
import comfy.model_management as mm


class LoadPhi35VisionModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "flash_attention_2"},
                ),
            },
        }

    RETURN_TYPES = ("PHI35MODEL",)
    RETURN_NAMES = ("phi35_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Phi-Vision"

    def loadmodel(self, precision, attention):
        device = mm.get_torch_device()
        model_id = "microsoft/Phi-3.5-vision-instruct"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation=attention,
        )

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, num_crops=4
        )

        phi35_model = {"model": model, "processor": processor}
        return (phi35_model,)


NODE_CLASS_MAPPINGS = {
    "LoadPhi35VisionModel": LoadPhi35VisionModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPhi35VisionModel": "Load Phi-3.5 Vision Model",
}
