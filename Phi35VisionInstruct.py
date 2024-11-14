from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class Phi35VisionInstruct:
    """
    A Phi-3.5 Vision Instruct node for image analysis
    """

    def __init__(self):
        self.model = None
        self.processor = None

    def initialize(self):
        """Lazy load model and processor"""
        if self.model is None:
            model_id = "microsoft/Phi-3.5-vision-instruct"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="flash_attention_2",
            )

            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, num_crops=4
            )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Desribe the image in detail",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "rinsanga"

    def analyze(self, image, instruction):
        # Initialize model if not done
        self.initialize()

        # Convert tensor image to PIL
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image = Image.fromarray((image * 255).numpy().astype("uint8"))

        # Prepare messages
        messages = [
            {"role": "user", "content": "<|image_1|>\n" + instruction},
        ]

        # Process prompt
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")

        # Generate response
        generation_args = {
            "max_new_tokens": 2000,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # Post-process response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return (response,)

