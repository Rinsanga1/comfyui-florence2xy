from .florence2coordinatesbutxy import Florence2toCoordinatesButxy
from .Phi35VisionInstruct import Phi35VisionInstruct

NODE_CLASS_MAPPINGS = {
    "Florence2toCoordinatesButxy": Florence2toCoordinatesButxy,
    "Phi35VisionInstruct": Phi35VisionInstruct
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2toCoordinatesButxy": "Florence2 Coordinates (XY Split)",
    "Phi35VisionInstruct": "Phi-3.5 Vision Instruct"
}
