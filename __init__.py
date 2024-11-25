from .phi35_run import Phi35VisionRun
from .phi35_loader import LoadPhi35VisionModel
from .florence2coordinatesbutxy import Florence2toCoordinatesButxy

from .load_img_with_name import LoadImageWithName

NODE_CLASS_MAPPINGS = {
    "Florence2toCoordinatesButxy": Florence2toCoordinatesButxy,
    "LoadImageWithName": LoadImageWithName,
    "LoadPhi35VisionModel": LoadPhi35VisionModel,
    "Phi35VisionRun": Phi35VisionRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2toCoordinatesButxy": "Florence2 Coordinates (XY Split)",
    "LoadImageWithName": "Load Image With Name",
    "LoadPhi35VisionModel": "Load Phi-3.5 Vision Model",
    "Phi35VisionRun": "Phi-3.5 Vision Run",
}
