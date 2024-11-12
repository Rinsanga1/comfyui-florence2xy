from .florence2coordinatesbutxy import Florence2toCoordinatesButxy
from .florence3img2txt import Florence3img2txt

NODE_CLASS_MAPPINGS = {
    "Florence2toCoordinatesButxy": Florence2toCoordinatesButxy,
    "Florence3img2txt": Florence3img2txt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2toCoordinatesButxy": "Florence2 Coordinates (XY Split)",
    "Florence3img2txt": "Florence3 Image to Text",
}
