from .florence2coordinatesbutxy import Florence2toCoordinatesButxy
from .BlipImgCaptioning import BlipImgCaptioning

NODE_CLASS_MAPPINGS = {
    "Florence2toCoordinatesButxy": Florence2toCoordinatesButxy,
    "BlipImgCaptioning": BlipImgCaptioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2toCoordinatesButxy": "Florence2 Coordinates (XY Split)",
    "BlipImgCaptioning": "BLIP Image Captioning",
}
