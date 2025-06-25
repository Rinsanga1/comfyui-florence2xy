from .florence2coordinatesbutxy import Florence2toCoordinatesButxy

from .load_img_with_name import LoadImageWithName

NODE_CLASS_MAPPINGS = {
    "Florence2toCoordinatesButxy": Florence2toCoordinatesButxy,
    "LoadImageWithName": LoadImageWithName
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2toCoordinatesButxy": "Florence2 Coordinates (XY Split)",
    "LoadImageWithName": "Load Image With Name"
}
