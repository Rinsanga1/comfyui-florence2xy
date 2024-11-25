import json


class Florence2toCoordinatesButxy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("JSON",),
                "source": ("IMAGE",),
                "index": ("STRING", {"default": "0"}),
                "batch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "BBOX")
    RETURN_NAMES = ("x", "y", "bboxes")
    FUNCTION = "segment"
    CATEGORY = "RinaRalte"

    def segment(self, data, source, index, batch=False):
        print(data)
        try:
            coordinates = data.replace("'", '"')
            coordinates = json.loads(coordinates)
        except:
            coordinates = data
        print("Type of data:", type(data))
        print("Data:", data)
        if len(data) == 0:
            return 0, 0, []  # Return integers
        top_left_x_points = []
        top_left_y_points = []
        if index.strip():  # Check if index is not empty
            indexes = [int(i) for i in index.split(",")]
        else:  # If index is empty, use all indices from data[0]
            indexes = list(range(len(data[0])))
        print("Indexes:", indexes)
        bboxes = []
        if batch:
            for idx in indexes:
                if 0 <= idx < len(data[0]):
                    for i in range(len(data)):
                        bbox = data[i][idx]
                        min_x, min_y, max_x, max_y = bbox
                        # Just use min_x and min_y (top-left corner)
                        top_left_x_points.append(int(min_x))
                        top_left_y_points.append(int(min_y))
                        bboxes.append(bbox)
        else:
            for idx in indexes:
                if 0 <= idx < len(data[0]):
                    bbox = data[0][idx]
                    min_x, min_y, max_x, max_y = bbox
                    # Just use min_x and min_y (top-left corner)
                    top_left_x_points.append(int(min_x))
                    top_left_y_points.append(int(min_y))
                    bboxes.append(bbox)
                else:
                    raise ValueError(f"There's nothing in index: {idx}")
        # Return integers
        x_coordinate = top_left_x_points[0] if top_left_x_points else 0
        y_coordinate = top_left_y_points[0] if top_left_y_points else 0
        print("Top-Left X:", x_coordinate)
        print("Top-Left Y:", y_coordinate)
        return x_coordinate, y_coordinate, bboxes
