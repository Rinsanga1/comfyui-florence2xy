import json


class Florence2toCoordinatesButxy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Changed 'JSON' to 'STRING' for 'data' input
                "data": ("STRING",),
                "source": ("IMAGE",),
                "index": ("STRING", {"default": "0"}),
                "batch": ("BOOLEAN", {"default": False}),
            },
        }

    # Changed 'BBOX' to 'STRING' in RETURN_TYPES
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("x", "y", "bboxes")
    FUNCTION = "segment"
    CATEGORY = "rinsanga"

    def segment(self, data, source, index, batch=False):
        print("Data received:", data)
        try:
            # Parse the JSON string
            coordinates = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")

        if not coordinates:
            return (0, 0, "[]")

        top_left_x_points = []
        top_left_y_points = []

        if index.strip():
            indexes = [int(i) for i in index.split(",")]
        else:
            indexes = list(range(len(coordinates[0])))

        print("Indexes:", indexes)
        bboxes = []

        if batch:
            for idx in indexes:
                if 0 <= idx < len(coordinates[0]):
                    for i in range(len(coordinates)):
                        bbox = coordinates[i][idx]
                        min_x, min_y, max_x, max_y = bbox
                        top_left_x_points.append(int(min_x))
                        top_left_y_points.append(int(min_y))
                        bboxes.append(bbox)
        else:
            for idx in indexes:
                if 0 <= idx < len(coordinates[0]):
                    bbox = coordinates[0][idx]
                    min_x, min_y, max_x, max_y = bbox
                    top_left_x_points.append(int(min_x))
                    top_left_y_points.append(int(min_y))
                    bboxes.append(bbox)
                else:
                    raise ValueError(f"There's nothing in index: {idx}")

        x_coordinate = top_left_x_points[0] if top_left_x_points else 0
        y_coordinate = top_left_y_points[0] if top_left_y_points else 0

        print("Top-Left X:", x_coordinate)
        print("Top-Left Y:", y_coordinate)

        # Convert bboxes list to JSON string for the return
        bboxes_json = json.dumps(bboxes)

        return (x_coordinate, y_coordinate, bboxes_json)
