from typing import Optional, Tuple, List
from PIL import Image, ImageDraw
import numpy as np
from cortexia_video.schemes import BoundingBox, DetectionResult, SegmentationResult, FrameData


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))
def draw_bounding_box(
    draw, 
    box: BoundingBox,
    label: Optional[str] = None,
    score: Optional[float] = None,
    color: str = "red",
    thickness: int = 2
):
    # Draw bounding box rectangle
    draw.rectangle(
        [(box.x_min, box.y_min), (box.x_max, box.y_max)],
        outline=color,
        width=thickness
    )
    # Draw label and score if provided
    display_text = ""
    if label:
        display_text = label
    if score is not None:
        display_text += f" {score:.2f}" if display_text else f"{score:.2f}"
        
    if display_text:
        # Estimate label position, ensure it's within the image
        label_x = box.x_min
        label_y = max(0, box.y_min - 10)
        draw.text((label_x, label_y), display_text, fill=color)

def draw_segmentation_mask(
    image: Image.Image,
    mask_array: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> Image.Image:
    # Prepare the colored overlay
    base = np.array(image).copy()
    if base.ndim == 2:  # grayscale
        base = np.stack([base]*3, axis=-1)
    if base.shape[2] == 4:
        base = base[:, :, :3]
    colored_mask = np.zeros_like(base, dtype=np.uint8)
    colored_mask[mask_array == 1] = color
    # Convert mask to PIL image
    colored_mask_pil = Image.fromarray(colored_mask).convert('RGB')
    # Blend images
    blended = Image.blend(image.convert('RGB'), colored_mask_pil, alpha)
    return blended

def generate_annotated_frame(
    original_image: Image.Image,
    frame_data: FrameData
) -> Image.Image:
    annotated_img = original_image.copy()
    draw = ImageDraw.Draw(annotated_img, "RGBA")
    
    # Draw detections
    for detection in frame_data.detections:
        if detection.box:
            draw_bounding_box(draw, detection.box, detection.label)
        if detection.mask is not None:
            annotated_img = draw_segmentation_mask(annotated_img, detection.mask)
            draw = ImageDraw.Draw(annotated_img, "RGBA")  # re-init after new image
    
    # Draw segments if they're not already represented in detections
    for segment in frame_data.segments:
        # Check if this segment is already represented in a detection
        already_drawn = any(
            d.box.x_min == segment.bbox.x_min and
            d.box.y_min == segment.bbox.y_min and
            d.label == segment.label
            for d in frame_data.detections
        )
        
        if not already_drawn:
            draw_bounding_box(draw, segment.bbox, segment.label)
            annotated_img = draw_segmentation_mask(annotated_img, segment.mask)
            draw = ImageDraw.Draw(annotated_img, "RGBA")  # re-init after new image
    return annotated_img