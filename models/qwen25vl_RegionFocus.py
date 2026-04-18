# ==============================================================================
# Copyright (c) 2025 Tiange Luo, tiange.cs@gmail.com
#
# This code is licensed under the MIT License.
# ==============================================================================

import os
import re
import json
import base64
import tempfile
import numpy as np
import cv2
import io
import math
from PIL import Image, ImageDraw
from openai import OpenAI

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from qwen_utils_agent_function_call import ComputerUse


from PIL import Image, ImageDraw, ImageColor

def draw_point(image: Image.Image, point: list, color=None):
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)  
        except ValueError:
            color = (255, 0, 0, 128)  
    else:
        color = (255, 0, 0, 128)  

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color
    )
    
    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius), 
         (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255)
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ---------------------
# Utility functions
# ---------------------

def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None

def extract_bbox(s):
    """
    Given a string containing Qwen bounding box like:
    <|box_start|>(200,100),(400,300)<|box_end|>
    extract and return ((200,100),(400,300)).
    """
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        # return the last match
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    return None

def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name

def image_to_base64(image_path):
    """Convert an image to base64 string."""
    if not isinstance(image_path, str):
        assert isinstance(image_path, Image.Image)
        image_buffer = io.BytesIO()
        image_path.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
    else:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    
    return base64.b64encode(image_bytes).decode("utf-8")

def plot_points_on_image(image, points, colors=None, sizes=None, markers=None, labels=None, save_path=None):
    """
    Draw points on the image with custom colors, sizes, markers, and optional labels.
    
    Args:
        image: PIL Image or numpy array
        points: List of (x, y) coordinates
        colors: List of colors for each point (default is magenta)
        sizes: List of sizes for each point (default is 10)
        markers: List of marker types ('star', 'circle', 'square', 'cross', 'diamond')
        labels: Optional list of text labels for each point
        save_path: Optional path to save the annotated image
        
    Returns:
        The annotated image as a PIL Image
    """
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.copy()
    
    draw = ImageDraw.Draw(image_pil)
    
    if colors is None:
        colors = [(255, 0, 255) for _ in range(len(points))]  # Default magenta
    elif isinstance(colors, tuple) and len(colors) == 3:
        colors = [colors for _ in range(len(points))]
        
    if sizes is None:
        sizes = [10 for _ in range(len(points))]
    elif isinstance(sizes, int):
        sizes = [sizes for _ in range(len(points))]
        
    if markers is None:
        markers = ['star' for _ in range(len(points))]
    elif isinstance(markers, str):
        markers = [markers for _ in range(len(points))]
    
    for i, (x, y) in enumerate(points):
        x, y = int(x), int(y)
        color = colors[i] if i < len(colors) else (255, 0, 255)
        size = sizes[i] if i < len(sizes) else 10
        marker = markers[i] if i < len(markers) else 'star'
        
        # Draw different marker types
        if marker == 'star':
            # Draw a star
            points = []
            for j in range(5):
                # Outer points of the star
                angle_outer = math.pi/2 + j * 2*math.pi/5
                px_outer = x + size * math.cos(angle_outer)
                py_outer = y + size * math.sin(angle_outer)
                points.append((px_outer, py_outer))
                
                # Inner points of the star
                angle_inner = math.pi/2 + (j+0.5) * 2*math.pi/5
                px_inner = x + size/2 * math.cos(angle_inner)
                py_inner = y + size/2 * math.sin(angle_inner)
                points.append((px_inner, py_inner))
            
            draw.polygon(points, fill=color)
            
        elif marker == 'circle':
            draw.ellipse((x-size, y-size, x+size, y+size), fill=color)
            
        elif marker == 'square':
            draw.rectangle((x-size, y-size, x+size, y+size), fill=color)
            
        elif marker == 'cross':
            draw.line((x-size, y-size, x+size, y+size), fill=color, width=2)
            draw.line((x-size, y+size, x+size, y-size), fill=color, width=2)
            
        elif marker == 'diamond':
            draw.polygon([(x, y-size), (x+size, y), (x, y+size), (x-size, y)], fill=color)
        
        # Add label if provided
        if labels and i < len(labels):
            label = labels[i]
            draw.text((x+size+2, y-size-2), str(label), fill=color)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_pil.save(save_path)
        
    return image_pil

def save_debug_image(image, filename_prefix, coords=None, radius=5, color=(255, 0, 255), 
                    thickness=-1, task_id=None, index=None):
    """
    Save a debug image with an optional marker at coordinates.
    
    Args:
        image: The image to save (PIL Image or numpy array)
        filename_prefix: Prefix for the saved image filename
        coords: Optional tuple of (x, y) coordinates to draw a circle at
        radius: Radius of the circle if coords is provided
        color: Color of the circle in RGB format
        thickness: Thickness of the circle (-1 for filled)
        task_id: Optional task ID for directory organization
        index: Optional index to include in the filename
    """
    # Create debug directory
    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_to_save = np.array(image)
    else:
        img_to_save = image.copy()
    
    # Draw circle if coordinates are provided
    if coords is not None:
        if len(img_to_save.shape) == 2:  # Grayscale
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2RGB)
        cv2.circle(img_to_save, (int(coords[0]), int(coords[1])), 
                  radius, color, thickness)
    
    # Create filename with index if provided
    if index is not None:
        filename = f"{filename_prefix}_{index}.png"
    else:
        filename = f"{filename_prefix}.png"
    
    filepath = os.path.join(debug_dir, filename)
    
    # Save the image
    if isinstance(img_to_save, np.ndarray):
        cv2.imwrite(filepath, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(img_to_save).save(filepath)
    
    print(f"Saved debug image to {filepath}")


# ---------------------
# Main class
# ---------------------

class Qwen25VLModel():
    def __init__(self, 
                 base_url="http://localhost:8300/v1",
                 api_key="empty",
                 model_name="Qwen/Qwen2.5-VL-72B-Instruct"):
        """
        Initialize the client that calls your remote model.
        :param base_url: The URL of your inference endpoint
        :param api_key:  The API key (if any)
        :param model_name: Model name for remote inference
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.regionfocus_coords = []
        self.generation_config = {}

    def load_model(self):
        """
        In the new endpoint-based version, we do not actually
        load a model locally. This can be a no-op (or removed).
        """
        pass

    def set_generation_config(self, **kwargs):
        """
        If your endpoint supports custom generation parameters
        (e.g., temperature, max tokens, etc.), you can set them
        here. Otherwise, this can be unused or extended as needed.
        """
        self.generation_config = kwargs

    def _call_endpoint(self, messages, temperature=0, top_p=1.0):
        """
        Helper method to call the OpenAI-compatible API endpoint with robust error handling.
        """
        max_retries = 2
        timeout = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    max_tokens=1024,
                    #**self.generation_config
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff time (1s, 2s, 4s, etc.)
                    wait_time = 2 ** attempt
                    print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"Error calling API after {max_retries} attempts: {e}")
                    return "Error: Unable to get a response from the model after multiple attempts."

    def ground(self, instruction, image):
        """
        Implementation derived from original ground method.
        This calls the UI-TARS model to find a point/bounding box for the instruction.
        """
        if not isinstance(image, str):
            # If it's a PIL.Image, save it temporarily
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
        else:
            image_path = image
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        input_image = Image.open(image_path)
        encoded_string = encode_image(image_path)
        resized_height, resized_width = smart_resize(
            input_image.height,
            input_image.width,
            min_pixels=3136,
            max_pixels=12845056,
        )
        display_image = input_image.resize((resized_width, resized_height))
        computer_use = ComputerUse(
            cfg={"display_width_px": resized_width, "display_height_px": resized_height}
        )

        # Build messages
        system_message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        system_message = system_message[0].model_dump()
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": 3136,
                        "max_pixels": 12845056,
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                    },
                    # {"type": "text", "text": instruction},
                    {"type": "text", "text": f'Output the most relevant point in the image corresponding to the instruction "{instruction}" with grounding.'},
                ],
            }
        ]

        # 4) Call the endpoint
        response = self._call_endpoint(messages)

        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n')[0])
            coordinates = action['arguments']['coordinate']
            # display_image = draw_point(display_image, coordinates, color='green')

            # 5) Parse and build result dict
            result_dict = {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "point": [coordinates[0]/resized_width, coordinates[1]/resized_height],
            }
        except:
            result_dict = {
                "result": "wrong_format",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "point": None
            }

        return result_dict, display_image, system_message

    def calculate_crop_region(self, coords, img, viewport_width=1280, viewport_height=720, 
                             ratio_x=0.5, ratio_y=0.5, min_size=100, debug=False, task_id=None, index=None):
        """
        Calculate the crop region based on focus coordinates and viewport dimensions.
        
        Args:
            coords: Tuple of (x, y) coordinates for the focus point
            viewport_width: Width of the viewport
            viewport_height: Height of the viewport
            ratio_x: Ratio of the viewport to use for cropping (default: 0.5)
            ratio_y: Ratio of the viewport to use for cropping (default: 0.5)
            
        Returns:
            tuple: (left, top, width, height) of the crop region
        """
        x_center, y_center = coords
        
        # Ensure coordinates are within viewport bounds
        viewport_width, viewport_height = img.size
        if x_center > viewport_width or y_center > viewport_height:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! regionfocus_returns coords out of bounds')
            print(coords)
            x_center = min(x_center, viewport_width)
            y_center = min(y_center, viewport_height)
        
        # Calculate crop dimensions (half the viewport by default)
        crop_w = float(viewport_width * ratio_x)
        crop_h = float(viewport_height * ratio_y)
        
        # Initial crop region centered on focus point
        left = x_center - crop_w / 2
        top = y_center - crop_h / 2
        right = left + crop_w
        bottom = top + crop_h
        
        # Adjust horizontally if out of bounds
        if left < 0:
            shift = -left
            left += shift
            right += shift
        if right > viewport_width:
            shift = right - viewport_width
            left -= shift
            right -= shift
            
        # Adjust vertically if out of bounds
        if top < 0:
            shift = -top
            top += shift
            bottom += shift
        if bottom > viewport_height:
            shift = bottom - viewport_height
            top -= shift
            bottom -= shift
        
        # Final safety check to ensure values are within bounds
        left = max(0, left)
        top = max(0, top)
        right = min(viewport_width, right)
        bottom = min(viewport_height, bottom)
        
        # Return as (left, top, width, height)
        
        if debug:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Draw the crop region on a copy of the image
            debug_img = img.copy()
            draw = ImageDraw.Draw(debug_img)
            
            # Draw the point of interest
            point_radius = 5
            draw.ellipse((x_center-point_radius, y_center-point_radius, x_center+point_radius, y_center+point_radius), 
                         fill=(255, 0, 0))
            
            # Draw the crop rectangle
            rect_coords = [(left, top), (left + crop_w, top), 
                           (left + crop_w, top + crop_h), (left, top + crop_h)]
            draw.line(rect_coords + [rect_coords[0]], fill=(0, 255, 0), width=2)
            
            # Save the debug image
            crop_debug_filename = f"crop_region_{index}.png" if index is not None else "crop_region.png"
            debug_img.save(os.path.join(debug_dir, crop_debug_filename))
        
        return left, top, right - left, bottom - top

    def judge_inference(self, instruction, image, point, debug=False, task_id=None, system_message=None):
        """
        Judge whether the initial inference is correct.
        
        Args:
            instruction: The instruction text
            image: PIL Image or path to image
            point: The initial point from ground() function
            debug: Whether to save debug images
            task_id: Optional task ID for directory organization
            
        Returns:
            bool: True if the inference seems correct, False otherwise
        """
        # Create a copy of the image for highlighting the point
        if isinstance(image, str):
            with open(image, "rb") as f:
                pil_image = Image.open(image).copy()
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).copy()
        else:
            pil_image = image.copy()
        
        # Highlight the initial point with a pink star
        highlighted_image = plot_points_on_image(
            pil_image,
            [point],
            colors=[(255, 0, 255, 128)],  
            markers=['star'],
            sizes=[12]
        )
        
        if debug:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            highlighted_image.save(os.path.join(debug_dir, "initial_point_highlighted.png"))
        
        # Convert highlighted image to base64
        image_buffer = io.BytesIO()
        highlighted_image.save(image_buffer, format="PNG")
        encoded_string = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        
        # Create prompt for judgment
        judge_prompt = (
            f'Given the instruction: "{instruction}", I highlighted a pink star on the image, '
            f'Is this pink star position correct and precise for the instruction? '
            f'Sometimes, the point might cover the target, which is correct, and you need to distinguish this scenario.'
            f'Answer YES if it accurately identifies the element mentioned in the instruction. '
            f'Answer NO if it\'s incorrect or imprecise. '
            f'Thoughts: Please explain your reasoning and be specific about why the point is correct or incorrect.'
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "min_pixels": 3136, "max_pixels": 12845056, "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                    {"type": "text", "text": judge_prompt}
                ],
            }
        ]
        
        # Call the endpoint
        response = self._call_endpoint(messages)
        
        # Extract the judgment
        is_correct = ("YES" in response.upper() or "CORRECT" in response.upper() or "正确" in response or "精准" in response) and not ("NO" in response.upper() or "INCORRECT" in response.upper() or "不正确" in response or "不精准" in response)
        
        if debug:
            # Save the judgment response to a file
            with open(os.path.join(debug_dir, "judgment_response.txt"), "w") as f:
                f.write(f"Instruction: {instruction}\n\n")
                f.write(f"Point: {point}\n\n")
                f.write(f"Judgment: {'CORRECT' if is_correct else 'INCORRECT'}\n\n")
                f.write(f"Response:\n{response}")
        
        return is_correct, response

    def crop_and_upsample(self, bbox, image, debug=False, task_id=None, index=None, keep_aspect_ratio=True):
        """
        Given bbox (x, y, w, h), this function:
          1) Screenshots the *entire* page/image.
          2) Crops out the bounding box.
          3) Calculates a zoom factor so the cropped area would fit 
             within the current viewport width/height.
          4) Upsamples the cropped region to simulate "zoom."
          5) Returns the upsampled image, zoom_x, zoom_y, offset_w, offset_h.
        """
        # Get image as PIL Image
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Extract bounding box coordinates
        left, top, w, h = bbox
        
        # Ensure coordinates are valid
        left = max(0, left)
        top = max(0, top)
        w = min(w, img_width - left)
        h = min(h, img_height - top)
        
        # Crop the bounding box
        cropped = img.crop((left, top, left + w, top + h))
        
        if debug:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            crop_filename = f"crop_{index}.png" if index is not None else "crop.png"
            cropped.save(os.path.join(debug_dir, crop_filename))
        
        # Define target viewport size (standard size for consistency)
        viewport_width = img_width
        viewport_height = img_height
        
        if not keep_aspect_ratio:
            # Simply resize to viewport dimensions
            upsampled = cropped.resize((viewport_width, viewport_height), Image.Resampling.LANCZOS)
            zoom_x = viewport_width / w
            zoom_y = viewport_height / h
            offset_w = 0
            offset_h = 0
        else:
            # Calculate zoom factors to maintain aspect ratio
            zoom_x = viewport_width / w
            zoom_y = viewport_height / h
            zoom_factor = min(zoom_x, zoom_y)
            
            # Apply same zoom factor to both dimensions to maintain aspect ratio
            new_w = round(w * zoom_factor)
            new_h = round(h * zoom_factor)
            upsampled = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Calculate offsets to center the image in the viewport
            offset_w = float(viewport_width - new_w) / 2
            offset_h = float(viewport_height - new_h) / 2
            
            # Use same zoom factor for both dimensions when preserving aspect ratio
            zoom_x = zoom_factor
            zoom_y = zoom_factor
        
        if debug:
            upsampled_filename = f"upsampled_{index}.png" if index is not None else "upsampled.png"
            upsampled.save(os.path.join(debug_dir, upsampled_filename))
        
        # Convert PIL Image to bytes
        output_buffer = io.BytesIO()
        upsampled.save(output_buffer, format="PNG")
        screenshot_bytes = output_buffer.getvalue()
        
        return screenshot_bytes, zoom_x, zoom_y, offset_w, offset_h

    def auto_focus(self, instruction, image, debug=False, task_id=None, temperature=0, top_p=1.0, system_message=None):
        """
        Identifies points of interest in the image based on the instruction.
        Similar to the auto_focus method in the mother codebase.
        
        Args:
            instruction: The instruction text
            image: PIL Image or path to image
            debug: Whether to save debug images
            task_id: Optional task ID for directory organization
            temperature: Temperature for generation (higher = more diverse)
            top_p: Top-p for generation
            
        Returns:
            tuple: (point, response) - point is (x, y) coordinates, response is the model output
        """
        # Convert image to base64
        if isinstance(image, str):
            with open(image, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            pil_image = Image.open(image)
        else:
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            encoded_string = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
            pil_image = image
            
        # Get image dimensions
        img_width, img_height = pil_image.size
        
        # Create prompt for region focus
        regionfocus_prompt = (
            f'Given the instruction: "{instruction}", locate the most relevant coordinates in the image that best matches the instruction.'
        )

        full_prompt = regionfocus_prompt

        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "min_pixels": 3136, "max_pixels": 12845056, "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                    {"type": "text", "text": full_prompt}
                ],
            }
        ]
        
        # Call the endpoint with specified temperature and top_p
        response = self._call_endpoint(messages, temperature=temperature, top_p=top_p)
        
        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n')[0])
            coordinates = action['arguments']['coordinate']
            return [coordinates[0]/img_width, coordinates[1]/img_height], response
        except:
            return None, response

    def next_action_regionfocus(self, instruction, zoomed_img_bytes, left, top, zoom_x, zoom_y, 
                              offset_w, offset_h, w, h, original_image, debug=False, 
                              task_id=None, index=None, temperature=0, top_p=1.0, system_message=None):
        """
        Predicts action on a zoomed region and projects coordinates back to original image.
        
        Args:
            instruction: The instruction text
            zoomed_img_bytes: Bytes of the zoomed image
            left, top: Original crop region top-left coordinates
            zoom_x, zoom_y: Zoom factors for x and y directions
            offset_w, offset_h: Offsets in the zoomed image (for centering)
            w, h: Width and height of the original crop region
            original_image: The original image for reference
            debug: Whether to save debug images
            task_id: Optional task ID for directory organization
            index: Optional index for the region focus point
            temperature, top_p: Generation parameters
            
        Returns:
            tuple: (projected_point, response) where projected_point is coords in original image space
        """
        # Convert zoomed image bytes to base64
        encoded_string = base64.b64encode(zoomed_img_bytes).decode("utf-8")
        
        # Create prompt for action on zoomed region
        action_prompt = (
            f'For this zoomed-in screenshot, identify the precise point that best matches '
            f'the instruction: "{instruction}". '
        )
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "min_pixels": 3136, "max_pixels": 12845056, "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                    {"type": "text", "text": action_prompt}
                ],
            }
        ]
        
        # Call the endpoint
        response = self._call_endpoint(messages, temperature=temperature, top_p=top_p)

        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n')[0])
            click_point = action['arguments']['coordinate']
        except:
            return None, response

        if click_point:
            x_upsampled, y_upsampled = click_point
            # Get dimensions of the zoomed image
            zoomed_img = Image.open(io.BytesIO(zoomed_img_bytes))
            print(f"zoomed_img: {zoomed_img.size}")
            zoomed_width, zoomed_height = zoomed_img.size

            x_upsampled = round(x_upsampled)
            y_upsampled = round(y_upsampled)
            
            # Calculate coordinates relative to the zoomed content
            rel_zoomed_x = x_upsampled
            rel_zoomed_y = y_upsampled
            
            # Project back to original image coordinates
            zoomed_width = w * zoom_x
            zoomed_height = h * zoom_y
            
            # Convert from zoomed coordinates to original coordinates
            if 0 <= rel_zoomed_x < zoomed_width and 0 <= rel_zoomed_y < zoomed_height:
                # Within the zoomed area - project back to original coordinates
                #x_orig = left + ((rel_zoomed_x - offset_w) / zoom_x)
                #y_orig = top + ((rel_zoomed_y - offset_h) / zoom_y)
                x_orig = left + (rel_zoomed_x / zoom_x)
                y_orig = top + (rel_zoomed_y / zoom_y)
            else:
                # Outside zoomed area - clamp to the nearest edge of the zoomed content
                clamped_rel_x = max(0, min(zoomed_width - 1, rel_zoomed_x))
                clamped_rel_y = max(0, min(zoomed_height - 1, rel_zoomed_y))
                
                x_orig = left + (clamped_rel_x / zoom_x)
                y_orig = top + (clamped_rel_y / zoom_y)
            
            # Ensure projected coordinates are within original image bounds
            if isinstance(original_image, Image.Image):
                img_width, img_height = original_image.size
            else:
                img_height, img_width = original_image.shape[:2]
                
            x_orig = max(0, min(x_orig, img_width - 1))
            y_orig = max(0, min(y_orig, img_height - 1))
            
            # Create point tuple with round values
            projected_point = (round(x_orig), round(y_orig))
            
            if debug:
                # Save debug image with both the zoomed point and the projected point
                if isinstance(original_image, str):
                    original_pil = Image.open(original_image).copy()
                elif isinstance(original_image, np.ndarray):
                    original_pil = Image.fromarray(original_image).copy()
                else:
                    original_pil = original_image.copy()
                    
                # Draw zoomed coordinates on zoomed image
                zoomed_debug = plot_points_on_image(
                    zoomed_img,
                    [(x_upsampled, y_upsampled)],
                    colors=[(255, 0, 255)],
                    markers=['star'],
                    sizes=[15]
                )
                
                # Draw projected coordinates on original image
                original_debug = plot_points_on_image(
                    original_pil,
                    [projected_point],
                    colors=[(255, 0, 255)],
                    markers=['star'],
                    sizes=[15]
                )
                
                debug_dir = f"./debug/{task_id}" if task_id else "./debug"
                os.makedirs(debug_dir, exist_ok=True)
                
                zoomed_debug.save(os.path.join(debug_dir, f"RegionFocus_upsampled_{index}.png"))
                original_debug.save(os.path.join(debug_dir, f"RegionFocus_unprojected_{index}.png"))
            
            return projected_point, response
            
        return None, response

    def next_action_regionfocus_aggregation(self, instruction, image, points, debug=False, task_id=None, system_message=None):
        """
        Aggregates multiple predicted points and selects the best one based on model judgment.
        
        Args:
            instruction: The instruction text
            image: The original image
            points: List of points (x,y) to aggregate
            debug: Whether to save debug images
            task_id: Optional task ID for directory organization
            
        Returns:
            tuple: (best_point, response) with the selected best point and model's reasoning
        """
        if not points:
            return None, "No points to aggregate"
        
        if len(points) == 1:
            # If only one point, return it directly
            return points[0], "Only one point available, selected automatically."
        
        # Create a copy of the image for visualization
        if isinstance(image, str):
            vis_image = Image.open(image).copy()
        else:
            vis_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()
        
        # Create visualization with numbered stars for each point
        labels = [str(i+1) for i in range(len(points))]
        aggregated_image = plot_points_on_image(
            vis_image,
            points,
            colors=[(255, 0, 255, 128) for _ in range(len(points))],
            markers=['star' for _ in range(len(points))],
            sizes=[8 for _ in range(len(points))],
            labels=labels
        )
        
        if debug:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            aggregated_image.save(os.path.join(debug_dir, "RegionFocus_aggregated.png"))
        
        # Convert to base64 for the model
        aggregated_buffer = io.BytesIO()
        aggregated_image.save(aggregated_buffer, format="PNG")
        encoded_string = base64.b64encode(aggregated_buffer.getvalue()).decode("utf-8")
        
        # Create selection prompt
        selection_prompt = (
            f'In the image, I\'ve identified {len(points)} potential points (numbered 1-{len(points)}) '
            f'that might match the instruction: "{instruction}". '
            f'Carefully analyze each point and select the ONE that best matches the instruction. '
            f'Sometimes, multiple points may overlap, and you need to select one from the overlapping area. Additionally, the correct point might sometimes cover the target, and you need to distinguish this scenario.'
            f'Provide your final answer in this format: '
            f'"Selected point: #" where # is the number of the best point.'
        )
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "min_pixels": 3136, "max_pixels": 12845056, "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                    {"type": "text", "text": selection_prompt}
                ],
            }
        ]
        
        # Call the endpoint
        response = self._call_endpoint(messages)
        if debug:
            with open(os.path.join(debug_dir, "aggregation_response.txt"), "w") as f:
                f.write(f"Instruction: {instruction}\n\n")
                f.write(f"Response:\n{response}")
        
        # Extract the selected point number
        pattern = r"Selected point: (\d+)"
        match = re.search(pattern, response)
        
        if match:
            selected_idx = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= selected_idx < len(points):
                selected_point = points[selected_idx]
                
                if debug:
                    # Create final visualization with only the selected point
                    final_image = plot_points_on_image(
                        vis_image,
                        [selected_point],
                        colors=[(0, 255, 0)],  # Green for final selection
                        markers=['star'],
                        sizes=[20]
                    )
                    final_image.save(os.path.join(debug_dir, "RegionFocus_final.png"))
                
                return selected_point, response
        
        # If no valid selection found, return the first point as fallback
        return points[0], response + "\n(No valid selection found, using first point as fallback.)"

    def ground_with_regionfocus(self, instruction, image, debug=False, task_id=None):
        """
        Main method that performs initial grounding, then applies RegionFocus if needed.
        
        Args:
            instruction: The instruction text
            image: Image to process (PIL Image or path to image file)
            debug: Whether to save debug images
            task_id: Optional task ID for directory organization
            
        Returns:
            dict: Result dictionary with point, bbox (if available), and other metadata
        """
        # Create debug directory if needed
        if debug:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
        
        # Step 1: Initial grounding
        initial_result, display_image, system_message = self.ground(instruction, image)
        
        original_image = display_image
        
        # Make a copy for debug visualization
        if debug:
            viz_image = original_image.copy()
            if initial_result["point"]:
                viz_image = plot_points_on_image(
                    viz_image,
                    [[round(initial_result["point"][0]*viz_image.width), round(initial_result["point"][1]*viz_image.height)]],
                    colors=[(255, 105, 180, 128)],  
                    markers=['star'],
                    sizes=[8]
                )
                viz_image.save(os.path.join(debug_dir, "initial_grounding.png"))
        
        # Step 2: Judge the initial grounding
        if initial_result["point"]:
            is_correct, judge_response = self.judge_inference(
                instruction, 
                original_image, 
                [round(initial_result["point"][0]*original_image.width), round(initial_result["point"][1]*original_image.height)],
                debug=debug,
                task_id=task_id,
                system_message=system_message
            )
            
            if debug:
                print(f"Initial grounding judgment: {'CORRECT' if is_correct else 'INCORRECT'}")
                print(f"Judgment response: {judge_response}")
            
            # If the initial grounding is correct, return it
            if is_correct:
                if debug:
                    print("Using initial grounding result as it was judged correct.")
                print(f"Judgment response: {judge_response}")
                return initial_result
        else:
            is_correct = False
            judge_response = "No valid point found in initial grounding."
            if debug:
                print("Initial grounding failed to find a valid point.")
        
        # Step 3: Apply RegionFocus to get a better point
        region_points = []
        region_responses = []
        print(f"Judgment response: {judge_response}")
        
        temperatures = [0.0, 0.3, 0.5, 0.7, 0.9]
        
        for i, temp in enumerate(temperatures):
            point, response = self.auto_focus(
                instruction, 
                original_image, 
                debug=debug, 
                task_id=task_id, 
                temperature=temp,
                top_p=0.90,
                system_message=system_message
            )
            
            if point:
                region_points.append(point)
                region_responses.append(response)
                if debug:
                    print(f"Region focus {i+1} found point: {point}")
                break # only one point is good
        
        if not region_points:
            if debug:
                print("RegionFocus failed to find any valid points.")
            # Return original result if no better option found
            return initial_result
        
        # Step 4: For each identified point, perform crop and zoom
        zoomed_results = []
        
        ratio_list = [[0.5, 0.5], [0.3, 0.3], [0.4, 0.8], [0.8, 0.4]]
        point = region_points[0]
        for i, ratio in enumerate(ratio_list):
            # Calculate the crop region
            left, top, w, h = self.calculate_crop_region(
                [round(point[0]*original_image.width), round(point[1]*original_image.height)],
                original_image,
                debug=debug,
                task_id=task_id,
                index=i,
                ratio_x=ratio[0],
                ratio_y=ratio[1]
            )
            
            # Crop and upsample the region
            zoomed_bytes, zoom_x, zoom_y, offset_w, offset_h = self.crop_and_upsample(
                (left, top, w, h),
                original_image,
                keep_aspect_ratio=True,
                debug=debug,
                task_id=task_id,
                index=i
            )
            
            # Step 5: Predict action on the zoomed region
            action_point, action_response = self.next_action_regionfocus(
                instruction,
                zoomed_bytes,
                left, top, zoom_x, zoom_y,
                offset_w, offset_h, w, h,
                original_image,
                debug=debug,
                task_id=task_id,
                index=i,
                temperature=0.0,
                top_p=1.0,
                system_message=system_message
            )
            
            if action_point:
                zoomed_results.append((action_point, action_response))
                if debug:
                    print(f"RegionFocus {i+1} action found point: {action_point}")
        
        if not zoomed_results:
            if debug:
                print("No valid points found from zoomed regions.")
            # Return the best initial point if available, otherwise the first region point
            return initial_result if initial_result["point"] else {"point": region_points[0], "bbox": None, "raw_response": 'no valid points found from zoomed regions'}
        
        # Step 6: Aggregate results if we have multiple zoomed predictions
        if len(zoomed_results) > 0:
            final_points = [p for p, _ in zoomed_results]
            best_point, agg_response = self.next_action_regionfocus_aggregation(
                instruction,
                original_image,
                final_points,
                debug=debug,
                task_id=task_id,
                system_message=system_message
            )
            print(f"Aggregated result: {best_point}")
            print(f"Aggregated response: {agg_response}")
            
            if debug:
                print(f"Aggregated result: {best_point}")
                # Save the final selected point
                final_viz = original_image.copy()
                final_viz = plot_points_on_image(
                    final_viz,
                    [best_point],
                    colors=[(0, 255, 0)],  # Green for final selection
                    markers=['star'],
                    sizes=[20]
                )
                final_viz.save(os.path.join(debug_dir, "regionfocus_final_selection.png"))
        else:
            # If only one result, use it directly
            best_point, agg_response = zoomed_results[0]
        
        # Step 7: Create the final result
        final_result = {
            "point": [best_point[0]/original_image.width, best_point[1]/original_image.height],
            "bbox": None,  # We don't have bbox info
            "regionfocus_applied": True,
            "initial_point": initial_result["point"],
            "initial_correct": is_correct,
            "num_candidates": len(zoomed_results),
            'raw_response': agg_response
        }

        return final_result
