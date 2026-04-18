import base64
import io
import json
import math
import os
import re
import tempfile
import time

import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw

from .prob_map2zoom_box import prob_map_to_multi_crops, prob_map_to_zoom_box_squ

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from qwen_utils_agent_function_call import ComputerUse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


def build_elliptical_prob_map(samples, img_shape, sigma_scale=10.0):
    H, W = img_shape
    prob_map = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]

    for s in samples:
        x_i, y_i = s['x'], s['y']
        ppl_x, ppl_y = s['ppl_x'], s['ppl_y']
        sig_x = max(5, ppl_x * sigma_scale)
        sig_y = max(5, ppl_y * sigma_scale)
        weight = 1.0 / (ppl_x * ppl_y + 1e-6)
        term_x = ((xx - x_i) ** 2) / (2 * sig_x ** 2)
        term_y = ((yy - y_i) ** 2) / (2 * sig_y ** 2)

        kernel = np.exp(-(term_x + term_y))
        prob_map += weight * kernel

    if np.max(prob_map) > 0:
        prob_map /= np.max(prob_map)

    return prob_map


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
        colors = [(255, 0, 255) for _ in range(len(points))]
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
        
        if marker == 'star':
            star_points = []
            for j in range(5):
                angle_outer = math.pi / 2 + j * 2 * math.pi / 5
                px_outer = x + size * math.cos(angle_outer)
                py_outer = y + size * math.sin(angle_outer)
                star_points.append((px_outer, py_outer))
                
                angle_inner = math.pi / 2 + (j + 0.5) * 2 * math.pi / 5
                px_inner = x + size / 2 * math.cos(angle_inner)
                py_inner = y + size / 2 * math.sin(angle_inner)
                star_points.append((px_inner, py_inner))
            
            draw.polygon(star_points, fill=color)
            
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


class Qwen25VLModel:
    def __init__(
        self,
        base_url="http://localhost:8400/v1",
        api_key="empty",
        model_name="Qwen/Qwen2.5-VL-72B-Instruct",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.regionfocus_coords = []
        self.generation_config = {}

    def load_model(self):
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
        timeout = 100
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    max_tokens=1024,
                    logprobs=True,
                    top_logprobs=3,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff time (1s, 2s, 4s, etc.)
                    wait_time = 3 ** attempt
                    print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error calling API after {max_retries} attempts: {e}")
                    return "Error: Unable to get a response from the model after multiple attempts."

    def _call_endpoint_ori_ouput(self, messages, temperature=0, top_p=1.0):
        max_retries = 2
        timeout = 10

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    max_tokens=1024,
                    logprobs=True,  # 开启 logprobs
                    top_logprobs=3
                    # **self.generation_config
                )
                print_log_N_tokens = False
                if print_log_N_tokens:
                    if response.choices[0].logprobs:
                        content_logprobs = response.choices[0].logprobs.content
                        print(f"生成的 Token 总数: {len(content_logprobs)}\n")

                        # Print the probability details of the first 3 generated tokens.
                        for i, token_data in enumerate(content_logprobs[:30]):
                            print(f"--- 位置 {i + 1}: 选中的 Token: '{token_data.token}' ---")
                            print(f"  Logprob: {token_data.logprob:.4f}")
                            prob = math.exp(token_data.logprob) * 100
                            print(f"  实际概率: {prob:.2f}%")
                            print("  候选 Token (Top 3):")
                            for top in token_data.top_logprobs:
                                top_prob = math.exp(top.logprob) * 100
                                print(f"    - '{top.token}': {top_prob:.2f}%")
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff time (1s, 2s, 4s, etc.)
                    wait_time = 2 ** attempt
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error calling API after {max_retries} attempts: {e}")
                    return "Error: Unable to get a response from the model after multiple attempts."

    def get_token_probs_between_strings(self, response, start_str, end_str):
        """
        从 API 响应中查找位于两个指定字符串之间的所有 token 及其概率

        Args:
            response: OpenAI ChatCompletion 响应对象
            start_str: 起始标记字符串 (如 "结果是：")
            end_str: 结束标记字符串 (如 "。总结")

        Returns:
            tuple: (List[Dict], str) 包含所有中间 token 信息的列表和错误信息
        """
        if not response.choices[0].logprobs:
            return [], "Logprobs not enabled in response."

        logprobs_list = response.choices[0].logprobs.content

        # 1. 重建完整文本并创建 [字符位置 -> Token] 的映射
        full_text = ""
        token_map = []  # 存储结构: {'start': int, 'end': int, 'data': object}

        current_idx = 0
        for item in logprobs_list:
            token_str = item.token
            start = current_idx
            end = current_idx + len(token_str)

            token_map.append({"start": start, "end": end, "data": item})

            full_text += token_str
            current_idx = end

        # 2. 定位起始和结束边界

        # 边界 A: 起始字符串的【结束】位置 (Token 必须在此之后开始)
        start_str_loc = full_text.find(start_str)
        if start_str_loc == -1:
            return [], f"未找到起始字符串: '{start_str}'"
        start_boundary_idx = start_str_loc + len(start_str)

        end_str_loc = full_text.find(end_str, start_boundary_idx)
        if end_str_loc == -1:
            return [], f"未找到结束字符串: '{end_str}'"
        end_boundary_idx = end_str_loc

        if start_boundary_idx >= end_boundary_idx:
            return [], "起始字符串紧接在结束字符串之前，或两者重叠/顺序错误，中间没有 Token。"

        # 3. 筛选位于边界之间的 Token
        tokens_in_between = []
        tokens_in_between_content = ''
        for t_map in token_map:
            token_start = t_map['start']
            token_data = t_map['data']

            if token_start >= start_boundary_idx and token_start < end_boundary_idx:
                prob_percent = math.exp(token_data.logprob) * 100
                tokens_in_between_content += token_data.token
                tokens_in_between.append({
                    "token": token_data.token,
                    "logprob": token_data.logprob,
                    "probability": prob_percent,
                    "top_logprobs": token_data.top_logprobs[:3],
                })

            if token_start >= end_boundary_idx:
                break

        return tokens_in_between, None, tokens_in_between_content

    def calculate_perplexity(self, token_data_list):
        """
        基于 Token 数据列表 (包含 logprob) 计算 Perplexity。

        Args:
            token_data_list: 由 get_token_probs_between_strings 返回的 List[Dict]，
                             其中每个字典必须包含 'logprob' 键。

        Returns:
            float or None: 计算出的 Perplexity 值，如果列表为空则返回 None。
        """
        if not token_data_list:
            print("警告: Token 列表为空，无法计算 Perplexity。")
            return None

        logprobs = [data['logprob'] for data in token_data_list]
        sum_logprobs = np.sum(logprobs)
        N = len(logprobs)
        ANLL = -sum_logprobs / N
        perplexity = math.exp(ANLL)

        return perplexity

    def ground(self, instruction, image):
        if not isinstance(image, str):
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
                    {
                        "type": "image_url",
                        "min_pixels": 3136,
                        "max_pixels": 12845056,
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Output the most relevant point in the image corresponding to "
                            f'the instruction "{instruction}" with grounding.'
                        ),
                    },
                ],
            }
        ]

        response_ori = self._call_endpoint_ori_ouput(messages)

        response = response_ori.choices[0].message.content
        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n')[0])
            coordinates = action['arguments']['coordinate']
            results, error, tokens_in_between_content = self.get_token_probs_between_strings(response_ori, '[', ']')
            results_x, _, _ = self.get_token_probs_between_strings(response_ori, '[', ',')
            results_y, _, _ = self.get_token_probs_between_strings(response_ori, ',', ']')
            if not error and results:
                ppl_value = self.calculate_perplexity(results)
                ppl_value_x = self.calculate_perplexity(results_x)
                ppl_value_y = self.calculate_perplexity(results_y)
            result_dict = {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "perplexity": ppl_value,
                "perplexity_x": ppl_value_x,
                "perplexity_y": ppl_value_y,
                "perplexity_content": tokens_in_between_content,
                "point": [coordinates[0] / resized_width, coordinates[1] / resized_height],
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

    def ground_sample_points(
        self, instruction, image, sample_number=5, sample_t=0.75, max_sample_number=10
    ):
        if not isinstance(image, str):
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

        system_message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
            ],
            functions=[computer_use.function],
            lang=None,
        )
        system_message = system_message[0].model_dump()
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
                    {
                        "type": "image_url",
                        "min_pixels": 3136,
                        "max_pixels": 12845056,
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Output the most relevant point in the image corresponding to "
                            f'the instruction "{instruction}" with grounding.'
                        ),
                    },
                ],
            }
        ]

        response_list = []
        for number in range(max_sample_number):
            response_ori = self._call_endpoint_ori_ouput(messages, temperature=sample_t)
            response = response_ori.choices[0].message.content
            try:
                if 'point_2d":' in response:
                    coordinates = json.loads(response.split('point_2d":')[1].split(', "label"')[0])
                else:
                    action = json.loads(response.split('<tool_call>\n')[1].split('\n')[0])
                    coordinates = action['arguments']['coordinate']
                results, error, tokens_in_between_content = self.get_token_probs_between_strings(response_ori, '[', ']')
                results_x, _, _ = self.get_token_probs_between_strings(response_ori, '[', ',')
                results_y, _, _ = self.get_token_probs_between_strings(response_ori, ',', ']')
                if not error and results:
                    ppl_value = self.calculate_perplexity(results)
                    ppl_value_x = self.calculate_perplexity(results_x)
                    ppl_value_y = self.calculate_perplexity(results_y)
                result_dict = {
                    "ppl_x": ppl_value_x,
                    "ppl_y": ppl_value_y,
                    "x": coordinates[0],
                    "y": coordinates[1],
                }
                response_list.append(result_dict)
            except:
                result_dict = {
                    "bbox": None,
                    "point": None
                }
                print('error')
                print(response)
            if len(response_list) == sample_number:
                break

        return response_list

    def judge_inference(self, instruction, image, point, debug=False, task_id=None, system_message=None):
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
        
        response_ori = self._call_endpoint_ori_ouput(messages)
        response = response_ori.choices[0].message.content
        is_correct = ("YES" in response.upper() or "CORRECT" in response.upper() or "正确" in response or "精准" in response) and not ("NO" in response.upper() or "INCORRECT" in response.upper() or "不正确" in response or "不精准" in response)
        if is_correct:
            for i in range(len(response_ori.choices[0].logprobs.content)):
                if response_ori.choices[0].logprobs.content[i].token.upper() in ['YES', 'CORRECT', '正确', '精准']:
                    log_prob = response_ori.choices[0].logprobs.content[i].logprob
                    break
                else:
                    log_prob = 2
        else:
            for i in range(len(response_ori.choices[0].logprobs.content)):
                if response_ori.choices[0].logprobs.content[i].token.upper() in ['NO', 'INCORRECT', '不正确', '不精准']:
                    log_prob = response_ori.choices[0].logprobs.content[i].logprob
                    break
                else:
                    log_prob = 2
        prob_cw = np.exp(log_prob)

        if debug:
            with open(os.path.join(debug_dir, "judgment_response.txt"), "w") as f:
                f.write(f"Instruction: {instruction}\n\n")
                f.write(f"Point: {point}\n\n")
                f.write(f"Judgment: {'CORRECT' if is_correct else 'INCORRECT'}\n\n")
                f.write(f"Response:\n{response}")
        
        return is_correct, response, prob_cw

    def crop_and_upsample(self, bbox, image, debug=False, task_id=None, index=None, keep_aspect_ratio=True):
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        
        img_width, img_height = img.size

        left, top, w, h = bbox
        
        left = max(0, left)
        top = max(0, top)
        w = min(w, img_width - left)
        h = min(h, img_height - top)
        
        cropped = img.crop((left, top, left + w, top + h))
        
        if debug:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            crop_filename = f"crop_{index}.png" if index is not None else "crop.png"
            cropped.save(os.path.join(debug_dir, crop_filename))
        
        viewport_width = img_width
        viewport_height = img_height
        
        if not keep_aspect_ratio:
            upsampled = cropped.resize((viewport_width, viewport_height), Image.Resampling.LANCZOS)
            zoom_x = viewport_width / w
            zoom_y = viewport_height / h
            offset_w = 0
            offset_h = 0
        else:
            zoom_x = viewport_width / w
            zoom_y = viewport_height / h
            zoom_factor = min(zoom_x, zoom_y)
            
            new_w = round(w * zoom_factor)
            new_h = round(h * zoom_factor)
            upsampled = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            offset_w = float(viewport_width - new_w) / 2
            offset_h = float(viewport_height - new_h) / 2
            
            zoom_x = zoom_factor
            zoom_y = zoom_factor
        
        if debug:
            upsampled_filename = f"upsampled_{index}.png" if index is not None else "upsampled.png"
            upsampled.save(os.path.join(debug_dir, upsampled_filename))
        
        output_buffer = io.BytesIO()
        upsampled.save(output_buffer, format="PNG")
        screenshot_bytes = output_buffer.getvalue()
        
        return screenshot_bytes, zoom_x, zoom_y, offset_w, offset_h

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
        
        response_ori = self._call_endpoint_ori_ouput(messages, temperature=temperature, top_p=top_p)
        response = response_ori.choices[0].message.content

        try:
            action = json.loads(response.split('<tool_call>\n')[1].split('\n')[0])
            click_point = action['arguments']['coordinate']
            results, error, tokens_in_between_content = self.get_token_probs_between_strings(response_ori, '[', ']')
            if not error and results:
                ppl_value = self.calculate_perplexity(results)
        except:
            return None, response

        if click_point:
            x_upsampled, y_upsampled = click_point
            zoomed_img = Image.open(io.BytesIO(zoomed_img_bytes))
            print(f"zoomed_img: {zoomed_img.size}")
            zoomed_width, zoomed_height = zoomed_img.size

            x_upsampled = round(x_upsampled)
            y_upsampled = round(y_upsampled)
            
            rel_zoomed_x = x_upsampled
            rel_zoomed_y = y_upsampled
            
            zoomed_width = w * zoom_x
            zoomed_height = h * zoom_y
            
            if 0 <= rel_zoomed_x < zoomed_width and 0 <= rel_zoomed_y < zoomed_height:
                x_orig = left + (rel_zoomed_x / zoom_x)
                y_orig = top + (rel_zoomed_y / zoom_y)
            else:
                clamped_rel_x = max(0, min(zoomed_width - 1, rel_zoomed_x))
                clamped_rel_y = max(0, min(zoomed_height - 1, rel_zoomed_y))
                
                x_orig = left + (clamped_rel_x / zoom_x)
                y_orig = top + (clamped_rel_y / zoom_y)
            
            if isinstance(original_image, Image.Image):
                img_width, img_height = original_image.size
            else:
                img_height, img_width = original_image.shape[:2]
                
            x_orig = max(0, min(x_orig, img_width - 1))
            y_orig = max(0, min(y_orig, img_height - 1))
            
            projected_point = (round(x_orig), round(y_orig))
            
            if debug:
                if isinstance(original_image, str):
                    original_pil = Image.open(original_image).copy()
                elif isinstance(original_image, np.ndarray):
                    original_pil = Image.fromarray(original_image).copy()
                else:
                    original_pil = original_image.copy()
                    
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
            
            return projected_point, (response, ppl_value, tokens_in_between_content)
            
        return None, response

    def next_action_regionfocus_aggregation(
        self, instruction, image, points, debug=False, task_id=None, system_message=None
    ):
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
        
        aggregated_buffer = io.BytesIO()
        aggregated_image.save(aggregated_buffer, format="PNG")
        encoded_string = base64.b64encode(aggregated_buffer.getvalue()).decode("utf-8")
        
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
        
        response = self._call_endpoint(messages)
        if debug:
            with open(os.path.join(debug_dir, "aggregation_response.txt"), "w") as f:
                f.write(f"Instruction: {instruction}\n\n")
                f.write(f"Response:\n{response}")
        
        pattern = r"Selected point: (\d+)"
        match = re.search(pattern, response)
        
        if match:
            selected_idx = int(match.group(1)) - 1
            if 0 <= selected_idx < len(points):
                selected_point = points[selected_idx]
                
                if debug:
                    final_image = plot_points_on_image(
                        vis_image,
                        [selected_point],
                        colors=[(0, 255, 0)],
                        markers=['star'],
                        sizes=[20]
                    )
                    final_image.save(os.path.join(debug_dir, "RegionFocus_final.png"))
                
                return (selected_point, selected_idx), response
        
        return points[0], response + "\n(No valid selection found, using first point as fallback.)"

    def ground_with_AutoFocus(self, instruction, image, debug=False, task_id=None, ground_truth=None):
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
        
        # Step 1: Initial grounding
        initial_result, display_image, system_message = self.ground(instruction, image)
        original_image = display_image

        if initial_result["point"]:
            is_correct, judge_response, _prob_cw = self.judge_inference(
                instruction,
                original_image,
                [
                    round(initial_result["point"][0] * original_image.width),
                    round(initial_result["point"][1] * original_image.height),
                ],
                debug=debug,
                task_id=task_id,
                system_message=system_message
            )
            
            if debug:
                print(f"Initial grounding judgment: {'CORRECT' if is_correct else 'INCORRECT'}")
                print(f"Judgment response: {judge_response}")
            
            # If the initial grounding is correct, return it
            if is_correct:
            # if is_correct :
                if debug:
                    print("Using initial grounding result as it was judged correct.")
                print(f"Judgment response: {judge_response}")
                # initial_result.update({"self_judgement": is_correct})
                return initial_result
        else:
            is_correct = False
            judge_response = "No valid point found in initial grounding."
            prob_cw = -2
            if debug:
                print("Initial grounding failed to find a valid point.")
        t_list = [0.1, 0.3, 0.5, 0.75]
        sample_results = []
        for t in t_list:
            sample_result = self.ground_sample_points(instruction, image, sample_t=t)
            sample_results = sample_results + sample_result
            if len(sample_results) > 4:
                break
        print('Original image shape:', (display_image.height, display_image.width))
        if len(sample_results) > 0:
            img_shape = (display_image.height, display_image.width)
            prob_map = build_elliptical_prob_map(sample_results, img_shape)
        else:
            prob_map = None

        if prob_map is None:
            return initial_result

        padding_list = [0.5, 1, 2, 4, 8]
        zoom_in_bbox_list = prob_map_to_multi_crops(prob_map, pad_ratio=0.7)
        for ratio in padding_list:
            zoom_in_bbox = prob_map_to_zoom_box_squ(prob_map, padding=ratio)
            zoom_in_bbox_list.append(zoom_in_bbox)

        print('**************** Phrase1 *****************')
        print(f"Judgment response: {judge_response}")

        
        # Step 4: For each identified point, perform crop and zoom
        zoomed_results = []
        zoomed_results_ppl_value = []
        zoomed_results_tokens_in_between_content = []
        print('**************** Phrase3 *****************')
        for i, zoom_in_bbox in enumerate(zoom_in_bbox_list[:5]):
            left, top, w, h = zoom_in_bbox
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
            if isinstance(action_response, tuple):
                action_response, ppl_value, tokens_in_between_content = action_response
            if action_point:
                zoomed_results.append((action_point, action_response))
                zoomed_results_ppl_value.append(ppl_value)
                zoomed_results_tokens_in_between_content.append(tokens_in_between_content)
                if debug:
                    print(f"RegionFocus {i+1} action found point: {action_point}")
        
        if not zoomed_results:
            if debug:
                print("No valid points found from zoomed regions.")
            return initial_result if initial_result["point"] else {"point": [][0], "bbox": None, "raw_response": 'no valid points found from zoomed regions'}
        print('**************** Phrase4 *****************')
        # Step 6: Aggregate results as we have multiple zoomed predictions
        if len(zoomed_results) > 0:
            final_points = [p for p, _ in zoomed_results]
            try:
                best_point, agg_response = self.next_action_regionfocus_aggregation(
                    instruction,
                    original_image,
                    final_points,
                    debug=debug,
                    task_id=task_id,
                    system_message=system_message
                )
                if isinstance(best_point, tuple) and isinstance(best_point[0], tuple):
                    best_point, best_index = best_point
                else:
                    best_index = 0
                print(f"Aggregated result: {best_point}")
                print(f"Aggregated response: {agg_response}")

                if debug:
                    print(f"Aggregated result: {best_point}")
                    final_viz = original_image.copy()
                    final_viz = plot_points_on_image(
                        final_viz,
                        [best_point],
                        colors=[(0, 255, 0)],
                        markers=['star'],
                        sizes=[20]
                    )
                    final_viz.save(os.path.join(debug_dir, "regionfocus_final_selection.png"))
            except Exception:
                best_point, agg_response = zoomed_results[0]
                best_index = 0
        else:
            best_point, agg_response = zoomed_results[0]
            best_index = 0

        best_ppl = zoomed_results_ppl_value[best_index]
        best_ppl_content = zoomed_results_tokens_in_between_content[best_index]
        # Step 7: Create the final result
        final_result = {
            "point": [best_point[0] / original_image.width, best_point[1] / original_image.height],
            "bbox": None,
            "regionfocus_applied": True,
            "initial_point": initial_result["point"],
            "initial_correct": is_correct,
            "num_candidates": len(zoomed_results),
            'raw_response': agg_response
        }

        if best_ppl and best_ppl_content:
            final_result.update({"perplexity": best_ppl,
                "perplexity_content": best_ppl_content})

        return final_result
