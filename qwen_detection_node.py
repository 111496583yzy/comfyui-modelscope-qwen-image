import os
import ast
import json
import base64
import io
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

try:
    import folder_paths
except ImportError:
    # folder_paths 只在 ComfyUI 环境中可用
    folder_paths = None


def parse_json(json_output: str) -> str:
    """Extract the JSON payload from a model response string."""
    if "```json" in json_output:
        json_output = json_output.split("```json", 1)[1]
        json_output = json_output.split("```", 1)[0]

    try:
        parsed = json.loads(json_output)
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                json_output = inner
    except Exception:
        pass
    return json_output


def draw_bboxes_on_image(image, bboxes_data, target_label="object"):
    """直接使用KJNodes的DrawInstanceDiffusionTracking实现"""
    import matplotlib.cm as cm
    import torch
    from torchvision import transforms
    
    if not bboxes_data:
        return image
    
    # 确保图像是PIL格式
    if hasattr(image, 'mode'):  # 已经是PIL图像
        pil_image = image.copy()
    else:  # 如果是tensor，转换为PIL
        if len(image.shape) == 4:  # batch
            current_image = image[0, :, :, :].permute(2, 0, 1)
        else:  # single image
            current_image = image.permute(2, 0, 1)
        pil_image = transforms.ToPILImage()(current_image)
    
    draw = ImageDraw.Draw(pil_image)
    
    # 使用KJNodes的彩虹色彩映射
    colormap = cm.get_cmap('rainbow', len(bboxes_data))
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 直接使用KJNodes的绘制逻辑
    for j, bbox_data in enumerate(bboxes_data):
        if isinstance(bbox_data, dict):
            bbox = bbox_data.get("bbox_2d", bbox_data.get("bbox", []))
            label = bbox_data.get("label", target_label)
        else:
            bbox = bbox_data
            label = target_label
            
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = bbox
        # 转换为整数（KJNodes的做法）
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 使用KJNodes的颜色生成方式
        color = tuple(int(255 * x) for x in colormap(j / len(bboxes_data)))[:3]
        
        # 添加调试信息
        print(f"绘制边界框 {j+1}: ({x1}, {y1}) -> ({x2}, {y2}), 标签: {label}")
        
        # 使用KJNodes的绘制方式
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        if font:
            # 使用KJNodes的文本绘制方式
            text = f"{j+1}.{label}"
            # 计算文本尺寸（KJNodes的方式）
            _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
            # 使用KJNodes的文本位置
            text_position = (x1, y1 - text_height)
            draw.text(text_position, text, fill=color, font=font)
    
    return pil_image


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return bounding boxes parsed from the model's raw JSON output."""
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            end_idx = text.rfind('"}') + len('"}')
            truncated = text[:end_idx] + "]"
            data = ast.literal_eval(truncated)
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    items: List[DetectedBox] = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        score = float(item.get("score", 1.0))
        # 修复坐标顺序：确保是 [x1, y1, x2, y2] 格式
        if len(box) >= 4:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
            
        abs_x1 = int(x1 * x_scale)
        abs_y1 = int(y1 * y_scale)
        abs_x2 = int(x2 * x_scale)
        abs_y2 = int(y2 * y_scale)
        
        # 确保坐标顺序正确
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
            
        if score >= score_threshold:
            items.append(DetectedBox([abs_x1, abs_y1, abs_x2, abs_y2], score, label))
    items.sort(key=lambda x: x.score, reverse=True)
    return [
        {"score": b.score, "bbox": b.bbox, "label": b.label}
        for b in items
    ]


@dataclass
class DetectedBox:
    bbox: List[int]
    score: float
    label: str = ""


@dataclass
class QwenAPIConfig:
    client: Any
    model_name: str
    base_url: str
    api_key: str


def encode_image_to_base64(image: Image.Image) -> str:
    """将PIL图像编码为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


class QwenAPIConfig:
    def __init__(self, client=None, model_name=None, base_url=None, api_key=None):
        self.client = client
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "https://api-inference.modelscope.cn/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": "Qwen/Qwen2.5-VL-72B-Instruct"}),
                "timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
            }
        }

    RETURN_TYPES = ("QWEN_API_CONFIG",)
    RETURN_NAMES = ("qwen_api_config",)
    FUNCTION = "configure"
    CATEGORY = "Qwen2.5-VL"

    def configure(self, base_url: str, api_key: str, model_name: str, timeout: int):
        """配置API客户端"""
        if not api_key:
            raise ValueError("API密钥不能为空")
        
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        
        return (QwenAPIConfig(
            client=client,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key
        ),)


class QwenAPIDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_api_config": ("QWEN_API_CONFIG",),
                "image": ("IMAGE",),
                "target": ("STRING", {"default": "object"}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX", "IMAGE")
    RETURN_NAMES = ("text", "bboxes", "preview")
    FUNCTION = "detect"
    CATEGORY = "Qwen2.5-VL"

    def detect(
        self,
        qwen_api_config: QwenAPIConfig,
        image,
        target: str,
        bbox_selection: str = "all",
        score_threshold: float = 0.0,
        merge_boxes: bool = False,
        seed: int = 0,
    ):
        """使用API生成目标检测边界框"""
        client = qwen_api_config.client
        model_name = qwen_api_config.model_name
        
        # 添加随机种子到prompt中，确保每次请求都不同
        random_seed_text = f"Random seed: {seed}. "
        
        prompt = f"""You are a precise object detection system. Your task is to detect {target} in the image.

{random_seed_text}STRICT REQUIREMENTS:
1. You MUST create a SEPARATE bounding box for EACH individual {target} you see
2. NEVER combine multiple {target} objects into one large bounding box
3. Each {target} should have its own tight, precise bounding box
4. If you see 6 {target} objects, you MUST return 6 separate bounding boxes
5. Count each {target} individually and create one box per object

OUTPUT FORMAT: Return a JSON array with separate objects for each {target}:
[{{"bbox_2d": [x1, y1, x2, y2], "label": "{target}"}}, {{"bbox_2d": [x1, y1, x2, y2], "label": "{target}"}}, ...]

EXAMPLE: If you see 3 logos, return 3 separate boxes:
[{{"bbox_2d": [100, 100, 150, 120], "label": "{target}"}}, {{"bbox_2d": [200, 100, 250, 120], "label": "{target}"}}, {{"bbox_2d": [300, 100, 350, 120], "label": "{target}"}}]

DO NOT return one large box covering multiple objects!"""

        # 处理图像输入
        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("不支持的图像类型")

        # 编码图像为base64
        image_data = encode_image_to_base64(image)
        
        try:
            # 调用API
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_data}}
                    ]
                }],
                max_tokens=1024,
                temperature=0.1
            )
            
            output_text = response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"API调用失败: {str(e)}")

        # 解析响应
        # 使用固定的输入尺寸，因为API返回的坐标通常是相对坐标
        input_h = 1024  # 假设的输入高度
        input_w = 1024  # 假设的输入宽度
        
        items = parse_boxes(
            output_text,
            image.width,
            image.height,
            input_w,
            input_h,
            score_threshold,
        )

        
        # 处理边界框选择
        selection = bbox_selection.strip().lower()
        boxes = items
        if selection != "all" and selection:
            idxs = []
            for part in selection.replace(",", " ").split():
                try:
                    idxs.append(int(part))
                except Exception:
                    continue
            boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

        # 合并边界框
        if merge_boxes and boxes:
            x1 = min(b["bbox"][0] for b in boxes)
            y1 = min(b["bbox"][1] for b in boxes)
            x2 = max(b["bbox"][2] for b in boxes)
            y2 = max(b["bbox"][3] for b in boxes)
            score = max(b["score"] for b in boxes)
            label = boxes[0].get("label", target)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        # 格式化输出
        json_boxes = [
            {"bbox_2d": b["bbox"], "label": b.get("label", target)} for b in boxes
        ]
        json_output = json.dumps(json_boxes, ensure_ascii=False)
        bboxes_only = [b["bbox"] for b in boxes]
        
        # 生成预览图像
        preview_image = self._create_preview_image(image, json_boxes, target)
        
        return (json_output, bboxes_only, preview_image)
    
    def _create_preview_image(self, image, bboxes_data, target_label):
        """创建带边界框的预览图像"""
        try:
            # 转换tensor为PIL图像
            if isinstance(image, torch.Tensor):
                print(f"原始图像tensor形状: {image.shape}")
                
                # 假设图像是 [batch, channels, height, width] 格式 (ComfyUI标准)
                if image.dim() == 4:
                    image_tensor = image[0]  # 取第一个batch，现在是 [C, H, W]
                else:
                    image_tensor = image
                
                print(f"处理后图像tensor形状: {image_tensor.shape}")
                
                # 转换为numpy然后PIL
                image_np = image_tensor.cpu().numpy()
                print(f"转换为numpy后形状: {image_np.shape}")
                
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype('uint8')
                else:
                    image_np = image_np.astype('uint8')
                
                # 从CHW转换为HWC格式
                if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3, 4]:
                    image_np = np.transpose(image_np, (1, 2, 0))
                    print(f"转置后形状: {image_np.shape}")
                
                # 确保图像是RGB格式（3通道）
                if len(image_np.shape) == 3 and image_np.shape[2] == 1:
                    # 如果是单通道，转换为RGB
                    image_np = np.repeat(image_np, 3, axis=2)
                    print(f"单通道转RGB后形状: {image_np.shape}")
                elif len(image_np.shape) == 2:
                    # 如果是灰度图，转换为RGB
                    image_np = np.stack([image_np] * 3, axis=2)
                    print(f"灰度图转RGB后形状: {image_np.shape}")
                
                pil_image = Image.fromarray(image_np, 'RGB')
            else:
                pil_image = image
            
            # 绘制边界框
            preview_image = draw_bboxes_on_image(pil_image, bboxes_data, target_label)
            
            # 转换回tensor格式 - ComfyUI的IMAGE类型期望(B, H, W, C)格式
            preview_np = np.array(preview_image)
            print(f"绘制边界框后numpy形状: {preview_np.shape}")
            
            # 确保是RGB格式
            if len(preview_np.shape) == 3 and preview_np.shape[2] == 3:
                # 已经是HWC格式，直接转换
                preview_tensor = torch.from_numpy(preview_np).float() / 255.0
                print(f"转换为tensor后形状: {preview_tensor.shape}")
                
                # 添加batch维度 - ComfyUI的IMAGE类型期望(B, H, W, C)格式
                preview_tensor = preview_tensor.unsqueeze(0)  # HWC -> BHWC
                print(f"添加batch维度后形状: {preview_tensor.shape}")
                
                # 确保数据类型和范围正确
                preview_tensor = torch.clamp(preview_tensor, 0.0, 1.0)
                print(f"最终tensor形状: {preview_tensor.shape}, 数据类型: {preview_tensor.dtype}")
                
                # 确保tensor是连续的，这对ComfyUI很重要
                preview_tensor = preview_tensor.contiguous()
                
                return preview_tensor
            else:
                print(f"警告: 预览图像格式不正确: {preview_np.shape}")
                return image
            
        except Exception as e:
            print(f"创建预览图像时出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果出错，返回原始图像
            return image
    


class BBoxesToSAM2:
    """Convert a list of bounding boxes to the format expected by SAM2 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bboxes": ("BBOX",)}}

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("sam2_bboxes",)
    FUNCTION = "convert"
    CATEGORY = "Qwen2.5-VL"

    def convert(self, bboxes):
        if not isinstance(bboxes, list):
            raise ValueError("bboxes must be a list")

        # If already batched, return as-is
        if bboxes and isinstance(bboxes[0], (list, tuple)) and bboxes[0] and isinstance(bboxes[0][0], (list, tuple)):
            return (bboxes,)

        return ([bboxes],)


NODE_CLASS_MAPPINGS = {
    "QwenAPIConfig": QwenAPIConfig,
    "QwenAPIDetection": QwenAPIDetection,
    "BBoxesToSAM2": BBoxesToSAM2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenAPIConfig": "Qwen2.5-VL API Configuration",
    "QwenAPIDetection": "Qwen2.5-VL API Object Detection",
    "BBoxesToSAM2": "Prepare BBoxes for SAM2",
}
