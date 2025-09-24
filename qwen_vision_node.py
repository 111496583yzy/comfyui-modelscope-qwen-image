import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import tempfile

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装openai库，图生文功能将不可用")
    print("请运行: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "default_model": "stepfun-ai/step3",
            "timeout": 60,
            "default_prompt": "描述这幅图"
        }

def load_api_token():
    token_path = os.path.join(os.path.dirname(__file__), '.qwen_token')
    try:
        cfg = load_config()
        token_from_cfg = cfg.get("api_token", "").strip()
        if token_from_cfg:
            return token_from_cfg
    except Exception as e:
        print(f"读取config.json中的token失败: {e}")
    try:
        if os.path.exists(token_path):
            with open(token_path, 'r', encoding='utf-8') as f:
                token = f.read().strip()
                return token if token else ""
        return ""
    except Exception as e:
        print(f"加载token失败: {e}")
        return ""

def save_api_token(token):
    token_path = os.path.join(os.path.dirname(__file__), '.qwen_token')
    try:
        with open(token_path, 'w', encoding='utf-8') as f:
            f.write(token)
        cfg = load_config()
        cfg["api_token"] = token
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存token失败: {e}")
        return False

def tensor_to_base64_url(image_tensor):
    try:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        
        if image_tensor.max() <= 1.0:
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image_tensor.cpu().numpy().astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        print(f"图像转换失败: {e}")
        raise Exception(f"图像格式转换失败: {str(e)}")

class QwenVisionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        if not OPENAI_AVAILABLE:
            return {
                "required": {
                    "error_message": ("STRING", {
                        "default": "请先安装openai库: pip install openai",
                        "multiline": True
                    }),
                }
            }
        config = load_config()
        saved_token = load_api_token()
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "描述这幅图")
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "请输入您的魔搭API Token"
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": config.get("default_vision_model", "stepfun-ai/step3")
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 4000
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": config.get("default_vision_seed", -1),
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze_image"
    CATEGORY = "QwenImage"

    def analyze_image(self, image=None, prompt="", api_token="", model="stepfun-ai/step3", max_tokens=1000, temperature=0.7, seed=-1, error_message=""):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)
        
        config = load_config()
        
        if not api_token or api_token.strip() == "":
            raise Exception("请输入有效的API Token")
        
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("✅ API Token已自动保存")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        try:
            print(f"🔍 开始分析图像...")
            print(f"📝 提示词: {prompt}")
            print(f"🤖 模型: {model}")
            
            # 处理随机种子
            if seed != -1:
                print(f"🎯 使用指定种子: {seed}")
            else:
                import random
                random_seed = random.randint(0, 2147483647)
                print(f"🎲 使用随机种子: {random_seed}")
                seed = random_seed
            
            image_url = tensor_to_base64_url(image)
            print(f"🖼️ 图像已转换为base64格式")
            
            client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1',
                api_key=api_token
            )
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                }],
            }]
            
            print(f"🚀 发送API请求...")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                seed=seed
            )
            
            description = response.choices[0].message.content
            print(f"✅ 分析完成!")
            print(f"📄 结果: {description[:100]}...")
            
            return (description,)
            
        except Exception as e:
            error_msg = f"图像分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "QwenVisionNode": QwenVisionNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenVisionNode": "Qwen-Vision 图生文节点"
    }
else:
    class OpenAINotInstalledNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "install_command": ("STRING", {
                        "default": "pip install openai",
                        "multiline": False
                    }),
                }
            }
        
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("message",)
        FUNCTION = "show_install_message"
        CATEGORY = "QwenImage"
        
        def show_install_message(self, install_command):
            return ("请先安装openai库才能使用图生文功能: " + install_command,)
    
    NODE_CLASS_MAPPINGS = {
        "QwenVisionNode": OpenAINotInstalledNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenVisionNode": "Qwen-Vision 图生文节点 (需要安装openai)"
    }