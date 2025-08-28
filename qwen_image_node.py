import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import folder_paths
import base64
import tempfile

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "default_model": "Qwen/Qwen-Image",
            "timeout": 720,
            "image_download_timeout": 30,
            "default_prompt": "A beautiful landscape"
        }

def save_config(config: dict) -> bool:
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False

def save_api_token(token):
    token_path = os.path.join(os.path.dirname(__file__), '.qwen_token')
    try:
        with open(token_path, 'w', encoding='utf-8') as f:
            f.write(token)
    except Exception as e:
        print(f"保存token失败(.qwen_token): {e}")
    try:
        cfg = load_config()
        cfg["api_token"] = token
        if save_config(cfg):
            return True
        return False
    except Exception as e:
        print(f"保存token失败(config.json): {e}")
        return False

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

class QwenImageNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_token = load_api_token()
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "A beautiful landscape")
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "请输入您的魔搭API Token"
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": config.get("default_model", "Qwen/Qwen-Image")
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_negative_prompt", "")
                }),
                "width": ("INT", {
                    "default": config.get("default_width", 512),
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": config.get("default_height", 512),
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "seed": ("INT", {
                    "default": config.get("default_seed", -1),
                    "min": -1,
                    "max": 2147483647
                }),
                "steps": ("INT", {
                    "default": config.get("default_steps", 30),
                    "min": 1,
                    "max": 100
                }),
                "guidance": ("FLOAT", {
                    "default": config.get("default_guidance", 7.5),
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "QwenImage"
    
    def generate_image(self, prompt, api_token, model="Qwen/Qwen-Image", negative_prompt="", width=512, height=512, seed=-1, steps=30, guidance=7.5):
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
            url = 'https://api-inference.modelscope.cn/v1/images/generations'
            payload = {
                'model': model,
                'prompt': prompt,
                'size': f"{width}x{height}",
                'steps': steps,
                'guidance': guidance
            }
            if negative_prompt.strip():
                payload['negative_prompt'] = negative_prompt
                print(f"🚫 负向提示词: {negative_prompt}")
            if seed != -1:
                payload['seed'] = seed
                print(f"🎯 使用指定种子: {seed}")
            else:
                import random
                random_seed = random.randint(0, 2147483647)
                payload['seed'] = random_seed
                print(f"🎲 使用随机种子: {random_seed}")
            print(f"📐 图像尺寸: {width}x{height}")
            print(f"🔧 采样步数: {steps}")
            print(f"🎨 引导系数: {guidance}")
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json',
                'X-ModelScope-Async-Mode': 'true'
            }
            submission_response = requests.post(
                url, 
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), 
                headers=headers,
                timeout=config.get("timeout", 60)
            )
            if submission_response.status_code == 400:
                print("⚠️ 提交失败，尝试使用最小参数重试...")
                minimal_payload = {
                    'model': model,
                    'prompt': prompt
                }
                submission_response = requests.post(
                    url,
                    data=json.dumps(minimal_payload, ensure_ascii=False).encode('utf-8'),
                    headers=headers,
                    timeout=config.get("timeout", 60)
                )
            if submission_response.status_code != 200:
                raise Exception(f"API请求失败: {submission_response.status_code}, {submission_response.text}")
            submission_json = submission_response.json()
            image_url = None
            if 'task_id' in submission_json:
                task_id = submission_json['task_id']
                print(f"🕒 已提交任务，任务ID: {task_id}，开始轮询...")
                poll_start = time.time()
                max_wait_seconds = max(60, config.get('timeout', 720))
                while True:
                    task_resp = requests.get(
                        f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                        headers={
                            'Authorization': f'Bearer {api_token}',
                            'X-ModelScope-Task-Type': 'image_generation'
                        },
                        timeout=config.get("image_download_timeout", 120)
                    )
                    if task_resp.status_code != 200:
                        raise Exception(f"任务查询失败: {task_resp.status_code}, {task_resp.text}")
                    task_data = task_resp.json()
                    status = task_data.get('task_status')
                    if status == 'SUCCEED':
                        output_images = task_data.get('output_images') or []
                        if not output_images:
                            raise Exception("任务成功但未返回图片URL")
                        image_url = output_images[0]
                        print("✅ 任务完成，开始下载图片...")
                        break
                    if status == 'FAILED':
                        raise Exception(f"任务失败: {task_data}")
                    if time.time() - poll_start > max_wait_seconds:
                        raise Exception("任务轮询超时，请稍后重试或降低并发")
                    time.sleep(5)
            elif 'images' in submission_json and len(submission_json['images']) > 0:
                image_url = submission_json['images'][0]['url']
                print(f"⬇️ 下载生成的图片...")
            else:
                raise Exception(f"未识别的API返回格式: {submission_json}")
            img_response = requests.get(image_url, timeout=config.get("image_download_timeout", 30))
            if img_response.status_code != 200:
                raise Exception(f"图片下载失败: {img_response.status_code}")
            pil_image = Image.open(BytesIO(img_response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            print(f"🎉 图片处理完成！")
            return (image_tensor,)
        except Exception as e:
            print(f"Qwen-Image API调用失败: {str(e)}")
            error_image = Image.new('RGB', (width, height), color='red')
            error_np = np.array(error_image).astype(np.float32) / 255.0
            error_tensor = torch.from_numpy(error_np)[None,]
            return (error_tensor,)

class QwenImageEditNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_token = load_api_token()
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "修改图片中的内容"
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "请输入您的魔搭API Token"
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "Qwen/Qwen-Image-Edit"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "QwenImage"

    def edit_image(self, image, prompt, api_token, model="Qwen/Qwen-Image-Edit", negative_prompt="", 
                   width=512, height=512, steps=30, guidance=3.5, seed=-1):
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
            # 将图像转换为临时文件并上传获取URL
            temp_img_path = None
            image_url = None
            try:
                # 保存图像到临时文件
                temp_img_path = os.path.join(tempfile.gettempdir(), f"qwen_edit_temp_{int(time.time())}.jpg")
                if len(image.shape) == 4:
                    img = image[0]
                else:
                    img = image
                
                i = 255. * img.cpu().numpy()
                img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img_pil.save(temp_img_path)
                print(f"✅ 图像已保存到临时文件: {temp_img_path}")
                
                # 上传图像到kefan.cn获取URL
                upload_url = 'https://ai.kefan.cn/api/upload/local'
                with open(temp_img_path, 'rb') as img_file:
                    files = {'file': img_file}
                    upload_response = requests.post(
                        upload_url,
                        files=files,
                        timeout=30
                    )
                    if upload_response.status_code == 200:
                        upload_data = upload_response.json()
                        # 修复这里的判断逻辑，kefan.cn返回code=200表示成功
                        if upload_data.get('success') == True and 'data' in upload_data:
                            image_url = upload_data['data']
                            print(f"✅ 图像已上传成功，获取URL: {image_url}")
                        else:
                            print(f"⚠️ 图像上传返回错误: {upload_response.text}")
                    else:
                        print(f"⚠️ 图像上传失败: {upload_response.status_code}, {upload_response.text}")
            except Exception as e:
                print(f"⚠️ 图像上传异常: {str(e)}")
            
            # 如果上传失败，回退到base64
            if not image_url:
                print("⚠️ 图像URL获取失败，回退到使用base64")
                image_data = tensor_to_base64_url(image)
                payload = {
                    'model': model,
                    'prompt': prompt,
                    'image': image_data
                }
            else:
                payload = {
                    'model': model,
                    'prompt': prompt,
                    'image_url': image_url
                }
            
            if negative_prompt.strip():
                payload['negative_prompt'] = negative_prompt
                print(f"🚫 负向提示词: {negative_prompt}")
                
            # 添加新参数
            if width != 512 or height != 512:
                size = f"{width}x{height}"
                payload['size'] = size
                print(f"📏 图像尺寸: {size}")
                
            if steps != 30:
                payload['steps'] = steps
                print(f"🔄 采样步数: {steps}")
                
            if guidance != 3.5:
                payload['guidance'] = guidance
                print(f"🧭 引导系数: {guidance}")
                
            if seed != -1:
                payload['seed'] = seed
                print(f"🎲 随机种子: {seed}")
            
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json',
                'X-ModelScope-Async-Mode': 'true'
            }
            
            print(f"🖼️ 开始编辑图片...")
            print(f"✏️ 编辑提示: {prompt}")
            
            url = 'https://api-inference.modelscope.cn/v1/images/generations'
            submission_response = requests.post(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers=headers,
                timeout=config.get("timeout", 60)
            )
            
            if submission_response.status_code != 200:
                raise Exception(f"API请求失败: {submission_response.status_code}, {submission_response.text}")
                
            submission_json = submission_response.json()
            result_image_url = None
            
            if 'task_id' in submission_json:
                task_id = submission_json['task_id']
                print(f"🕒 已提交任务，任务ID: {task_id}，开始轮询...")
                poll_start = time.time()
                max_wait_seconds = max(60, config.get('timeout', 720))
                
                while True:
                    task_resp = requests.get(
                        f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                        headers={
                            'Authorization': f'Bearer {api_token}',
                            'X-ModelScope-Task-Type': 'image_generation'
                        },
                        timeout=config.get("image_download_timeout", 120)
                    )
                    
                    if task_resp.status_code != 200:
                        raise Exception(f"任务查询失败: {task_resp.status_code}, {task_resp.text}")
                        
                    task_data = task_resp.json()
                    status = task_data.get('task_status')
                    
                    if status == 'SUCCEED':
                        output_images = task_data.get('output_images') or []
                        if not output_images:
                            raise Exception("任务成功但未返回图片URL")
                        result_image_url = output_images[0]
                        print("✅ 任务完成，开始下载编辑后的图片...")
                        break
                        
                    if status == 'FAILED':
                        error_message = task_data.get('errors', {}).get('message', '未知错误')
                        error_code = task_data.get('errors', {}).get('code', '未知错误码')
                        raise Exception(f"任务失败: 错误码 {error_code}, 错误信息: {error_message}")
                        
                    if time.time() - poll_start > max_wait_seconds:
                        raise Exception("任务轮询超时，请稍后重试或降低并发")
                        
                    time.sleep(5)
            else:
                raise Exception(f"未识别的API返回格式: {submission_json}")
                
            img_response = requests.get(result_image_url, timeout=config.get("image_download_timeout", 30))
            if img_response.status_code != 200:
                raise Exception(f"图片下载失败: {img_response.status_code}")
                
            pil_image = Image.open(BytesIO(img_response.content))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            # 清理临时文件
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
                except:
                    pass
            
            print(f"🎉 图片编辑完成！")
            return (image_tensor,)
            
        except Exception as e:
            print(f"Qwen-Image-Edit API调用失败: {str(e)}")
            # 返回原图像作为错误回退
            return (image.unsqueeze(0),)

NODE_CLASS_MAPPINGS = {
    "QwenImageNode": QwenImageNode,
    "QwenImageEditNode": QwenImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageNode": "Qwen-Image 生图节点",
    "QwenImageEditNode": "Qwen-Image 图像编辑节点"
} 