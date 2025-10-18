import requests
import json
import time
import os
import tempfile
import base64

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("警告: 未安装openai库，视频生文功能将不可用")
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
            "default_model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "timeout": 60,
            "default_prompt": "描述这个视频的内容",
            "cloudinary_cloud_name": "",
            "cloudinary_api_key": "",
            "cloudinary_api_secret": ""
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

def save_cloudinary_config(cloud_name, api_key, api_secret):
    """保存Cloudinary配置到config.json"""
    try:
        cfg = load_config()
        cfg["cloudinary_cloud_name"] = cloud_name
        cfg["cloudinary_api_key"] = api_key
        cfg["cloudinary_api_secret"] = api_secret
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存Cloudinary配置失败: {e}")
        return False

def upload_video_to_cloudinary(video_path, cloud_name=None, api_key=None, api_secret=None):
    """上传视频到Cloudinary获取URL"""
    try:
        import hashlib
        import time
        
        # 如果没有传入参数，从配置文件读取
        if not cloud_name or not api_key or not api_secret:
            config = load_config()
            cloud_name = cloud_name or config.get('cloudinary_cloud_name', '').strip()
            api_key = api_key or config.get('cloudinary_api_key', '').strip()
            api_secret = api_secret or config.get('cloudinary_api_secret', '').strip()
        
        if not all([cloud_name, api_key, api_secret]):
            print("Cloudinary配置不完整，请检查config.json中的cloudinary配置")
            return None
            
        # 生成签名
        timestamp = str(int(time.time()))
        public_id = f"comfyui_video_{timestamp}"
        
        # 创建签名字符串
        sign_string = f"public_id={public_id}&timestamp={timestamp}{api_secret}"
        signature = hashlib.sha1(sign_string.encode()).hexdigest()
        
        # Cloudinary上传URL
        upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
        
        # 准备上传数据
        with open(video_path, 'rb') as video_file:
            files = {'file': video_file}
            data = {
                'api_key': api_key,
                'timestamp': timestamp,
                'signature': signature,
                'public_id': public_id,
                'resource_type': 'video'
            }
            
            print(f"正在上传视频到Cloudinary...")
            upload_response = requests.post(
                upload_url,
                files=files,
                data=data,
                timeout=120  # 视频文件可能较大，增加超时时间
            )
            
            if upload_response.status_code == 200:
                upload_data = upload_response.json()
                if 'secure_url' in upload_data:
                    video_url = upload_data['secure_url']
                    print(f"视频已上传到Cloudinary成功，获取URL: {video_url}")
                    return video_url
                else:
                    print(f"Cloudinary上传返回格式错误: {upload_response.text}")
                    return None
            else:
                print(f"Cloudinary上传失败: {upload_response.status_code}, {upload_response.text}")
                return None
    except Exception as e:
        print(f"Cloudinary上传异常: {str(e)}")
        return None


def video_to_base64(video_path):
    """将视频文件转换为base64格式"""
    try:
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            return f"data:video/mp4;base64,{video_base64}"
    except Exception as e:
        print(f"视频base64转换失败: {e}")
        raise Exception(f"视频格式转换失败: {str(e)}")

class QwenVideoNode:
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
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "描述这个视频的内容")
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "请输入您的魔搭API Token"
                }),
            },
            "optional": {
                "video": ("VIDEO",),
                "video_path": ("STRING", {
                    "default": "",
                    "placeholder": "或者直接输入视频文件路径"
                }),
                "model": ("STRING", {
                    "default": config.get("default_video_model", "stepfun-ai/step3")
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
                    "default": config.get("default_video_seed", -1),
                    "min": -1,
                    "max": 2147483647
                }),
                "cloudinary_cloud_name": ("STRING", {
                    "default": config.get("cloudinary_cloud_name", ""),
                    "placeholder": "Cloudinary Cloud Name"
                }),
                "cloudinary_api_key": ("STRING", {
                    "default": config.get("cloudinary_api_key", ""),
                    "placeholder": "Cloudinary API Key"
                }),
                "cloudinary_api_secret": ("STRING", {
                    "default": config.get("cloudinary_api_secret", ""),
                    "placeholder": "Cloudinary API Secret"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze_video"
    CATEGORY = "QwenImage"

    def analyze_video(self, prompt="", api_token="", video=None, video_path="", model="stepfun-ai/step3", max_tokens=1000, temperature=0.7, seed=-1, cloudinary_cloud_name="", cloudinary_api_key="", cloudinary_api_secret="", error_message=""):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)
        
        config = load_config()
        
        if not api_token or api_token.strip() == "":
            raise Exception("请输入有效的API Token")
        
        # 优先使用VIDEO输入，如果没有则使用video_path
        actual_video_path = None
        
        if video is not None:
            # 从VIDEO输入中提取视频路径
            try:
                print(f"调试VIDEO输入:")
                print(f"   类型: {type(video)}")
                print(f"   内容: {video}")
                
                # 打印所有属性
                if hasattr(video, '__dict__'):
                    print(f"   属性: {video.__dict__}")
                elif hasattr(video, '__slots__'):
                    print(f"   槽位: {video.__slots__}")
                
                # 尝试各种可能的属性名，包括私有属性
                possible_attrs = ['filename', 'path', 'file_path', 'name', 'file', 'video_path', 'src', 'url', 'file_path', 'input_path', '_VideoFromFile__file']
                for attr in possible_attrs:
                    if hasattr(video, attr):
                        value = getattr(video, attr)
                        print(f"   找到属性 {attr}: {value}")
                        if isinstance(value, str) and value.strip():
                            actual_video_path = value
                            break
                
                # 如果是字符串
                if isinstance(video, str) and video.strip():
                    actual_video_path = video
                    print(f"   VIDEO是字符串: {video}")
                
                # 如果是列表或元组
                elif hasattr(video, '__getitem__') and len(video) > 0:
                    print(f"   VIDEO是序列，长度: {len(video)}")
                    first_item = video[0]
                    print(f"   第一个元素类型: {type(first_item)}")
                    print(f"   第一个元素内容: {first_item}")
                    
                    for attr in possible_attrs:
                        if hasattr(first_item, attr):
                            value = getattr(first_item, attr)
                            print(f"   第一个元素属性 {attr}: {value}")
                            if isinstance(value, str) and value.strip():
                                actual_video_path = value
                                break
                
                if actual_video_path:
                    print(f"成功提取视频路径: {actual_video_path}")
                else:
                    print(f"无法从VIDEO输入中提取路径")
                    
            except Exception as e:
                print(f"从VIDEO输入提取路径失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 如果VIDEO输入没有提供有效路径，使用video_path
        if not actual_video_path and video_path:
            actual_video_path = video_path
        
        if not actual_video_path or not os.path.exists(actual_video_path):
            raise Exception("请提供有效的视频文件（通过VIDEO输入或video_path参数）")
        
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("API Token已自动保存")
            else:
                print("API Token保存失败，但不影响当前使用")
        
        # 保存Cloudinary配置（如果提供了新的配置）
        if cloudinary_cloud_name and cloudinary_api_key and cloudinary_api_secret:
            current_config = load_config()
            if (current_config.get("cloudinary_cloud_name") != cloudinary_cloud_name or 
                current_config.get("cloudinary_api_key") != cloudinary_api_key or 
                current_config.get("cloudinary_api_secret") != cloudinary_api_secret):
                if save_cloudinary_config(cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret):
                    print("Cloudinary配置已自动保存")
                else:
                    print("Cloudinary配置保存失败，但不影响当前使用")
        
        try:
            print(f"🎬 开始分析视频...")
            print(f"📝 提示词: {prompt}")
            print(f"🤖 模型: {model}")
            print(f"视频路径: {actual_video_path}")
            
            # 处理随机种子
            if seed != -1:
                print(f"🎯 使用指定种子: {seed}")
            else:
                import random
                random_seed = random.randint(0, 2147483647)
                print(f"🎲 使用随机种子: {random_seed}")
                seed = random_seed
            
            # 尝试上传视频到Cloudinary获取URL
            video_url = upload_video_to_cloudinary(actual_video_path, cloudinary_cloud_name, cloudinary_api_key, cloudinary_api_secret)
            
            if video_url:
                # 使用URL方式
                print(f"🌐 使用视频URL: {video_url}")
                video_content = {
                    'type': 'video_url',
                    'video_url': {
                        'url': video_url,
                    },
                }
            else:
                # 回退到base64方式
                print("视频URL获取失败，回退到使用base64")
                video_data = video_to_base64(actual_video_path)
                video_content = {
                    'type': 'video_url',
                    'video_url': {
                        'url': video_data,
                    },
                }
            
            client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1',
                api_key=api_token
            )
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, video_content],
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
            print(f"视频分析完成!")
            print(f"📄 结果: {description[:100]}...")
            
            return (description,)
            
        except Exception as e:
            error_msg = f"视频分析失败: {str(e)}"
            print(f"{error_msg}")
            return (error_msg,)

if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "QwenVideoNode": QwenVideoNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenVideoNode": "Qwen-Video 视频生文节点"
    }
else:
    class OpenAINotInstalledVideoNode:
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
            return ("请先安装openai库才能使用视频生文功能: " + install_command,)
    
    NODE_CLASS_MAPPINGS = {
        "QwenVideoNode": OpenAINotInstalledVideoNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenVideoNode": "Qwen-Video 视频生文节点 (需要安装openai)"
    }
