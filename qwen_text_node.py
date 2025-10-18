import requests
import json
import time
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("警告: 未安装openai库，文本生成功能将不可用")
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
            "default_text_model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "timeout": 60,
            "default_system_prompt": "You are a helpful assistant.",
            "default_user_prompt": "你好"
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

class QwenTextNode:
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
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_user_prompt", "你好")
                }),
                "api_token": ("STRING", {
                    "default": saved_token,
                    "placeholder": "请输入您的魔搭API Token"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_system_prompt", "You are a helpful assistant.")
                }),
                "model": ("STRING", {
                    "default": config.get("default_text_model", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
                }),
                "max_tokens": ("INT", {
                    "default": 2000,
                    "min": 100,
                    "max": 8000
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
                "stream": ("BOOLEAN", {
                    "default": True
                }),
                "seed": ("INT", {
                    "default": config.get("default_text_seed", -1),
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "QwenImage"

    def generate_text(self, user_prompt="", api_token="", system_prompt="You are a helpful assistant.", model="Qwen/Qwen3-Coder-480B-A35B-Instruct", max_tokens=2000, temperature=0.7, stream=True, seed=-1, error_message=""):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)
        
        config = load_config()
        
        if not api_token or api_token.strip() == "":
            raise Exception("请输入有效的API Token")
        
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("API Token已自动保存")
            else:
                print("API Token保存失败，但不影响当前使用")
        
        try:
            print(f"💬 开始文本生成...")
            print(f"🤖 模型: {model}")
            print(f"📝 用户提示: {user_prompt[:50]}...")
            print(f"系统提示: {system_prompt[:50]}...")
            print(f"温度: {temperature}")
            print(f"📊 最大tokens: {max_tokens}")
            print(f"⚡ 流式输出: {stream}")
            
            # 处理随机种子
            if seed != -1:
                print(f"🎯 使用指定种子: {seed}")
            else:
                import random
                random_seed = random.randint(0, 2147483647)
                print(f"🎲 使用随机种子: {random_seed}")
                seed = random_seed
            
            client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1',
                api_key=api_token
            )
            
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ]
            
            print(f"🚀 发送API请求...")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                seed=seed
            )
            
            if stream:
                print("📡 接收流式响应...")
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end='', flush=True)
                
                print(f"\n流式生成完成!")
                print(f"📄 总长度: {len(full_response)} 字符")
                return (full_response,)
            else:
                result = response.choices[0].message.content
                print(f"文本生成完成!")
                print(f"📄 结果长度: {len(result)} 字符")
                print(f"📝 结果预览: {result[:100]}...")
                return (result,)
            
        except Exception as e:
            error_msg = f"文本生成失败: {str(e)}"
            print(f"{error_msg}")
            return (error_msg,)

if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "QwenTextNode": QwenTextNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenTextNode": "Qwen-Text 文本生成节点"
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
            return ("请先安装openai库才能使用文本生成功能: " + install_command,)
    
    NODE_CLASS_MAPPINGS = {
        "QwenTextNode": OpenAINotInstalledNode
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "QwenTextNode": "Qwen-Text 文本生成节点 (需要安装openai)"
    }