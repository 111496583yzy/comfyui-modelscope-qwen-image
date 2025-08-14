#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def check_files():
    required_files = [
        '__init__.py',
        'qwen_image_node.py',
        'qwen_vision_node.py',
        'qwen_text_node.py',
        'config.json',
        'README.md',
        'requirements.txt'
    ]
    
    print("📁 检查文件完整性...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (缺失)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dependencies():
    print("\n📦 检查依赖包...")
    
    deps = {
        'requests': '网络请求',
        'PIL': '图像处理',
        'torch': '深度学习框架',
        'numpy': '数值计算',
        'openai': '文本生成和图生文功能',
        'httpx': '高级HTTP客户端',
        'socksio': 'SOCKS代理支持'
    }
    
    missing_deps = []
    
    for dep, desc in deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep} ({desc})")
        except ImportError:
            print(f"❌ {dep} ({desc}) - 未安装")
            missing_deps.append(dep)
    
    return len(missing_deps) == 0, missing_deps

def check_proxy_support():
    print("\n🌐 检查代理支持...")
    
    try:
        import httpx
        try:
            import socksio
            print("✅ SOCKS代理支持已安装")
            return True
        except ImportError:
            print("⚠️ SOCKS代理支持未安装，如果使用代理可能会出错")
            print("   建议运行: pip install httpx[socks] socksio")
            return False
    except ImportError:
        print("❌ httpx未安装")
        return False

def check_node_loading():
    print("\n🔧 检查节点加载...")
    
    try:
        from qwen_image_node import QwenImageNode
        node = QwenImageNode()
        input_types = node.INPUT_TYPES()
        print("✅ 文生图节点加载成功")
        
        from qwen_vision_node import QwenVisionNode, OPENAI_AVAILABLE
        if OPENAI_AVAILABLE:
            vision_node = QwenVisionNode()
            vision_input_types = vision_node.INPUT_TYPES()
            print("✅ 图生文节点加载成功")
        else:
            print("⚠️ 图生文节点加载成功，但OpenAI库不可用")
        
        from qwen_text_node import QwenTextNode
        if OPENAI_AVAILABLE:
            text_node = QwenTextNode()
            text_input_types = text_node.INPUT_TYPES()
            print("✅ 文本生成节点加载成功")
        else:
            print("⚠️ 文本生成节点加载成功，但OpenAI库不可用")
        
        return True
    except Exception as e:
        print(f"❌ 节点加载失败: {e}")
        return False

def check_config():
    print("\n⚙️ 检查配置文件...")
    
    try:
        import json
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = [
            'default_model',
            'default_vision_model',
            'default_text_model',
            'timeout',
            'default_prompt'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key in config:
                print(f"✅ {key}: {config[key]}")
            else:
                print(f"❌ {key} (缺失)")
                missing_keys.append(key)
        
        return len(missing_keys) == 0
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False

def main():
    print("=" * 60)
    print("Qwen-Image ComfyUI 插件安装验证")
    print("=" * 60)
    
    checks = [
        ("文件完整性", check_files),
        ("依赖包", lambda: check_dependencies()[0]),
        ("代理支持", check_proxy_support),
        ("配置文件", check_config),
        ("节点加载", check_node_loading),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}检查...")
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name}检查通过")
            else:
                print(f"❌ {check_name}检查失败")
        except Exception as e:
            print(f"❌ {check_name}检查出错: {e}")
        
        print("-" * 40)
    
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"\n📦 缺失的依赖包: {', '.join(missing_deps)}")
        print("运行以下命令安装:")
        print("python install_dependencies.py")
        print("或手动安装:")
        for dep in missing_deps:
            if dep == 'httpx':
                print(f"  pip install httpx[socks]")
            else:
                print(f"  pip install {dep}")
    
    print(f"\n📊 验证结果: {passed}/{total} 项检查通过")
    
    if passed >= total - 1:
        print("\n🎉 插件安装验证成功！")
        print("\n📋 下一步操作:")
        print("1. 将整个插件文件夹复制到 ComfyUI/custom_nodes/ 目录")
        print("2. 重启ComfyUI")
        print("3. 在节点列表中查找 'QwenImage' 分类")
        print("4. 准备好您的魔搭API Token")
        
        current_path = os.getcwd()
        if 'custom_nodes' in current_path:
            print("\n✅ 检测到您已在ComfyUI的custom_nodes目录中")
            print("   请直接重启ComfyUI即可使用")
        else:
            print(f"\n📁 当前路径: {current_path}")
            print("   请确保将插件复制到正确的ComfyUI目录")
            
        if not check_proxy_support():
            print("\n⚠️ 代理支持提醒:")
            print("   如果您使用代理上网，建议安装代理支持包:")
            print("   pip install httpx[socks] socksio")
    else:
        print("\n⚠️ 插件安装验证失败，请修复上述问题后重试")

if __name__ == "__main__":
    main()