#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen-Image ComfyUI 插件故障排除工具
自动诊断和解决常见问题
"""

import os
import sys
import subprocess
import json

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title):
    """打印章节标题"""
    print(f"\n🔍 {title}")
    print("-" * 40)

def run_command(command, description):
    """运行命令并返回结果"""
    print(f"📋 {description}")
    print(f"💻 命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print(f"📄 输出: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print("❌ 失败")
            if result.stderr.strip():
                print(f"🚨 错误: {result.stderr.strip()}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print("⏰ 超时")
        return False, "命令执行超时"
    except Exception as e:
        print(f"💥 异常: {str(e)}")
        return False, str(e)

def check_python_environment():
    """检查Python环境"""
    print_section("Python环境检查")
    
    # Python版本
    run_command("python --version", "检查Python版本")
    
    # pip版本
    run_command("pip --version", "检查pip版本")
    
    # 已安装的包
    print("\n📦 检查关键包安装状态:")
    packages = ['requests', 'pillow', 'torch', 'numpy', 'openai', 'httpx', 'socksio']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")

def check_files():
    """检查文件完整性"""
    print_section("文件完整性检查")
    
    required_files = [
        '__init__.py',
        'qwen_image_node.py', 
        'qwen_vision_node.py',
        'config.json',
        'requirements.txt'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} (缺失)")

def check_config():
    """检查配置文件"""
    print_section("配置文件检查")
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✅ config.json 格式正确")
        
        # 检查关键配置项
        key_configs = [
            'default_model',
            'default_vision_model', 
            'timeout',
            'default_prompt'
        ]
        
        for key in key_configs:
            if key in config:
                print(f"✅ {key}: {config[key]}")
            else:
                print(f"❌ {key} (缺失)")
                
    except Exception as e:
        print(f"❌ config.json 读取失败: {e}")

def check_network():
    """检查网络连接"""
    print_section("网络连接检查")
    
    # 检查基本网络连接
    run_command("ping -c 3 8.8.8.8", "检查基本网络连接")
    
    # 检查API服务器连接
    try:
        import requests
        response = requests.get('https://api-inference.modelscope.cn', timeout=10)
        print(f"✅ API服务器连接正常 (状态码: {response.status_code})")
    except Exception as e:
        print(f"❌ API服务器连接失败: {e}")
    
    # 检查代理设置
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY']
    print("\n🌐 代理环境变量:")
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚪ {var}: 未设置")

def check_token():
    """检查API Token"""
    print_section("API Token检查")
    
    token_sources = ['.qwen_token', 'config.json']
    token_found = False
    
    for source in token_sources:
        if source == '.qwen_token' and os.path.exists(source):
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    token = f.read().strip()
                if token:
                    print(f"✅ 在 {source} 中找到token (长度: {len(token)})")
                    token_found = True
                else:
                    print(f"⚪ {source} 存在但为空")
            except Exception as e:
                print(f"❌ 读取 {source} 失败: {e}")
        
        elif source == 'config.json':
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                token = config.get('api_token', '').strip()
                if token:
                    print(f"✅ 在 {source} 中找到token (长度: {len(token)})")
                    token_found = True
                else:
                    print(f"⚪ {source} 中token为空")
            except Exception as e:
                print(f"❌ 读取 {source} 失败: {e}")
    
    if not token_found:
        print("❌ 未找到有效的API token")

def run_diagnostic_tests():
    """运行诊断测试"""
    print_section("诊断测试")
    
    tests = [
        ("python verify_installation.py", "运行安装验证"),
        ("python test_vision_with_proxy.py", "运行代理测试"),
    ]
    
    for command, description in tests:
        if os.path.exists(command.split()[1]):
            success, output = run_command(command, description)
            if not success:
                print(f"⚠️ {description} 失败，请查看详细输出")
        else:
            print(f"⚪ {command.split()[1]} 不存在，跳过测试")

def suggest_solutions():
    """建议解决方案"""
    print_section("建议解决方案")
    
    solutions = [
        "🔧 安装缺失依赖: python install_dependencies.py",
        "🔍 验证安装: python verify_installation.py", 
        "🌐 测试代理: python test_vision_with_proxy.py",
        "📖 查看快速指南: cat QUICKSTART.md",
        "🔗 查看代理指南: cat PROXY_GUIDE.md",
        "🖼️ 查看图生文指南: cat VISION_GUIDE.md",
        "🔄 重启ComfyUI以加载更新",
        "🧹 清理Python缓存: rm -rf __pycache__",
    ]
    
    for solution in solutions:
        print(solution)

def main():
    print_header("Qwen-Image ComfyUI 插件故障排除工具")
    
    print("🚀 开始全面诊断...")
    
    # 运行所有检查
    check_python_environment()
    check_files()
    check_config()
    check_network()
    check_token()
    run_diagnostic_tests()
    suggest_solutions()
    
    print_header("诊断完成")
    
    print("\n💡 根据上述诊断结果:")
    print("1. 如果发现缺失依赖，运行: python install_dependencies.py")
    print("2. 如果网络有问题，查看: PROXY_GUIDE.md")
    print("3. 如果token有问题，重新输入API token")
    print("4. 如果文件缺失，重新下载插件")
    print("5. 完成修复后，重启ComfyUI")
    
    print("\n📞 如果问题仍然存在:")
    print("- 查看ComfyUI控制台的完整错误日志")
    print("- 尝试在不同网络环境下测试")
    print("- 确认ComfyUI版本兼容性")

if __name__ == "__main__":
    main()