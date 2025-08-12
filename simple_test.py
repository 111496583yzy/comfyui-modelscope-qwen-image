#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import os

def load_token():
    if os.path.exists('.qwen_token'):
        with open('.qwen_token', 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        token = input("请输入您的魔搭API Token: ").strip()
        if token:
            with open('.qwen_token', 'w', encoding='utf-8') as f:
                f.write(token)
        return token

def test_api_endpoint(url, payload, headers, name):
    print(f"\n🧪 测试 {name}")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text[:500]}...")
        if response.status_code == 200:
            print("✅ 请求成功")
            return True
        else:
            print("❌ 请求失败")
            return False
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("魔搭 API 接口测试工具")
    print("=" * 60)
    api_token = load_token()
    if not api_token:
        print("❌ 未获取到API token")
        return
    common_headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    test_prompt = "A beautiful sunset over the mountains"
    test_cases = [
        {
            'name': 'DashScope 图像合成 (异步)',
            'url': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis',
            'payload': {
                'model': 'wanx-v1',
                'input': {'prompt': test_prompt},
                'parameters': {'style': 'photography', 'size': '1024*1024'}
            },
            'headers': {**common_headers, 'X-DashScope-Async': 'enable'}
        },
        {
            'name': 'DashScope 图像合成 (同步)',
            'url': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis',
            'payload': {
                'model': 'wanx-v1',
                'input': {'prompt': test_prompt},
                'parameters': {'style': 'photography', 'size': '1024*1024'}
            },
            'headers': common_headers
        },
        {
            'name': '万相API',
            'url': 'https://api-inference.modelscope.cn/v1/models/wanx/text-to-image-synthesis',
            'payload': {
                'input': {'text': test_prompt}
            },
            'headers': common_headers
        },
        {
            'name': 'Qwen-Image模型',
            'url': 'https://api-inference.modelscope.cn/v1/models/qwen/qwen-image/text-to-image',
            'payload': {
                'input': {'prompt': test_prompt}
            },
            'headers': common_headers
        },
        {
            'name': '原始格式',
            'url': 'https://api-inference.modelscope.cn/v1/images/generations',
            'payload': {
                'model': 'Qwen/Qwen-Image',
                'prompt': test_prompt
            },
            'headers': common_headers
        }
    ]
    success_count = 0
    for test_case in test_cases:
        if test_api_endpoint(
            test_case['url'],
            test_case['payload'],
            test_case['headers'],
            test_case['name']
        ):
            success_count += 1
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{len(test_cases)} 个接口调用成功")
    print("=" * 60)

if __name__ == "__main__":
    main() 