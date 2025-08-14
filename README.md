# Qwen-Image ComfyUI 插件

这是一个ComfyUI自定义节点插件，用于调用魔搭(ModelScope)的Qwen-Image API生成图片。

## 功能特性

### 文生图功能
- 🎨 直接在ComfyUI中调用魔搭的Qwen-Image API生成图片
- ⚙️ 支持配置文件管理API token和其他设置
- 🔧 完整的错误处理和超时管理
- 🖼️ 自动将生成的图片转换为ComfyUI格式
- 📏 支持自定义图片尺寸（64x64 到 2048x2048）
- 🚫 支持负向提示词，提高图片质量控制
- 🎲 支持随机种子和固定种子控制
- ⚙️ 支持采样步数调节（1-100步）
- 🎛️ 支持引导系数调节（1.5-20.0）

### 图生文功能 🆕
- 🔍 直接在ComfyUI中调用魔搭的Vision API分析图片
- 📝 支持中英文提示词，智能图像描述
- 🖼️ 自动将ComfyUI图像tensor转换为API兼容格式
- 🎯 支持多种分析任务：图像描述、内容分析、情感识别等
- 🔧 可调节温度、最大token数等参数

### 文本生成功能 🆕
- 💬 直接在ComfyUI中调用魔搭的Qwen3-Coder大语言模型
- 🧠 支持对话、代码生成、文本创作等多种任务
- ⚙️ 支持自定义系统提示词和用户提示词
- ⚡ 支持流式和非流式两种输出模式
- 🎛️ 可调节温度、最大token数等生成参数

## 安装步骤

### 1. 复制插件文件
将整个插件文件夹复制到ComfyUI的自定义节点目录：
```bash
# ComfyUI安装目录/custom_nodes/qwen-image/
cp -r qwen-image/ /path/to/ComfyUI/custom_nodes/
```

### 2. 安装依赖

#### 自动安装（推荐）
运行依赖安装脚本：
```bash
python install_dependencies.py
```

#### 手动安装
确保安装了必要的Python包：
```bash
# 核心依赖（通常ComfyUI环境已包含）
pip install requests pillow torch numpy

# 图生文功能依赖
pip install openai
```

#### 从requirements.txt安装
```bash
pip install -r requirements.txt
```

### 3. 准备API Token
获取您的魔搭API token（见下方"获取API Token"章节），在ComfyUI界面中直接输入即可，无需预先配置。

### 4. 验证安装
运行安装验证脚本：
```bash
python verify_installation.py
```

### 5. 重启ComfyUI
重启ComfyUI以加载新的自定义节点。

## 使用方法

### 1. 在ComfyUI中查找节点
- 重启ComfyUI后，在节点列表中查找 "QwenImage" 分类
- 找到以下节点：
  - "Qwen-Image 生图节点" - 文生图功能
  - "Qwen-Vision 图生文节点" - 图生文功能
  - "Qwen-Text 文本生成节点" - 文本生成功能

### 2. 连接节点
- 拖拽节点到工作流中
- 输入您的提示词(prompt)
- **首次使用时**：在api_token字段输入您的魔搭API token，插件会自动保存
- **后续使用**：token会自动加载，无需重复输入
- 如需要可以修改模型名称
- 将输出连接到预览节点或保存节点

### 3. 参数说明

#### 文生图节点参数
- **prompt** (必填): 图片生成的提示词，支持中英文，建议使用英文效果更好
- **api_token** (必填): 您的魔搭API token，首次输入后会自动保存和加载
- **model** (可选): 使用的模型，默认为 "Qwen/Qwen-Image"
- **negative_prompt** (可选): 负向提示词，描述不希望出现的内容
- **width** (可选): 图片宽度，范围64-2048，默认512，步长64
- **height** (可选): 图片高度，范围64-2048，默认512，步长64
- **seed** (可选): 随机种子，-1为随机，0-2147483647为固定种子
- **steps** (可选): 采样步数，范围1-100，默认30，更高值质量更好但速度更慢
- **guidance** (可选): 提示词引导系数，范围1.5-20.0，默认7.5，更高值更贴近提示词

#### 图生文节点参数
- **image** (必填): 输入的图像，来自ComfyUI工作流中的其他节点
- **prompt** (必填): 分析提示词，如"描述这幅图"、"分析图像内容"等
- **api_token** (必填): 您的魔搭API token，与文生图节点共享
- **model** (可选): 使用的视觉模型，默认为 "stepfun-ai/step3"
- **max_tokens** (可选): 最大输出token数，范围100-4000，默认1000
- **temperature** (可选): 生成温度，范围0.1-2.0，默认0.7，越高越随机

#### 文本生成节点参数
- **user_prompt** (必填): 用户输入的提示词或问题
- **api_token** (必填): 您的魔搭API token，与其他节点共享
- **system_prompt** (可选): 系统提示词，定义AI的角色和行为
- **model** (可选): 使用的文本模型，默认为 "Qwen/Qwen3-Coder-480B-A35B-Instruct"
- **max_tokens** (可选): 最大输出token数，范围100-8000，默认2000
- **temperature** (可选): 生成温度，范围0.1-2.0，默认0.7
- **stream** (可选): 是否使用流式输出，默认为True

## 配置文件说明

`config.json` 文件包含以下配置项（API token会自动保存到单独文件）：

```json
{
  "default_model": "Qwen/Qwen-Image",
  "timeout": 720,  // API请求超时时间（秒），免费API可能需要排队
  "image_download_timeout": 120,  // 图片下载超时时间（秒）
  "default_prompt": "A beautiful landscape",
  "default_negative_prompt": "lowres, bad anatomy, bad hands...",  // 默认负向提示词
  "default_width": 512,  // 默认图片宽度
  "default_height": 512,  // 默认图片高度
  "default_seed": -1,  // 默认种子（-1为随机）
  "default_steps": 30,  // 默认采样步数
  "default_guidance": 4  // 默认引导系数
}
```

**Token管理**：
- API token在ComfyUI界面输入后，会自动保存到 `.qwen_token` 文件
- 下次使用时会自动加载，无需重复输入
- 如需更换token，直接在界面输入新的token即可自动更新

## 获取API Token

1. 访问 [魔搭社区](https://modelscope.cn)
2. 注册并登录账号
3. 进入个人中心，获取API Token
4. 将token填入配置文件

⚠️ **免费API使用说明**：
- 免费API每天有使用额度限制（约2000次）
- 可能需要排队等待，请耐心等候（最长12分钟）
- 如需稳定服务，建议使用商业版API

## 故障排除

### 常见问题

1. **节点没有出现在ComfyUI中**
   - 检查文件是否正确放置在 `custom_nodes` 目录下
   - 确保文件权限正确
   - 重启ComfyUI

2. **插件导入失败 (IMPORT FAILED)**
   - 运行 `python install_dependencies.py` 检查并安装依赖
   - 手动安装缺失的包：`pip install openai`
   - 检查ComfyUI控制台的详细错误信息

3. **代理相关错误 (SOCKS proxy error)**
   - 错误信息：`Using SOCKS proxy, but the 'socksio' package is not installed`
   - 解决方案：`pip install httpx[socks] socksio`
   - 或运行：`python install_dependencies.py`
   - 详细指南：查看 `PROXY_GUIDE.md`

2. **API调用失败**
   - 检查在ComfyUI界面输入的API token是否正确
   - 确认网络连接正常
   - 查看ComfyUI控制台的错误信息

3. **生成红色错误图片**
   - 通常表示API调用出现问题
   - 查看终端日志了解具体错误信息
   - 检查API token和网络连接
   - 尝试在界面重新输入token

### 调试方法

#### 自动诊断（推荐）
```bash
python troubleshoot.py
```
这个脚本会自动检查所有可能的问题并提供解决方案。

#### 手动调试
- 查看ComfyUI控制台输出的错误信息
- 检查 `config.json` 文件格式是否正确
- 确认在ComfyUI界面输入的API token有效且有足够的使用额度
- 可以删除 `.qwen_token` 文件重新输入token
- 运行 `python verify_installation.py` 验证安装
- 运行 `python test_vision_with_proxy.py` 测试代理环境

## 使用示例

### 文生图示例

#### 基础使用
```
prompt: "A cute orange cat"
width: 512, height: 512
seed: -1（随机）
```

#### 高质量精细图像
```  
prompt: "A beautiful mountain landscape with flowing river"
negative_prompt: "blurry, low quality, distorted"
width: 1024, height: 768
seed: 12345（固定）
steps: 50
guidance: 10.0
```

#### 艺术风格控制
```
prompt: "Anime character with blue hair"
negative_prompt: "realistic, photography, 3d render"
width: 768, height: 768
steps: 35
guidance: 8.5
```

### 图生文示例 🆕

#### 基础图像描述
```
prompt: "描述这幅图像"
model: "stepfun-ai/step3"
max_tokens: 500
temperature: 0.7
```

#### 详细内容分析
```
prompt: "请详细分析这幅图像，包括主要对象、场景、颜色、构图、风格等方面。"
max_tokens: 1000
temperature: 0.5
```

#### 专业用途分析
```
prompt: "从摄影技术角度分析这张照片的构图、光线和色彩运用。"
max_tokens: 800
temperature: 0.3
```

#### 创意描述
```
prompt: "用诗意的语言描述这幅图像，营造艺术氛围。"
max_tokens: 600
temperature: 1.0
```

### 文本生成示例 🆕

#### 基础对话
```
user_prompt: "你好，请介绍一下你自己"
system_prompt: "You are a helpful assistant."
model: "Qwen/Qwen3-Coder-480B-A35B-Instruct"
max_tokens: 1000
temperature: 0.7
stream: true
```

#### 代码生成
```
user_prompt: "请写一个Python函数来计算斐波那契数列"
system_prompt: "You are a professional programmer."
max_tokens: 2000
temperature: 0.3
```

#### 文本创作
```
user_prompt: "写一首关于春天的短诗"
system_prompt: "You are a creative writer."
max_tokens: 800
temperature: 1.0
```

#### 技术问答
```
user_prompt: "解释一下什么是机器学习"
system_prompt: "You are a technical expert who explains complex concepts clearly."
max_tokens: 1500
temperature: 0.5
```


- 将环境变量 `MODELSCOPE_API_KEY` 设置为你的魔搭 API Token，或直接替换示例中的 `YOUR_API_TOKEN`。

## 文件结构

```
qwen-image/
├── __init__.py              # 插件初始化文件
├── qwen_image_node.py       # 文生图节点实现
├── qwen_vision_node.py      # 图生文节点实现 🆕
├── qwen_text_node.py        # 文本生成节点实现 🆕
├── config.json              # 配置文件
├── requirements.txt         # 依赖包列表 🆕
├── install_dependencies.py # 依赖安装脚本 🆕
├── verify_installation.py  # 安装验证脚本 🆕
├── troubleshoot.py          # 故障排除工具 🆕
├── .qwen_token              # 自动生成的token存储文件（首次使用后出现）
└── README.md               # 使用说明
```

## 技术支持

如遇到问题，请检查：
1. ComfyUI版本兼容性
2. Python依赖包安装情况
3. 网络连接状态
4. API token有效性
