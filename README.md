# Qwen-Image ComfyUI 插件

这是一个ComfyUI自定义节点插件，用于调用魔搭(ModelScope)的Qwen-Image API生成图片。

## 功能特性

- 🎨 直接在ComfyUI中调用魔搭的Qwen-Image API
- ⚙️ 支持配置文件管理API token和其他设置
- 🔧 完整的错误处理和超时管理
- 🖼️ 自动将生成的图片转换为ComfyUI格式
- 📏 支持自定义图片尺寸（64x64 到 2048x2048）
- 🚫 支持负向提示词，提高图片质量控制
- 🎲 支持随机种子和固定种子控制
- ⚙️ 支持采样步数调节（1-100步）
- 🎛️ 支持引导系数调节（1.5-20.0）

## 安装步骤

### 1. 复制插件文件
将整个插件文件夹复制到ComfyUI的自定义节点目录：
```bash
# ComfyUI安装目录/custom_nodes/qwen-image/
cp -r qwen-image/ /path/to/ComfyUI/custom_nodes/
```

### 2. 安装依赖
确保安装了必要的Python包（通常ComfyUI环境已包含大部分）：
```bash
pip install requests pillow torch numpy
```

### 3. 准备API Token
获取您的魔搭API token（见下方"获取API Token"章节），在ComfyUI界面中直接输入即可，无需预先配置。

### 4. 重启ComfyUI
重启ComfyUI以加载新的自定义节点。

## 使用方法

### 1. 在ComfyUI中查找节点
- 重启ComfyUI后，在节点列表中查找 "QwenImage" 分类
- 找到 "Qwen-Image 生图节点" 节点

### 2. 连接节点
- 拖拽节点到工作流中
- 输入您的提示词(prompt)
- **首次使用时**：在api_token字段输入您的魔搭API token，插件会自动保存
- **后续使用**：token会自动加载，无需重复输入
- 如需要可以修改模型名称
- 将输出连接到预览节点或保存节点

### 3. 参数说明
- **prompt** (必填): 图片生成的提示词，支持中英文，建议使用英文效果更好
- **api_token** (必填): 您的魔搭API token，首次输入后会自动保存和加载
- **model** (可选): 使用的模型，默认为 "Qwen/Qwen-Image"
- **negative_prompt** (可选): 负向提示词，描述不希望出现的内容
- **width** (可选): 图片宽度，范围64-2048，默认512，步长64
- **height** (可选): 图片高度，范围64-2048，默认512，步长64
- **seed** (可选): 随机种子，-1为随机，0-2147483647为固定种子
- **steps** (可选): 采样步数，范围1-100，默认30，更高值质量更好但速度更慢
- **guidance** (可选): 提示词引导系数，范围1.5-20.0，默认7.5，更高值更贴近提示词

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

- 查看ComfyUI控制台输出的错误信息
- 检查 `config.json` 文件格式是否正确
- 确认在ComfyUI界面输入的API token有效且有足够的使用额度
- 可以删除 `.qwen_token` 文件重新输入token

## 使用示例

### 基础使用
```
prompt: "A cute orange cat"
width: 512, height: 512
seed: -1（随机）
```

### 高质量精细图像
```  
prompt: "A beautiful mountain landscape with flowing river"
negative_prompt: "blurry, low quality, distorted"
width: 1024, height: 768
seed: 12345（固定）
steps: 50
guidance: 10.0
```

### 艺术风格控制
```
prompt: "Anime character with blue hair"
negative_prompt: "realistic, photography, 3d render"
width: 768, height: 768
steps: 35
guidance: 8.5
```


- 将环境变量 `MODELSCOPE_API_KEY` 设置为您的魔搭 API Token，或直接替换示例中的 `YOUR_API_TOKEN`。

## 文件结构

```
qwen-image/
├── __init__.py              # 插件初始化文件
├── qwen_image_node.py       # 主节点实现
├── config.json              # 配置文件
├── test_api.py              # API测试脚本
├── .qwen_token              # 自动生成的token存储文件（首次使用后出现）
└── README.md               # 使用说明
```

## 技术支持

如遇到问题，请检查：
1. ComfyUI版本兼容性
2. Python依赖包安装情况
3. 网络连接状态
4. API token有效性
