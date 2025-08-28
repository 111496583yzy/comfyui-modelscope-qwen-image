# ComfyUI Qwen-Image 节点

本仓库提供了 [魔搭社区开放API](https://modelscope.cn/) 的 Qwen-Image 模型在 ComfyUI 中的节点实现。

## 特性

- 支持通过 [魔搭社区](https://modelscope.cn/) 的 API 调用 Qwen-Image 模型
- 支持图像尺寸、采样步数、引导系数等参数设置
- 支持随机种子与固定种子
- 支持负向提示词
- 支持 API Token 保存（首次填写后自动保存到 config.json）
- 支持图像编辑功能 (Qwen-Image-Edit 模型)

## 安装

1. 克隆本仓库到 ComfyUI 的 `custom_nodes` 目录下：

```
cd ComfyUI/custom_nodes
git clone https://github.com/111496583yzy/comfyui-modelscope-qwen-image.git comfyui-qwen-image
```

2. 重启 ComfyUI 服务

## 使用方法

### 1. 获取魔搭API Token

访问 [魔搭社区](https://modelscope.cn/) 并登录，在个人资料页获取 API Token。

### 2. Qwen-Image 生图节点

在 ComfyUI 编辑器中添加 `Qwen-Image 生图节点`，设置以下参数：

- **prompt**: 文本提示词
- **api_token**: 魔搭API Token (首次填写后会自动保存)
- **model**: 模型名称（默认为 "Qwen/Qwen-Image"）
- **negative_prompt**: 负向提示词（可选）
- **width/height**: 图像宽高（默认512x512）
- **seed**: 随机种子（-1表示使用随机种子）
- **steps**: 采样步数（默认30）
- **guidance**: 引导系数（默认7.5）

### 3. Qwen-Image 图像编辑节点

在 ComfyUI 编辑器中添加 `Qwen-Image 图像编辑节点`，设置以下参数：

- **image**: 要编辑的原始图像
- **prompt**: 描述要进行的编辑的文本提示词
- **api_token**: 魔搭API Token (首次填写后会自动保存)
- **model**: 模型名称（默认为 "Qwen/Qwen-Image-Edit"）
- **negative_prompt**: 负向提示词（可选）
- **width/height**: 图像宽高（默认512x512，范围64-1664）
- **steps**: 采样步数（范围1-100，默认30）
- **guidance**: 引导系数（范围1.5-20.0，默认3.5）
- **seed**: 随机种子（-1表示使用随机种子，0-2147483647为固定种子）

## 工作流示例

### 文本生图

1. 添加 `Qwen-Image 生图节点` 并设置提示词和其他参数
2. 连接输出到 `Preview Image` 节点

### 图像编辑

1. 准备一张原始图像（使用 `Load Image` 或其他方式）
2. 添加 `Qwen-Image 图像编辑节点`
3. 将原始图像连接到编辑节点的 `image` 输入
4. 设置编辑提示词（如"把狗变成猫"）
5. 连接输出到 `Preview Image` 节点

## 注意事项

- API 调用需要网络连接
- 高峰时期可能需要等待较长时间
- 请遵守魔搭社区的使用政策
- 如遇到错误代码429，表示请求过多，需要等待一段时间后重试

## License

MIT
