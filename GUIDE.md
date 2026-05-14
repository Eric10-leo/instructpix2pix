# InstructPix2Pix 机器人帧预测 —— 从零完整指南

## 项目概述

微调 [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) 模型，输入当前图像 + 文字指令，预测机械臂 50 帧后的状态。

**整体流程：**

```
原始数据 → 数据预处理 → 制作训练/验证/测试集 → 微调模型 → 评估 → 可视化对比
```

---

## 第一步：环境搭建

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate    # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

---

## 第二步：准备原始数据

原始数据应放在 `dataset/` 目录下，按以下结构组织：

```
dataset/
├── block_hammer_beat_D435_pkl/
│   ├── episode0/
│   │   ├── front_camera_0.jpg
│   │   ├── front_camera_1.jpg
│   │   ├── head_camera_0.jpg
│   │   ├── right_camera_0.jpg
│   │   ├── left_camera_0.jpg
│   │   └── ...
│   ├── episode1/
│   └── ...
├── block_handover_D435_pkl/
│   └── ...
└── blocks_stack_easy_D435_pkl/
    └── ...
```

- 每个任务一个文件夹，命名格式：`{任务名}_D435_pkl`
- 每个 episode 一个子文件夹
- 每个 episode 内按视角分文件：`{front/head/right/left}_camera_{帧号}.jpg`
- 支持的任务类型（定义在 `process_robot_dataset.py` 的 `TASKS` 字典中）：
  - `block_hammer_beat` — 锤子击打方块
  - `block_handover` — 传递方块
  - `blocks_stack_easy` — 堆叠方块

---

## 第三步：数据预处理（原始帧 → 配对数据）

运行 `process_robot_dataset.py`，将原始帧处理为配对数据：

```bash
python scripts/process_robot_dataset.py \
  --source_dir ./dataset \
  --output_dir ./processed_data
```

**这个脚本做了什么：**
1. 遍历所有任务、所有 episode、所有视角
2. 将每一帧和它 50 帧之后的帧配对（输入 → 目标）
3. 将图像缩放到 128×128
4. 按 9:1 比例随机划分为训练集和测试集（以 episode 为单位，防止数据泄漏）
5. 生成 `train_annotations.txt` 和 `test_annotations.txt`

**输出结构：**

```
processed_data/
├── train/
│   ├── {task}_{view}_{episode}_f{start}_to_f{end}_input.jpg
│   └── {task}_{view}_{episode}_f{start}_to_f{end}_target.jpg
├── test/
│   └── （同上）
├── train_annotations.txt    # 格式: input.jpg|指令|target.jpg
└── test_annotations.txt
```

---

## 第四步：制作训练/验证/测试数据集

### 4.1 方法 A：从 processed_data 直接制作（推荐）

使用 `reprocess_dataset.py`：

```bash
python scripts/reprocess_dataset.py
```

**注意：** 脚本中 `scene_id` 变量（第 10 行）需要改成你自己的场景编号。该脚本会：
1. 读取 `train_annotations.txt`，将图片拷贝到 `data/train/sceneX/`
2. 读取 `test_annotations.txt`，将图片拷贝到 `data/validation/sceneX/`
3. 生成对应的 JSONL 元数据文件

### 4.2 方法 B：制作训练集 + 验证集

使用 `make_train_data.py`：

```bash
python scripts/make_train_data.py
```

- 从 `processed_data/train_annotations.txt` 随机抽取 900 条作为训练集，100 条作为验证集
- 帧按顺序重命名为 `frame_00001.png`、`frame_00002.png`...
- 输出到 `data/train/scene6/` 和 `data/validation/scene6/`（默认 scene6）

### 4.3 方法 C：制作测试集

使用 `make_test_data.py`：

```bash
python scripts/make_test_data.py
```

- 随机抽取 100 条数据
- 输出到 `data/test/`
- 生成 `metadata_test.jsonl`

**最终数据目录结构：**

```
data/
├── train/
│   └── sceneX/
│       ├── frame_00001.png
│       ├── frame_00002.png
│       ├── ...
│       └── metadata_sceneX.jsonl
├── validation/
│   └── sceneX/
│       ├── frame_00101.png
│       └── metadata_sceneX_validation.jsonl
└── test/
    ├── frame_00001.png
    ├── ...
    └── metadata_test.jsonl
```

**JSONL 格式（每行一条）：**

```json
{
  "image": "frame_00001.png",
  "edited_image": "frame_00002.png",
  "edit_prompt": "Predict the state of the arm after 50 frames: hit the block with the hammer"
}
```

---

## 第五步：微调模型

```bash
export SCENE_ID=4          # 设置场景编号
python scripts/finetune.py
```

**关键超参数（可在 `parse_args()` 中调整）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `train_batch_size` | 32 | 训练批大小 |
| `max_train_steps` | 300 | 最大训练步数 |
| `learning_rate` | 2e-5 | 学习率 |
| `gradient_accumulation_steps` | 8 | 梯度累积 |
| `mixed_precision` | fp16 | 混合精度训练 |
| `checkpointing_steps` | 50 | 多少步保存一次检查点 |
| `resolution` | 256 | 图像分辨率 |
| `seed` | 42 | 随机种子 |

**训练输出：**

```
robot_arm_model_scene4/
├── unet/                    # 微调后的 UNet 权重
├── scheduler/               # 噪声调度器
├── tokenizer/               # CLIP tokenizer
├── text_encoder/            # 文本编码器
├── vae/                     # VAE 解码器
├── logs/                    # TensorBoard 日志
└── training_val_loss.png    # 训练/验证损失曲线
```

**注意事项：**
- `finetune.py` 第 33 行设置了 HF 镜像加速 `HF_ENDPOINT = 'https://hf-mirror.com'`，国内环境如有需要可保留，国外环境可删除
- 首次运行会自动从 Hugging Face 下载预训练模型 `timbrooks/instruct-pix2pix`
- VAE 和 Text Encoder 被冻结，只训练 UNet

---

## 第六步：评估模型

```bash
python scripts/eval.py --models 0 1 2 3 4
```

- `--models 0` 表示评估原始预训练模型（作为基准）
- `--models 1 2 3 4` 表示评估 `robot_arm_model_scene1/` ~ `robot_arm_model_scene4/`
- 评估指标：**SSIM**（结构相似性）和 **PSNR**（峰值信噪比）

**输出：**

```
results/
├── samples/
│   ├── original/            # 原始模型生成样本
│   │   ├── input_0.png
│   │   ├── output_0.png
│   │   └── sample_info.json
│   └── scene4/              # 微调模型生成样本
│       └── ...
├── metrics.json             # 所有模型的 SSIM/PSNR 汇总
├── base_metrics.csv         # 原始模型逐样本指标
├── scene4_metrics.csv       # 微调模型逐样本指标
├── summary_metrics.csv      # 汇总对比表
└── evaluation_results.png   # 柱状图对比
```

---

## 第七步：可视化对比

```bash
python scripts/sample_figures.py --models 0 3 4 5 --num_samples 10
```

**输出：**

```
sample_figures/
├── sample_000/
│   ├── input.png            # 输入图像
│   ├── target.png           # 真实目标图像
│   ├── output_model_0.png   # 原始模型预测
│   ├── output_model_3.png   # 模型3预测
│   └── ...
├── sample_001/
│   └── ...
└── ...
```

每个 `sample_XXX/` 文件夹内包含一个测试样本的输入、目标真值、以及各模型的预测输出，方便并排对比。

---

## 常见问题

### 显存不足
- 减小 `train_batch_size`（如 32 → 8）
- 增大 `gradient_accumulation_steps` 保持等效批大小
- 降低 `resolution`（如 256 → 128）

### 修改帧间隔
- 所有脚本默认帧间隔为 50，在 `process_robot_dataset.py` 的 `frame_gap` 参数中修改

### 添加新任务
- 在 `process_robot_dataset.py` 的 `TASKS` 字典中添加任务名和对应指令

### CUDA 版本不匹配
- `requirements.txt` 中的 `torch==2.3.0` 对应 CUDA 12.2，可根据实际 CUDA 版本调整：
  ```bash
  pip install torch==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 联系

- Tian Zeyu: [183441801@qq.com](mailto:183441801@qq.com)
