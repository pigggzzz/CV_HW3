# 2DGS + AIGC 多源资产生成与真实场景融合（可复现工程骨架）

本目录提供一个**配置驱动、模块化、可脚本化**的工程骨架，用于完成作业要求的全链路流程：

- **Object A**：真实多视角（COLMAP 位姿）+ 2DGS 重建
- **Object B**：threestudio 文本到 3D（SDS）
- **Object C**：Magic123 单图到 3D
- **Background**：开源数据集场景 2DGS 重建
- **Fusion & Rendering**：统一导出 + Blender 融合漫游渲染
- **WandB**：训练/生成过程记录与本地 logs 备份

> 说明：本仓库不内置 COLMAP / 2DGS / threestudio / Magic123 的完整源码；而是提供**统一入口与封装**，通过配置里指定外部项目路径与命令行参数实现可复现运行。

---

## 目录结构

与作业要求一致（已在 `2dgs_aigc/` 下创建）：

- `configs/`：所有 Prompt、路径、训练/融合参数
- `data/raw/`：原始输入（视频、图片、Prompt、背景数据集）
- `data/processed/`：中间结果（COLMAP、2DGS 输出等）
- `assets/`：统一输出（mesh/gaussians/blender）
- `script/`：一键脚本与 Blender 渲染脚本
- `src/`：核心模块封装（reconstruction / text_to_3d / image_to_3d / fusion / rendering / utils）
- `output/`：最终视频与截图
- `logs/`：本地日志与 wandb 备份

---

## 环境（四套 Conda + 四套 requirements）

| 环境名 | Python | CUDA | 主要用途 | 依赖文件 |
|--------|--------|------|----------|----------|
| `env_colmap` | 3.10 | — | **仅** Object A 的抽帧 + COLMAP | `requirements_env_colmap.txt` |
| `env_gs` | 3.10 / 3.8* | 12.x | **仅** 2DGS（Object A / Background 重建、mesh 导出） | `requirements_env_gs.txt` |
| `env_threestudio` | 3.10 | 12.x | Object B 文本生成 3D | `requirements_env_threestudio.txt` |
| `env_magic123` | 3.10 | 12.x | Object C 单图生成 3D | `requirements_env_magic123.txt` |

\* 若用 2DGS 官方 `environment.yml` 创建 `env_gs`，其 Python 可能为 3.8，与课程建议 3.10 略有出入，以 2DGS 仓库为准。

创建并安装示例：

```bash
# COLMAP 专用
conda create -n env_colmap python=3.10 -y
conda activate env_colmap
pip install -r 2dgs_aigc/requirements_env_colmap.txt
conda install -c conda-forge colmap ffmpeg -y

# 2DGS 专用（推荐用官方 environment.yml 命名 env_gs）
cd 2dgs_aigc/dependences/2d-gaussian-splatting
conda env create -f environment.yml -n env_gs
conda activate env_gs
pip install -r ../../requirements_env_gs.txt
bash ../../script/install_2dgs_deps.sh

conda create -n env_threestudio python=3.10 -y && conda activate env_threestudio
pip install -r 2dgs_aigc/requirements_env_threestudio.txt

conda create -n env_magic123 python=3.10 -y && conda activate env_magic123
pip install -r 2dgs_aigc/requirements_env_magic123.txt
```

`run_objectA.sh` 会**先后** `conda activate env_colmap`（`--stage colmap`）与 `env_gs`（`--stage gs`）。`run_background.sh` 仅使用 `env_gs`。

Blender 独立安装；`run_fusion_render.sh` 用 `env_gs` 的 Python 读取 `fusion.yaml` 中的 GPU 配置。

---

## CUDA / GPU 配置

每个 `configs/*.yaml` 均包含：

```yaml
cuda:
  enable: true       # false：禁用 GPU（CUDA_VISIBLE_DEVICES=""）
  device_ids: "0"    # 物理 GPU 编号："0" / "1" / "0,1"
```

- `script/*.sh` 在 `conda activate` 后调用 `apply_cuda_from_config`，向当前 shell 注入 `CUDA_VISIBLE_DEVICES`
- Python pipeline 子进程继承该变量，并向 WandB 记录 `cuda_enable`、`cuda_device_ids`
- 训练命令模板可使用 `{cuda_device_ids}`、`{cuda_enable}` 占位符（若外部仓库需要显式传参）

并行示例：Object A 用 GPU0、Object B 用 GPU1 时，分别修改 `objectA.yaml` 与 `objectB.yaml` 的 `device_ids`。

---

## 你需要准备的外部项目（路径在配置里写）

- **COLMAP**：命令行可用（`colmap`）或指定可执行文件路径
- **2DGS**：你使用的 2D Gaussian Splatting 实现（带训练与导出脚本/命令）
- **threestudio**：可通过其官方 CLI/脚本训练并导出 mesh
- **Magic123**：可通过其官方 CLI/脚本生成并导出 mesh
- **Blender**：已安装（命令 `blender` 可用）

> 由于不同同学/不同实现的命令行略有差异，本工程将“具体命令”做成可配置的模板（见 `configs/*.yaml`）。

---

## 快速开始（建议执行顺序）

脚本已内置分环境 `conda activate`，直接执行即可（需事先创建好四个环境）：

1) 背景场景（建议先跑，或与 Object A 并行）

```bash
bash 2dgs_aigc/script/run_background.sh   # 仅 env_gs
```

2) Object A（真实物体：env_colmap → env_gs）

```bash
bash 2dgs_aigc/script/run_objectA.sh      # env_colmap + env_gs
```

3) Object B（文本生成）

```bash
bash 2dgs_aigc/script/run_objectB.sh      # env_threestudio
```

4) Object C（单图生成）

```bash
bash 2dgs_aigc/script/run_objectC.sh      # env_magic123
```

5) 统一导出

```bash
bash 2dgs_aigc/script/export_meshes.sh    # env_gs
```

6) Blender 融合渲染

```bash
bash 2dgs_aigc/script/run_fusion_render.sh
```

---

## WandB

所有 pipeline 都会读取配置项并自动初始化 WandB：

- `wandb.project` / `wandb.entity` / `wandb.mode`（online/offline/disabled）
- `wandb.tags` / `wandb.name`

本地也会写入 `2dgs_aigc/logs/`（包含你执行的命令、stdout/stderr 日志）。

---

## 需要你按自己机器修改的关键配置

按顺序检查这几个文件：

- `configs/objectA.yaml`：COLMAP 输入、2DGS 命令、`cuda`、`conda.env_colmap` / `conda.env_gs`
- `configs/background.yaml`：背景数据集路径、2DGS 训练命令模板、`cuda`
- `configs/objectB.yaml`：threestudio 项目路径/命令模板 + prompt、`cuda`
- `configs/objectC.yaml`：Magic123 项目路径/命令模板 + 输入图像、`cuda`
- `configs/fusion.yaml`：mesh 路径、摆放参数、相机轨迹、渲染与 `cuda`（Blender GPU）

---

## 复现实验建议

- 每次运行都把 `configs/*.yaml` 与 `logs/` 一起提交/打包
- ObjectA/Background 的 2DGS 训练最好固定随机种子（在命令模板里传参）
- Blender 渲染固定分辨率、帧率、采样数，确保可比对

