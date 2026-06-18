# 2dgs_aigc — 多源 3D 资产生产子工程

本目录为仓库核心代码，提供配置驱动的多源 3D 资产生成与 Blender 融合渲染管线。完整使用说明（环境、数据、训练、测试命令）见仓库根目录 [`../README.md`](../README.md)。

## 演示视频

最终融合漫游渲染：[`rendering.mp4`](rendering.mp4)

## 目录说明

```
2dgs_aigc/
├── configs/          # YAML 配置（路径、GPU、训练命令模板）
├── data/
│   ├── raw/          # 用户放置的原始输入
│   └── processed/    # 训练中间结果（gitignore）
├── assets/           # 导出的 mesh / Blender 场景
├── script/           # Shell 入口脚本
├── src/              # Python pipeline 封装
│   ├── reconstruction/   # run_objectA, run_background
│   ├── text_to_3d/       # run_objectB（SDI）
│   ├── image_to_3d/      # run_objectC（Magic123）
│   ├── fusion/           # export_assets
│   └── utils/            # 配置解析、CUDA、WandB、日志
├── dependences/      # 第三方 git 仓库挂载点（需 clone）
├── requirements_env_*.txt
└── rendering.mp4     # 演示输出
```

## `dependences/` 克隆清单

```bash
cd dependences

git clone https://github.com/hbb1/2d-gaussian-splatting.git
git clone https://github.com/threestudio-project/threestudio.git
git clone https://github.com/colmap/colmap.git          # 或 conda install colmap
git clone https://github.com/guochengqian/Magic123.git  # 可选
```

克隆 2DGS / threestudio 后务必执行 `git submodule update --init --recursive`。

## 环境与依赖文件

| 环境 | requirements | 用途 |
|------|-------------|------|
| `env_colmap` | `requirements_env_colmap.txt` | Object A COLMAP |
| `env_gs` | `requirements_env_gs.txt` + `dependences/2d-gaussian-splatting/environment.yml` | 2DGS |
| `env_sdi` | `requirements_env_sdi.txt` + `dependences/threestudio/requirements.txt` | SDI |
| `env_magic123` | `requirements_env_magic123.txt` + threestudio requirements | Magic123 |

2DGS CUDA 扩展安装：

```bash
bash script/install_2dgs_deps.sh
```

## 快速命令

```bash
# 在仓库根目录 CV_HW3/ 下执行

bash 2dgs_aigc/script/run_background.sh
bash 2dgs_aigc/script/run_objectA.sh
bash 2dgs_aigc/script/run_objectB.sh
bash 2dgs_aigc/script/run_objectC.sh
bash 2dgs_aigc/script/export_meshes.sh
bash 2dgs_aigc/script/run_fusion_render.sh
```

## `src/` 模块与配置对应关系

| Python 入口 | 配置 | 外部依赖 |
|-------------|------|----------|
| `src.reconstruction.run_objectA` | `configs/objectA.yaml` | colmap, 2d-gaussian-splatting |
| `src.reconstruction.run_background` | `configs/background.yaml` | 2d-gaussian-splatting |
| `src.text_to_3d.run_objectB` | `configs/objectB.yaml` | threestudio + SD 2.1 |
| `src.image_to_3d.run_objectC` | `configs/objectC.yaml` | threestudio Magic123 + SD 1.5 |
| `script/blender_fusion.py` | `configs/fusion.yaml` | Blender |

配置中的 `{work_dir}`、`{gpu}` 等占位符由 `src/utils/` 在运行时展开为绝对路径与实际命令。

## 预训练权重

- **SDI（Object B）**：在 `dependences/threestudio/configs/sdi.yaml` 设置本地 SD 2.1 路径（`pretrained_model_name_or_path`）。
- **Magic123（Object C）**：使用 HuggingFace 上的 `runwayml/stable-diffusion-v1-5` 与 Zero123，首次训练自动下载。

## 脚本索引

| 脚本 | 功能 |
|------|------|
| `run_objectA.sh` | COLMAP → 2DGS → mesh 导出 |
| `run_background.sh` | 背景 2DGS 训练 + mesh |
| `run_objectB.sh` | SDI 训练 + 高精度导出 |
| `run_objectB_export.sh` | 仅重新导出 Object B mesh |
| `run_objectC.sh` | Magic123 coarse + refine + 导出 |
| `export_meshes.sh` | 汇总 mesh 到 Blender 目录 |
| `run_fusion_render.sh` | Blender 融合渲染 |
| `plot_training_curves.py` | 从 metrics / 日志绘制 Loss |
| `install_2dgs_deps.sh` | 编译 2DGS CUDA 子模块 |
