from __future__ import annotations

# Blender 运行方式：
# blender -b -P 2dgs_aigc/script/blender_fusion.py -- --config 2dgs_aigc/configs/fusion.yaml

import argparse
import math
from pathlib import Path
from typing import Any

import bpy

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for b in list(block):
            try:
                block.remove(b)  # type: ignore[attr-defined]
            except Exception:
                pass


def import_mesh(path: Path, name: str):
    ext = path.suffix.lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(path))
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"不支持的格式: {path}")

    # import 后，最新导入的对象会被选中
    imported = [o for o in bpy.context.selected_objects]
    if not imported:
        raise RuntimeError(f"导入失败: {path}")
    obj = imported[0]
    obj.name = name
    return obj


def set_transform(obj, loc, rot_deg, scale):
    obj.location = loc
    obj.rotation_euler = (deg2rad(rot_deg[0]), deg2rad(rot_deg[1]), deg2rad(rot_deg[2]))
    obj.scale = scale


def set_smooth(obj, smooth: bool = True):
    if obj.type != "MESH":
        return
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if smooth:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()
    obj.select_set(False)


def setup_light(cfg: dict[str, Any]):
    lighting = cfg.get("lighting", {}) or {}

    ambient = float(lighting.get("ambient_strength", 0.2))
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = ambient

    if lighting.get("use_hdri", False) and lighting.get("hdri_path"):
        hdri_path = Path(lighting["hdri_path"]).expanduser().resolve()
        if hdri_path.exists():
            env_tex = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
            env_tex.image = bpy.data.images.load(str(hdri_path))
            world.node_tree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])

    sun = (lighting.get("sun", {}) or {}) if isinstance(lighting.get("sun", {}), dict) else {}
    if sun.get("enable", True):
        light_data = bpy.data.lights.new(name="Sun", type="SUN")
        light_data.energy = float(sun.get("strength", 2.5))
        light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        rot = sun.get("rotation_euler_deg", [45, 0, 35])
        light_obj.rotation_euler = (deg2rad(rot[0]), deg2rad(rot[1]), deg2rad(rot[2]))


def setup_camera_and_animation(cfg: dict[str, Any]):
    cam_cfg = cfg.get("camera", {}) or {}
    target = cam_cfg.get("target", [0.0, 0.0, 0.8])
    radius = float(cam_cfg.get("radius", 4.0))
    height = float(cam_cfg.get("height", 1.6))

    cam_data = bpy.data.cameras.new(name="Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    empty = bpy.data.objects.new("LookAt", None)
    empty.location = target
    bpy.context.collection.objects.link(empty)

    track = cam.constraints.new(type="TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # 动画：绕 target 做圆周运动
    render = cfg.get("render", {}) or {}
    f0 = int(render.get("frame_start", 1))
    f1 = int(render.get("frame_end", 240))

    for i, frame in enumerate([f0, f1]):
        t = i  # 0 -> 1
        theta = 2.0 * math.pi * t
        x = target[0] + radius * math.cos(theta)
        y = target[1] + radius * math.sin(theta)
        z = target[2] + height
        cam.location = (x, y, z)
        cam.keyframe_insert(data_path="location", frame=frame)

    return cam


def setup_render(cfg: dict[str, Any], video_out: Path):
    render = cfg.get("render", {}) or {}
    engine = str(render.get("engine", "CYCLES")).upper()

    scene = bpy.context.scene
    scene.render.engine = engine

    res = render.get("resolution", {}) or {}
    scene.render.resolution_x = int(res.get("width", 1280))
    scene.render.resolution_y = int(res.get("height", 720))
    scene.render.fps = int(render.get("fps", 24))
    scene.frame_start = int(render.get("frame_start", 1))
    scene.frame_end = int(render.get("frame_end", 240))

    if engine == "CYCLES":
        cycles = render.get("cycles", {}) or {}
        scene.cycles.samples = int(cycles.get("samples", 128))
        if bool(cycles.get("denoise", True)):
            scene.cycles.use_denoising = True

        cuda_cfg = cfg.get("cuda", {}) or {}
        cuda_enable = bool(cuda_cfg.get("enable", True))
        device = str(render.get("device", "GPU")).upper()
        if not cuda_enable:
            device = "CPU"

        prefs = bpy.context.preferences
        cprefs = prefs.addons["cycles"].preferences if "cycles" in prefs.addons else None
        if device == "GPU" and cprefs is not None:
            try:
                cprefs.compute_device_type = "CUDA"
                scene.cycles.device = "GPU"
            except Exception:
                scene.cycles.device = "CPU"
        else:
            scene.cycles.device = "CPU"

    # 输出为 mp4
    ensure_dir(video_out.parent)
    scene.render.filepath = str(video_out)
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, _ = ap.parse_known_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    paths = cfg.get("paths", {}) or {}
    video_out = Path(paths.get("video_out", "output/videos/final.mp4")).expanduser().resolve()
    blend_out = Path(paths.get("blender_scene_out", "assets/blender/fusion_scene.blend")).expanduser().resolve()

    clear_scene()

    # 导入对象
    for obj_cfg in cfg.get("objects", []) or []:
        name = obj_cfg.get("name", "obj")
        path = Path(obj_cfg.get("path")).expanduser().resolve()
        obj = import_mesh(path, name)
        tf = obj_cfg.get("transform", {}) or {}
        set_transform(
            obj,
            tf.get("location", [0.0, 0.0, 0.0]),
            tf.get("rotation_euler_deg", [0.0, 0.0, 0.0]),
            tf.get("scale", [1.0, 1.0, 1.0]),
        )
        shading = obj_cfg.get("shading", {}) or {}
        set_smooth(obj, bool(shading.get("smooth", True)))

    setup_light(cfg)
    setup_camera_and_animation(cfg)
    setup_render(cfg, video_out)

    ensure_dir(blend_out.parent)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_out))

    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()

