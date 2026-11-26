"""Headless OpenGL ES 3.2 Compute Shader Verification Harness.

Initializes an offscreen EGL + OpenGL ES 3.2 context via ctypes (`libEGL.so.1` + `libGLESv2.so.2`),
compiles `client_sdk/shaders/client_sr_v3.comp`, loads baked FP16 `CSR1` weight blobs
(or folds + segment-bakes `ClientSRNet(arch_version="v3")` checkpoints/models), dispatches
all compute passes on GPU/Mesa llvmpipe, and verifies output matches PyTorch IEEE-754
`baked_fp16` at PSNR >= 45.0 dB with zero NaN/Inf.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/root/workspace/video-chaofen")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_core.client_archs.models import ClientSRNet
from research_core.export.fold_and_verify import fold_and_verify
from research_core.export.pack_shader_weights import pack_deploy_weights
from scripts.eval_protocol.metrics import compute_psnr

SHADER_PATH = ROOT / "client_sdk" / "shaders" / "client_sr_v3.comp"

# EGL constants
EGL_DEFAULT_DISPLAY = 0
EGL_NO_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_NO_SURFACE = 0
EGL_OPENGL_ES_API = 0x30A0
EGL_OPENGL_ES3_BIT_KHR = 0x0040
EGL_RENDERABLE_TYPE = 0x3040
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056
EGL_NONE = 0x3038
EGL_CONTEXT_MAJOR_VERSION = 0x3098
EGL_CONTEXT_MINOR_VERSION = 0x30FB
EGL_PLATFORM_DEVICE_EXT = 0x313F

# GLES constants
GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02
GL_COMPUTE_SHADER = 0x91B9
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_INFO_LOG_LENGTH = 0x8B84
GL_SHADER_STORAGE_BUFFER = 0x90D2
GL_DYNAMIC_COPY = 0x88EA
GL_STATIC_DRAW = 0x88E4
GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000
GL_MAP_READ_BIT = 0x0001
GL_COPY_READ_BUFFER = 0x8F36
GL_COPY_WRITE_BUFFER = 0x8F37


class HeadlessGLES32:
    """Minimal ctypes wrapper for headless EGL + OpenGL ES 3.2 compute execution."""

    def __init__(self) -> None:
        self.egl = ctypes.CDLL("libEGL.so.1")
        self.gles = ctypes.CDLL("libGLESv2.so.2")
        self._setup_egl_protos()
        self.dpy = EGL_NO_DISPLAY
        self.ctx = EGL_NO_CONTEXT
        self.surf = EGL_NO_SURFACE
        self._init_context()
        self._setup_gl_protos()

    def _setup_egl_protos(self) -> None:
        VP = ctypes.c_void_p
        I = ctypes.c_int
        PI = ctypes.POINTER(I)

        self.egl.eglGetDisplay.argtypes = [VP]
        self.egl.eglGetDisplay.restype = VP
        self.egl.eglInitialize.argtypes = [VP, PI, PI]
        self.egl.eglInitialize.restype = I
        self.egl.eglTerminate.argtypes = [VP]
        self.egl.eglTerminate.restype = I
        self.egl.eglBindAPI.argtypes = [I]
        self.egl.eglBindAPI.restype = I
        self.egl.eglChooseConfig.argtypes = [VP, PI, ctypes.POINTER(VP), I, PI]
        self.egl.eglChooseConfig.restype = I
        self.egl.eglCreatePbufferSurface.argtypes = [VP, VP, PI]
        self.egl.eglCreatePbufferSurface.restype = VP
        self.egl.eglDestroySurface.argtypes = [VP, VP]
        self.egl.eglDestroySurface.restype = I
        self.egl.eglCreateContext.argtypes = [VP, VP, VP, PI]
        self.egl.eglCreateContext.restype = VP
        self.egl.eglDestroyContext.argtypes = [VP, VP]
        self.egl.eglDestroyContext.restype = I
        self.egl.eglMakeCurrent.argtypes = [VP, VP, VP, VP]
        self.egl.eglMakeCurrent.restype = I
        self.egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        self.egl.eglGetProcAddress.restype = VP
        self.egl.eglGetError.argtypes = []
        self.egl.eglGetError.restype = I

    def _egl_ext(self, name: bytes, restype: Any, argtypes: list[Any]):
        addr = self.egl.eglGetProcAddress(name)
        if not addr:
            return None
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

    def _try_open_display(self) -> Any:
        maj = ctypes.c_int(0)
        min_v = ctypes.c_int(0)
        dpy = self.egl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if dpy and self.egl.eglInitialize(dpy, ctypes.byref(maj), ctypes.byref(min_v)):
            return dpy

        q_devs = self._egl_ext(
            b"eglQueryDevicesEXT",
            ctypes.c_int,
            [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int)],
        )
        get_plat = self._egl_ext(
            b"eglGetPlatformDisplayEXT",
            ctypes.c_void_p,
            [ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
        )
        if q_devs and get_plat:
            num = ctypes.c_int(0)
            q_devs(0, None, ctypes.byref(num))
            if num.value > 0:
                devs = (ctypes.c_void_p * num.value)()
                q_devs(num.value, devs, ctypes.byref(num))
                for i in range(num.value):
                    cand = get_plat(EGL_PLATFORM_DEVICE_EXT, devs[i], None)
                    if cand and self.egl.eglInitialize(cand, ctypes.byref(maj), ctypes.byref(min_v)):
                        return cand
        raise RuntimeError(f"Unable to initialize headless EGL display (eglGetError=0x{self.egl.eglGetError():x})")

    def _init_context(self) -> None:
        self.dpy = self._try_open_display()
        self.egl.eglBindAPI(EGL_OPENGL_ES_API)

        for use_pbuf in (False, True):
            cfg_attrs = (
                [EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE]
                if use_pbuf
                else [EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE]
            )
            arr = (ctypes.c_int * len(cfg_attrs))(*cfg_attrs)
            cfg = ctypes.c_void_p()
            ncfg = ctypes.c_int(0)
            if not self.egl.eglChooseConfig(self.dpy, arr, ctypes.byref(cfg), 1, ctypes.byref(ncfg)) or ncfg.value == 0:
                continue

            surf = EGL_NO_SURFACE
            if use_pbuf:
                p_attrs = (ctypes.c_int * 5)(EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE)
                surf = self.egl.eglCreatePbufferSurface(self.dpy, cfg, p_attrs)
                if not surf:
                    continue

            for minor in (2, 1):
                ctx_attrs = (ctypes.c_int * 5)(EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, minor, EGL_NONE)
                ctx = self.egl.eglCreateContext(self.dpy, cfg, EGL_NO_CONTEXT, ctx_attrs)
                if ctx and self.egl.eglMakeCurrent(self.dpy, surf, surf, ctx):
                    self.surf = surf
                    self.ctx = ctx
                    return
                if ctx:
                    self.egl.eglDestroyContext(self.dpy, ctx)
            if surf:
                self.egl.eglDestroySurface(self.dpy, surf)

        raise RuntimeError(f"Unable to create EGL GLES 3.1/3.2 compute context (eglGetError=0x{self.egl.eglGetError():x})")

    def _bind_gl(self, name: str, restype: Any, argtypes: list[Any]):
        fn = getattr(self.gles, name, None)
        if fn is None:
            addr = self.egl.eglGetProcAddress(name.encode("ascii"))
            if not addr:
                raise RuntimeError(f"Missing GLES symbol: {name}")
            fn = ctypes.CFUNCTYPE(restype, *argtypes)(addr)
        else:
            fn.restype = restype
            fn.argtypes = argtypes
        return fn

    def _setup_gl_protos(self) -> None:
        VP = ctypes.c_void_p
        U = ctypes.c_uint
        I = ctypes.c_int
        F = ctypes.c_float
        SZ = ctypes.c_ssize_t
        PU = ctypes.POINTER(U)
        PI = ctypes.POINTER(I)

        self.glGetString = self._bind_gl("glGetString", ctypes.c_char_p, [U])
        self.glGetError = self._bind_gl("glGetError", U, [])
        self.glCreateShader = self._bind_gl("glCreateShader", U, [U])
        self.glShaderSource = self._bind_gl("glShaderSource", None, [U, I, ctypes.POINTER(ctypes.c_char_p), PI])
        self.glCompileShader = self._bind_gl("glCompileShader", None, [U])
        self.glGetShaderiv = self._bind_gl("glGetShaderiv", None, [U, U, PI])
        self.glGetShaderInfoLog = self._bind_gl("glGetShaderInfoLog", None, [U, I, PI, ctypes.c_char_p])
        self.glDeleteShader = self._bind_gl("glDeleteShader", None, [U])
        self.glCreateProgram = self._bind_gl("glCreateProgram", U, [])
        self.glAttachShader = self._bind_gl("glAttachShader", None, [U, U])
        self.glLinkProgram = self._bind_gl("glLinkProgram", None, [U])
        self.glGetProgramiv = self._bind_gl("glGetProgramiv", None, [U, U, PI])
        self.glGetProgramInfoLog = self._bind_gl("glGetProgramInfoLog", None, [U, I, PI, ctypes.c_char_p])
        self.glUseProgram = self._bind_gl("glUseProgram", None, [U])
        self.glDeleteProgram = self._bind_gl("glDeleteProgram", None, [U])
        self.glGetUniformLocation = self._bind_gl("glGetUniformLocation", I, [U, ctypes.c_char_p])
        self.glUniform1i = self._bind_gl("glUniform1i", None, [I, I])
        self.glUniform1ui = self._bind_gl("glUniform1ui", None, [I, U])
        self.glUniform1f = self._bind_gl("glUniform1f", None, [I, F])
        self.glGenBuffers = self._bind_gl("glGenBuffers", None, [I, PU])
        self.glDeleteBuffers = self._bind_gl("glDeleteBuffers", None, [I, PU])
        self.glBindBuffer = self._bind_gl("glBindBuffer", None, [U, U])
        self.glBufferData = self._bind_gl("glBufferData", None, [U, SZ, VP, U])
        self.glBindBufferBase = self._bind_gl("glBindBufferBase", None, [U, U, U])
        self.glCopyBufferSubData = self._bind_gl("glCopyBufferSubData", None, [U, U, SZ, SZ, SZ])
        self.glDispatchCompute = self._bind_gl("glDispatchCompute", None, [U, U, U])
        self.glMemoryBarrier = self._bind_gl("glMemoryBarrier", None, [U])
        self.glFinish = self._bind_gl("glFinish", None, [])
        self.glMapBufferRange = self._bind_gl("glMapBufferRange", VP, [U, SZ, SZ, U])
        self.glUnmapBuffer = self._bind_gl("glUnmapBuffer", ctypes.c_ubyte, [U])

    def gl_str(self, enum_id: int) -> str:
        raw = self.glGetString(enum_id)
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    def compile_compute_program(self, source: str) -> int:
        sh = self.glCreateShader(GL_COMPUTE_SHADER)
        src_bytes = source.encode("utf-8")
        src_ptr = ctypes.c_char_p(src_bytes)
        src_len = ctypes.c_int(len(src_bytes))
        self.glShaderSource(sh, 1, ctypes.byref(src_ptr), ctypes.byref(src_len))
        self.glCompileShader(sh)

        ok = ctypes.c_int(0)
        self.glGetShaderiv(sh, GL_COMPILE_STATUS, ctypes.byref(ok))
        if ok.value != 1:
            log_len = ctypes.c_int(0)
            self.glGetShaderiv(sh, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
            buf = ctypes.create_string_buffer(max(log_len.value, 512))
            self.glGetShaderInfoLog(sh, len(buf), None, buf)
            self.glDeleteShader(sh)
            raise RuntimeError(f"Compute shader compile failed:\n{buf.value.decode('utf-8', errors='replace')}")

        prog = self.glCreateProgram()
        self.glAttachShader(prog, sh)
        self.glLinkProgram(prog)
        self.glDeleteShader(sh)

        link_ok = ctypes.c_int(0)
        self.glGetProgramiv(prog, GL_LINK_STATUS, ctypes.byref(link_ok))
        if link_ok.value != 1:
            log_len = ctypes.c_int(0)
            self.glGetProgramiv(prog, GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
            buf = ctypes.create_string_buffer(max(log_len.value, 512))
            self.glGetProgramInfoLog(prog, len(buf), None, buf)
            self.glDeleteProgram(prog)
            raise RuntimeError(f"Compute program link failed:\n{buf.value.decode('utf-8', errors='replace')}")
        return int(prog)

    def create_ssbo(self, nbytes: int, data: bytes | None = None, usage: int = GL_DYNAMIC_COPY) -> int:
        buf_id = ctypes.c_uint(0)
        self.glGenBuffers(1, ctypes.byref(buf_id))
        self.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_id.value)
        ptr = ctypes.c_char_p(data) if data is not None else None
        self.glBufferData(GL_SHADER_STORAGE_BUFFER, nbytes, ptr, usage)
        self.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return int(buf_id.value)

    def delete_ssbos(self, buf_ids: list[int]) -> None:
        if not buf_ids:
            return
        arr = (ctypes.c_uint * len(buf_ids))(*buf_ids)
        self.glDeleteBuffers(len(buf_ids), arr)

    def copy_ssbo(self, src_id: int, dst_id: int, nbytes: int) -> None:
        self.glBindBuffer(GL_COPY_READ_BUFFER, src_id)
        self.glBindBuffer(GL_COPY_WRITE_BUFFER, dst_id)
        self.glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, nbytes)
        self.glBindBuffer(GL_COPY_READ_BUFFER, 0)
        self.glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

    def read_ssbo_f32(self, buf_id: int, count: int) -> np.ndarray:
        self.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf_id)
        nbytes = count * 4
        ptr = self.glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, nbytes, GL_MAP_READ_BIT)
        if not ptr:
            raise RuntimeError(f"glMapBufferRange returned NULL (glGetError=0x{self.glGetError():x})")
        raw = ctypes.string_at(ptr, nbytes)
        self.glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        self.glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self) -> None:
        if self.dpy:
            self.egl.eglMakeCurrent(self.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
            if self.ctx:
                self.egl.eglDestroyContext(self.dpy, self.ctx)
            if self.surf:
                self.egl.eglDestroySurface(self.dpy, self.surf)
            self.egl.eglTerminate(self.dpy)
            self.dpy = EGL_NO_DISPLAY


def _load_csr1_and_bake(bin_path: Path, segment_prior: torch.Tensor) -> tuple[dict[str, Any], ClientSRNet]:
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        if magic != b"CSR1":
            raise ValueError(f"{bin_path} magic is {magic!r}, expected b'CSR1'")
        meta_len = struct.unpack("<I", f.read(4))[0]
        meta = json.loads(f.read(meta_len).decode("utf-8"))
        blob = f.read()

    tier = str(meta["tier"])
    scale = float(meta["scale"])
    use_prior = bool(meta["use_prior"])
    arch_v = str(meta.get("arch_version", "v3"))
    if arch_v != "v3":
        raise ValueError(f"{bin_path} has arch_version={arch_v!r}; shader harness validates V3 CSR1 blobs")

    model = ClientSRNet(
        tier=tier,
        scale=scale,
        use_prior=use_prior,
        use_rep=True,
        deploy=True,
        arch_version=arch_v,
    ).eval()

    sd_in: dict[str, torch.Tensor] = {}
    for t in meta["tensors"]:
        off = int(t["offset"])
        nbytes = int(t["nbytes"])
        shape = tuple(int(d) for d in t["shape"])
        arr = np.frombuffer(blob[off : off + nbytes], dtype=np.float16).astype(np.float32).copy()
        sd_in[t["name"]] = torch.from_numpy(arr.reshape(shape)) if shape else torch.tensor(float(arr[0]), dtype=torch.float32)

    model.load_state_dict(sd_in, strict=True)
    pdim = int(meta.get("prior_dim", 7))
    baked = model.bake_segment_prior(segment_prior[:pdim] if use_prior else None).eval()
    return meta, baked


def _pack_baked_weights_ssbo(baked: ClientSRNet) -> tuple[bytes, dict[str, int]]:
    """Pack baked V3 model state_dict into contiguous FP16 bytes aligned to 4B + element offsets."""
    chunks: list[bytes] = []
    half_offsets: dict[str, int] = {}
    cur_half_idx = 0
    for name, tensor in baked.state_dict().items():
        half_offsets[name] = cur_half_idx
        t16 = tensor.detach().cpu().to(torch.float16).contiguous().view(-1)
        chunks.append(t16.numpy().tobytes())
        cur_half_idx += int(t16.numel())

    raw = b"".join(chunks)
    if len(raw) % 4 != 0:
        raw += b"\x00" * (4 - (len(raw) % 4))
    return raw, half_offsets


def run_gles_shader_for_baked_model(
    gl: HeadlessGLES32,
    prog: int,
    baked: ClientSRNet,
    lr_np: np.ndarray,
) -> np.ndarray:
    """Execute full V3 pipeline on GPU/GLES compute shader using packed FP16 WeightBuf."""
    lr_h, lr_w = int(lr_np.shape[0]), int(lr_np.shape[1])
    out_h = int(round(lr_h * baked.scale))
    out_w = int(round(lr_w * baked.scale))
    uh, uw = lr_h // 2, lr_w // 2
    r_tail = 4 if abs(baked.scale - 2.0) < 1e-4 else 3

    weight_bytes, offs = _pack_baked_weights_ssbo(baked)

    gl.glUseProgram(prog)
    locs = {
        u: gl.glGetUniformLocation(prog, u.encode("ascii"))
        for u in (
            "u_op", "u_in_c", "u_in_h", "u_in_w", "u_out_c", "u_out_h", "u_out_w",
            "u_w_off", "u_b_off", "u_pb_off", "u_act_off", "u_has_pb", "u_has_act",
            "u_scale_r", "u_add_b", "u_alpha", "u_beta", "u_delta",
        )
    }

    max_ch_hw = max(80 * uh * uw, 48 * lr_h * lr_w, out_h * out_w)
    scratch_bytes = max_ch_hw * 4

    w_ssbo = gl.create_ssbo(len(weight_bytes), weight_bytes, GL_STATIC_DRAW)
    x_ssbo = gl.create_ssbo(lr_h * lr_w * 4, np.ascontiguousarray(lr_np, dtype=np.float32).tobytes(), GL_STATIC_DRAW)
    base_ssbo = gl.create_ssbo(out_h * out_w * 4)
    res_ssbo = gl.create_ssbo(out_h * out_w * 4)
    f_head_ssbo = gl.create_ssbo(scratch_bytes)
    f_mid_ssbo = gl.create_ssbo(scratch_bytes)
    s2_ssbo = gl.create_ssbo(scratch_bytes)
    ping_ssbo = gl.create_ssbo(scratch_bytes)
    pong_ssbo = gl.create_ssbo(scratch_bytes)
    created = [w_ssbo, x_ssbo, base_ssbo, res_ssbo, f_head_ssbo, f_mid_ssbo, s2_ssbo, ping_ssbo, pong_ssbo]

    def _dispatch(
        op: int,
        in_c: int, in_h: int, in_w: int,
        out_c: int, oh: int, ow: int,
        buf_a: int, buf_b: int, dst: int,
        w_off: int = 0, b_off: int = 0, pb_off: int = 0, act_off: int = 0,
        has_pb: int = 0, has_act: int = 0,
        scale_r: int = 1, add_b: int = 0,
        alpha: float = 0.0, beta: float = 0.0, delta: float = 0.0,
    ) -> None:
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w_ssbo)
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buf_a)
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buf_b)
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, dst)

        gl.glUniform1i(locs["u_op"], op)
        gl.glUniform1i(locs["u_in_c"], in_c)
        gl.glUniform1i(locs["u_in_h"], in_h)
        gl.glUniform1i(locs["u_in_w"], in_w)
        gl.glUniform1i(locs["u_out_c"], out_c)
        gl.glUniform1i(locs["u_out_h"], oh)
        gl.glUniform1i(locs["u_out_w"], ow)
        gl.glUniform1ui(locs["u_w_off"], w_off)
        gl.glUniform1ui(locs["u_b_off"], b_off)
        gl.glUniform1ui(locs["u_pb_off"], pb_off)
        gl.glUniform1ui(locs["u_act_off"], act_off)
        gl.glUniform1i(locs["u_has_pb"], has_pb)
        gl.glUniform1i(locs["u_has_act"], has_act)
        gl.glUniform1i(locs["u_scale_r"], scale_r)
        gl.glUniform1i(locs["u_add_b"], add_b)
        gl.glUniform1f(locs["u_alpha"], alpha)
        gl.glUniform1f(locs["u_beta"], beta)
        gl.glUniform1f(locs["u_delta"], delta)

        gx = (ow + 15) // 16
        gy = (oh + 15) // 16
        gl.glDispatchCompute(gx, gy, 1)
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _conv(
        prefix: str, act_prefix: str | None,
        in_c: int, out_c: int, h: int, w: int,
        in_ssbo: int, out_ssbo: int, has_pb: bool = False,
    ) -> None:
        _dispatch(
            op=2,
            in_c=in_c, in_h=h, in_w=w, out_c=out_c, oh=h, ow=w,
            buf_a=in_ssbo, buf_b=x_ssbo, dst=out_ssbo,
            w_off=offs[f"{prefix}.base.reparam_conv.weight"],
            b_off=offs[f"{prefix}.base.reparam_conv.bias"],
            pb_off=offs["phase_bias"] if has_pb else 0,
            act_off=offs[f"{act_prefix}.weight"] if act_prefix is not None else 0,
            has_pb=1 if has_pb else 0,
            has_act=1 if act_prefix is not None else 0,
        )

    try:
        # 0. Bicubic base (a=-0.75)
        _dispatch(op=0, in_c=1, in_h=lr_h, in_w=lr_w, out_c=1, oh=out_h, ow=out_w, buf_a=x_ssbo, buf_b=x_ssbo, dst=base_ssbo)

        mid_idx = len(baked.body) // 2
        if baked.is_hourglass:
            stem_ch = int(baked.stem1.base.reparam_conv.out_channels)
            mid_ch = int(baked.down.base.reparam_conv.out_channels)
            _conv("stem1", "act_stem1", 1, stem_ch, lr_h, lr_w, x_ssbo, pong_ssbo, has_pb=False)
            _conv("stem2", "act_stem2", stem_ch, stem_ch, lr_h, lr_w, pong_ssbo, s2_ssbo, has_pb=False)
            _dispatch(op=1, in_c=stem_ch, in_h=lr_h, in_w=lr_w, out_c=4 * stem_ch, oh=uh, ow=uw, buf_a=s2_ssbo, buf_b=x_ssbo, dst=ping_ssbo)
            _conv("down", "act_down", 4 * stem_ch, mid_ch, uh, uw, ping_ssbo, f_head_ssbo, has_pb=True)
            cur_in = f_head_ssbo
            cur_out = ping_ssbo
            for idx in range(len(baked.body)):
                _conv(f"body.{idx}", f"body_acts.{idx}", mid_ch, mid_ch, uh, uw, cur_in, cur_out, has_pb=False)
                if idx == mid_idx:
                    gl.copy_ssbo(cur_out, f_mid_ssbo, mid_ch * uh * uw * 4)
                cur_in = cur_out
                cur_out = pong_ssbo if cur_out == ping_ssbo else ping_ssbo

            _dispatch(
                op=3,
                in_c=mid_ch, in_h=uh, in_w=uw, out_c=mid_ch, oh=uh, ow=uw,
                buf_a=f_head_ssbo, buf_b=f_mid_ssbo, dst=cur_in,
                alpha=float(baked.skip_alpha.item()), beta=float(baked.skip_beta.item()),
            )

            if baked.hg_has_up:
                up_out_c = int(baked.up.base.reparam_conv.out_channels)
                _conv("up", "act_up", mid_ch, up_out_c, uh, uw, cur_in, cur_out, has_pb=False)
                _dispatch(op=4, in_c=up_out_c, in_h=uh, in_w=uw, out_c=stem_ch, oh=lr_h, ow=lr_w, buf_a=cur_out, buf_b=s2_ssbo, dst=f_head_ssbo, scale_r=2, add_b=1)
                _conv("refine", "act_refine", stem_ch, stem_ch, lr_h, lr_w, f_head_ssbo, ping_ssbo, has_pb=False)
                _conv("tail", None, stem_ch, 4, lr_h, lr_w, ping_ssbo, pong_ssbo, has_pb=False)
                _dispatch(op=4, in_c=4, in_h=lr_h, in_w=lr_w, out_c=1, oh=out_h, ow=out_w, buf_a=pong_ssbo, buf_b=x_ssbo, dst=res_ssbo, scale_r=2, add_b=0)
            else:
                if getattr(baked, "hg_has_hr_skip", False):
                    _dispatch(
                        op=1,
                        in_c=stem_ch, in_h=lr_h, in_w=lr_w,
                        out_c=4 * stem_ch, oh=uh, ow=uw,
                        buf_a=s2_ssbo, buf_b=x_ssbo, dst=cur_out,
                    )
                    _dispatch(
                        op=3,
                        in_c=mid_ch, in_h=uh, in_w=uw,
                        out_c=mid_ch, oh=uh, ow=uw,
                        buf_a=cur_out, buf_b=cur_out, dst=cur_in,
                        alpha=float(baked.skip_hr.item()), beta=0.0,
                    )
                _conv("tail", None, mid_ch, 9, uh, uw, cur_in, cur_out, has_pb=False)
                _dispatch(op=4, in_c=9, in_h=uh, in_w=uw, out_c=1, oh=out_h, ow=out_w, buf_a=cur_out, buf_b=x_ssbo, dst=res_ssbo, scale_r=3, add_b=0)
        else:
            mid_ch = int(baked.head.base.reparam_conv.out_channels)
            _dispatch(op=1, in_c=1, in_h=lr_h, in_w=lr_w, out_c=4, oh=uh, ow=uw, buf_a=x_ssbo, buf_b=x_ssbo, dst=ping_ssbo)
            _conv("head", "act_head", 4, mid_ch, uh, uw, ping_ssbo, f_head_ssbo, has_pb=True)
            cur_in = f_head_ssbo
            cur_out = ping_ssbo
            for idx in range(len(baked.body)):
                _conv(f"body.{idx}", f"body_acts.{idx}", mid_ch, mid_ch, uh, uw, cur_in, cur_out, has_pb=False)
                if idx == mid_idx:
                    gl.copy_ssbo(cur_out, f_mid_ssbo, mid_ch * uh * uw * 4)
                cur_in = cur_out
                cur_out = pong_ssbo if cur_out == ping_ssbo else ping_ssbo

            _dispatch(
                op=3,
                in_c=mid_ch, in_h=uh, in_w=uw, out_c=mid_ch, oh=uh, ow=uw,
                buf_a=f_head_ssbo, buf_b=f_mid_ssbo, dst=cur_in,
                alpha=float(baked.skip_alpha.item()), beta=float(baked.skip_beta.item()),
            )
            _conv("tail", None, mid_ch, r_tail * r_tail, uh, uw, cur_in, cur_out, has_pb=False)
            _dispatch(op=4, in_c=r_tail * r_tail, in_h=uh, in_w=uw, out_c=1, oh=out_h, ow=out_w, buf_a=cur_out, buf_b=x_ssbo, dst=res_ssbo, scale_r=r_tail, add_b=0)

        if "seam_lut.table" in offs:
            _dispatch(
                op=5,
                in_c=1, in_h=lr_h, in_w=lr_w, out_c=1, oh=out_h, ow=out_w,
                buf_a=x_ssbo, buf_b=x_ssbo, dst=res_ssbo,
                w_off=offs["seam_lut.table"], b_off=offs["seam_lut.seam_mask"],
            )

        delta_seg = float(baked.trust_clamp.baked_delta.item())
        _dispatch(
            op=6,
            in_c=1, in_h=lr_h, in_w=lr_w, out_c=1, oh=out_h, ow=out_w,
            buf_a=base_ssbo, buf_b=x_ssbo, dst=res_ssbo,
            delta=delta_seg,
        )

        gl.glFinish()
        out_flat = gl.read_ssbo_f32(res_ssbo, out_h * out_w)
        return out_flat.reshape(out_h, out_w)
    finally:
        gl.delete_ssbos(created)


def verify_bin(
    gl: HeadlessGLES32,
    prog: int,
    bin_path: Path,
    patch_size: int = 64,
    seed: int = 123,
) -> dict[str, Any]:
    prior = torch.tensor([0.55, 1.0, 1.0, 0.42, 1.0, 0.35, 0.60], dtype=torch.float32)
    meta, baked = _load_csr1_and_bake(bin_path, prior)

    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, patch_size), np.linspace(0.0, 1.0, patch_size), indexing="ij")
    pattern = 0.45 + 0.25 * np.sin(14.0 * xx) * np.cos(10.0 * yy) + 0.15 * (xx > 0.45).astype(np.float32)
    lr_np = np.clip(pattern + 0.03 * rng.standard_normal((patch_size, patch_size)), 0.05, 0.95).astype(np.float32)

    gles_out = run_gles_shader_for_baked_model(gl, prog, baked, lr_np)
    if not np.all(np.isfinite(gles_out)):
        raise AssertionError(f"Non-finite output from GLES shader for {bin_path.name}")

    lr_t = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        ref_model = baked.half()
        pt_out = ref_model(lr_t.half(), None).float()[0, 0].numpy()

    psnr_db = compute_psnr(gles_out, pt_out, data_range=1.0)
    max_abs = float(np.max(np.abs(gles_out - pt_out)))
    if not math.isfinite(psnr_db) or psnr_db < 45.0:
        raise AssertionError(f"GLES vs PyTorch alignment failed for {bin_path.name}: PSNR={psnr_db:.2f} dB (< 45.0 dB)")

    return {
        "bin": str(bin_path),
        "tier": meta["tier"],
        "scale": meta["scale"],
        "use_prior": meta["use_prior"],
        "patch_lr": patch_size,
        "patch_sr": list(gles_out.shape),
        "gles_vs_pytorch_fp16_psnr_db": round(psnr_db, 2),
        "max_abs_err": round(max_abs, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GLES 3.2 compute shader against PyTorch baked FP16")
    parser.add_argument("--bin-path", type=Path, default=None, help="Verify a single CSR1 .bin file")
    parser.add_argument("--bin-dir", type=Path, default=None, help="Verify all V3 CSR1 .bin files in directory")
    parser.add_argument("--patch-size", type=int, default=64, help="Even LR test patch size")
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()

    if args.patch_size % 2 != 0:
        raise ValueError("--patch-size must be even for PixelUnshuffle(2)")

    gl = HeadlessGLES32()
    try:
        vendor = gl.gl_str(GL_VENDOR)
        renderer = gl.gl_str(GL_RENDERER)
        version = gl.gl_str(GL_VERSION)
        print(f"[GLES] Initialized headless compute context: {version} | {vendor} | {renderer}", flush=True)
        prog = gl.compile_compute_program(SHADER_PATH.read_text(encoding="utf-8"))
        print(f"[GLES] Compiled & linked {SHADER_PATH.name} OK (program={prog})", flush=True)

        results: list[dict[str, Any]] = []
        if args.bin_path is not None:
            res = verify_bin(gl, prog, args.bin_path, patch_size=args.patch_size)
            results.append(res)
            print(f"[PASS] {Path(res['bin']).name} (tier={res['tier']}, x{res['scale']}): PSNR={res['gles_vs_pytorch_fp16_psnr_db']} dB, max_err={res['max_abs_err']}", flush=True)
        elif args.bin_dir is not None:
            for bp in sorted(args.bin_dir.glob("*.bin")):
                with open(bp, "rb") as f:
                    magic = f.read(4)
                    if magic != b"CSR1":
                        continue
                    mlen = struct.unpack("<I", f.read(4))[0]
                    meta = json.loads(f.read(mlen).decode("utf-8"))
                if meta.get("arch_version") != "v3":
                    continue
                res = verify_bin(gl, prog, bp, patch_size=args.patch_size)
                results.append(res)
                print(f"[PASS] {bp.name} (tier={res['tier']}, x{res['scale']}): PSNR={res['gles_vs_pytorch_fp16_psnr_db']} dB, max_err={res['max_abs_err']}", flush=True)
            if not results:
                raise FileNotFoundError(f"No V3 CSR1 .bin files found in {args.bin_dir}")
        else:
            # Self-contained verification over all 6 V3 configs (540p & 720p x L/S/P)
            with tempfile.TemporaryDirectory() as tmpd:
                dp = Path(tmpd)
                exp_d = dp / "exp"
                for res_name, sc in [("540p", 2.0), ("720p", 1.5)]:
                    for tier in ("L", "S", "P"):
                        torch.manual_seed(42)
                        m = ClientSRNet(tier=tier, scale=sc, use_prior=True, use_rep=True, deploy=False, arch_version="v3").eval()
                        with torch.no_grad():
                            for p in m.parameters():
                                if p.ndim >= 2:
                                    p.add_(0.015 * torch.randn_like(p))
                        raw_p = dp / f"client_{tier}_{res_name}_prior.pt"
                        bin_p = dp / f"client_{tier}_{res_name}_prior.bin"
                        torch.save(
                            {
                                "model": m.state_dict(),
                                "tier": tier,
                                "scale": sc,
                                "use_prior": True,
                                "arch_version": "v3",
                                "prior_dim": 7,
                            },
                            raw_p,
                        )
                        rep_f = fold_and_verify(raw_p, exp_d)
                        pack_deploy_weights(Path(rep_f["deploy_ckpt"]), bin_p, export_int8=False)
                        res = verify_bin(gl, prog, bin_p, patch_size=args.patch_size)
                        results.append(res)
                        print(f"[PASS] {res_name} tier={tier}: PSNR={res['gles_vs_pytorch_fp16_psnr_db']} dB, max_err={res['max_abs_err']}", flush=True)

        payload = {
            "gl_version": version,
            "gl_vendor": vendor,
            "gl_renderer": renderer,
            "shader": str(SHADER_PATH),
            "verified": results,
        }
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[GLES] Report saved -> {args.report_out}", flush=True)
        gl.glDeleteProgram(prog)
    finally:
        gl.close()


if __name__ == "__main__":
    main()
