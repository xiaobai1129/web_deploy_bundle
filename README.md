# CodecSR V3.5 — Web 端本地超分部署开发包（`web_deploy_bundle`）

本文件夹包含了在**本地电脑（无需任何云端 GPU 或 Python/PyTorch 后端）**独立开发、调试并上线 **CodecSR V3.5 纯前端视频超分网站** 所需的全部正式版模型权重、参考着色器、零依赖核烘焙参考实现、精度对拍测试集与内置演示视频。

---

## 1. 目录结构总览

```text
web_deploy_bundle/
├── README.md                                  # 本文档：完整 Web 架构、二进制协议、核烘焙、WGSL 移植与对拍指南
├── weights/                                   # V3.5 正式版 FP16 二进制权重包（共 12 个模型，总计约 1.85 MB）
│   ├── client_L_540p_prior.bin                # 143 KB | 极速档 (L) 540p->1080p (×2.0) 带 7D 动态先验 + Seam3DLUT
│   ├── client_L_540p_noprior.bin              #  39 KB | 极速档 (L) 540p->1080p (×2.0) 无先验静态版
│   ├── client_L_720p_prior.bin                # 133 KB | 极速档 (L) 720p->1080p (×1.5) 带 7D 动态先验 + Seam3DLUT
│   ├── client_L_720p_noprior.bin              #  37 KB | 极速档 (L) 720p->1080p (×1.5) 无先验静态版
│   ├── client_S_540p_prior.bin                # 290 KB | 标准档 (S) 540p->1080p (×2.0) 带 7D 动态先验 + Seam3DLUT
│   ├── client_S_540p_noprior.bin              #  96 KB | 标准档 (S) 540p->1080p (×2.0) 无先验静态版
│   ├── client_S_720p_prior.bin                # 210 KB | 标准档 (S) 720p->1080p (×1.5) 带 7D 动态先验 + Seam3DLUT
│   ├── client_S_720p_noprior.bin              #  73 KB | 标准档 (S) 720p->1080p (×1.5) 无先验静态版
│   ├── client_P_540p_prior.bin                # 408 KB | 旗舰档 (P) 540p->1080p (×2.0) 双尺度沙漏架构 + 7D 先验
│   ├── client_P_720p_prior.bin                # 396 KB | 旗舰档 (P) 720p->1080p (×1.5) 双尺度沙漏架构 + 7D 先验
│   ├── chroma_S_540p.bin                      #   5 KB | 可选：YUV420 色度引导恢复头 (×2.0, CHR1 格式)
│   ├── chroma_S_720p.bin                      #  23 KB | 可选：YUV420 色度引导恢复头 (×1.5, CHR1 格式)
│   ├── shader_pack_summary.json               # 权重体积与量化精度摘要
│   ├── shader_verify.json                     # GLES Compute Shader vs PyTorch FP16 对拍报告 (>73.7~75.2 dB)
│   └── export_fold_summary.json               # 模型算子重参数化折叠与 GMACs 统计报告
├── reference_code/                            # 供翻译 TypeScript 与 WGSL 的核心参考源码
│   ├── standalone_numpy_reference.py          # ★ 最重要参考：零 PyTorch 依赖、仅 ~240 行纯数组运算的解析 + 7D 先验 + 核烘焙实现
│   ├── client_sr_v3.comp                      # ★ 最重要参考：GLSL ES 3.2 Compute Shader（共 7 个算子 op=0..6）
│   ├── verify_compute_shader.py               # GPU SSBO 绑定与全模型逐层 Dispatch 调度顺序的 1:1 参考（重点看 375~560 行）
│   ├── models.py                              # V3.5 完整网络结构源码（ClientSRNet / Seam3DLUT / TrustClamp / ChromaHead）
│   ├── fold_and_verify.py                     # 训练态折叠为推理态的验证脚本
│   └── pack_shader_weights.py                 # CSR1 二进制权重打包格式定义
├── golden_vectors/                            # 本地前端自动化像素级对拍测试集（PyTorch 标准输入/烘焙权重/超分输出）
│   ├── golden_manifest.json                   # 6 组对拍用例清单（L/S/P × 540p/720p）、偏移表与验收阈值
│   ├── input_540p_64x64_f32.bin               # 540p 真实测试裁剪块 [64, 64] float32 小端字节流
│   ├── input_720p_64x64_f32.bin               # 720p 真实测试裁剪块 [64, 64] float32 小端字节流
│   ├── baked_ssbo_{L,S,P}_{540p,720p}_prior_fp16.bin  # CPU 核烘焙后的标准 4B 对齐 FP16 WeightBuf（用于校验 TS 核烘焙代码）
│   └── output_{L,S,P}_{540p,720p}_prior_f32.bin       # PyTorch baked_fp16 标准超分结果（用于校验 WGSL 着色器输出）
└── demo_videos/                               # 4 段全平台浏览器原生硬解兼容（H.264 yuv420p）的低码率测试样片
    ├── demo_manifest.json                     # 4 段样片的元数据、真实码率/QP 及预计算好的 7D 先验向量
    ├── 01_ui_slide_editing_540p_x2.mp4        # 202 KB | 960×540 -> 1920×1080 | UI/PPT 文字边缘去振铃样片
    ├── 02_cg_china_speed_540p_x2.mp4          # 441 KB | 960×540 -> 1920×1080 | CG 动画锐利线条恢复样片
    ├── 03_detail_cactus_540p_x2.mp4           # 271 KB | 960×540 -> 1920×1080 | 低码率直播纹理去块效应样片
    └── 04_motion_jockey_720p_x1_5.mp4         # 232 KB | 1280×720 -> 1920×1080 | 高速实景 ×1.5 超分样片
```

---

## 2. 推荐纯前端技术架构（零后端 GPU 成本）

网站完全是**纯静态前端项目**（可直接部署在 Cloudflare Pages / Vercel / GitHub Pages），所有解码、超分、编码均在访问者自己的手机或电脑本地完成：

- **构建框架**：`Vite + TypeScript + React (或 Vue)`
- **核心 GPU 推理引擎**：**原生 WebGPU (WGSL Compute Shader)**
  - 直接读取 `weights/*.bin`，在浏览器 JS/TS 中用 < 2 ms 完成 7D 先验动态核烘焙，生成静态 `GPUBuffer`（`WeightBuf`），每帧按顺序 Dispatch 7 个极简算子（`op=0..6`）。
  - *可选兼容兜底*：针对不支持 WebGPU 的老旧手机内置浏览器，可单独为体积最小的 `Client-L` 写一套 WebGL2 Fragment Shader 回退管线。
- **双交互模式建议**：
  1. **实时对比播放器（首屏核心体验，零等待）**：
     - 内置 `demo_videos/` 下的 4 段典型低码率样片，同时支持用户拖入本地视频；
     - 用 `<video>` 硬件解码 + `requestVideoFrameCallback` 取帧 $\to$ WebGPU 超分 $\to$ `<canvas>` 渲染；
     - 提供 **左右分屏滑动对比条（`Bicubic` vs `CodecSR L/S/P`）**、**2×/4× 局部放大镜**、以及 **“压缩修复强度（先验 QP）”实时调节滑杆**。
  2. **本地导出超分 MP4（离线工具模式）**：
     - 使用 `mp4box.js` 解封装 + 浏览器原生 `WebCodecs`（`VideoDecoder` 逐帧解码 $\to$ WebGPU 超分 $\to$ `VideoEncoder` 硬件编码 H.264 1080p）+ `mp4-muxer` 在浏览器内存中封装并触发本地下载。
     - 建议对移动端导出设置合理保护（如建议时长 ≤ 15~30 秒；若传入高于 720p 的源视频，提示先下采样或仅超分短边 ≤ 720p 的视频以防手机显存溢出）。

---

## 3. 二进制权重文件协议（`CSR1` / `CHR1`）

`weights/*.bin` 采用极简自描述二进制格式，无需任何第三方序列化库：

| 字节区间 | 类型 | 含义 |
| :--- | :--- | :--- |
| `0 .. 3` (4 B) | ASCII 字符串 | 魔数：亮度主干为 `"CSR1"`，色度头为 `"CHR1"` |
| `4 .. 7` (4 B) | `uint32` (Little-Endian) | JSON 元数据字节长度 `meta_len` |
| `8 .. 8 + meta_len` | UTF-8 JSON 字符串 | 包含 `tier`、`scale`、`use_prior`、`prior_dim` 以及 `tensors` 数组（每项含 `name`、`shape`、`dtype: "float16"`、`offset`、`nbytes`） |
| `8 + meta_len .. EOF` | Raw `float16` 字节流 | 按每个 tensor 的 `offset`（相对于数据区起点）与 `nbytes` 紧密排列的小端 IEEE-754 binary16 数据 |

### TypeScript 读取示例（可直接复用）

```ts
export interface ParsedBlob {
  magic: string;
  meta: {
    tier?: string;
    scale: number;
    use_prior?: boolean;
    prior_dim?: number;
    tensors: Array<{ name: string; shape: number[]; dtype: string; offset: number; nbytes: number }>;
  };
  tensors: Record<string, Float32Array>;
}

// IEEE-754 float16 (uint16) -> float32 转换（兼容尚未支持 Float16Array 的浏览器）
function fp16ToFp32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export async function loadWeightBlob(url: string): Promise<ParsedBlob> {
  const buf = await (await fetch(url)).arrayBuffer();
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  const metaLen = view.getUint32(4, true);
  const metaJson = new TextDecoder().decode(new Uint8Array(buf, 8, metaLen));
  const meta = JSON.parse(metaJson);
  const dataStart = 8 + metaLen;

  const tensors: Record<string, Float32Array> = {};
  for (const t of meta.tensors) {
    const count = t.nbytes >>> 1;
    const out = new Float32Array(count);
    const base = dataStart + t.offset;
    for (let i = 0; i < count; i++) {
      out[i] = fp16ToFp32(view.getUint16(base + i * 2, true));
    }
    tensors[t.name] = out;
  }
  return { magic, meta, tensors };
}
```

---

## 4. 7D 码流先验与 CPU 动态核烘焙（`bake_csr1_to_ssbo`）

**请直接对照 `reference_code/standalone_numpy_reference.py` 翻译成 TypeScript**（总共不到 200 行纯加减乘除，无任何框架依赖）。

### 4.1 7D 先验向量定义
`prior_7d = [qp_norm, codec_id, is_live_vbv, log_bpp_norm, rd_complexity, is_screen_content, has_prior]`：
1. **内置样片**：直接读取 `demo_videos/demo_manifest.json` 中预计算好的 `prior_7d`。
2. **用户上传本地视频时**：
   - 从 `<video>` 或 `mp4box.js` 获取 `width`、`height`、`fps`（未知则默认 `30`）、`bitrate_kbps = (file.size * 8 / 1000) / duration`、以及编码格式 `avc1 -> h264 (0.0)` / `hvc1,hev1 -> hevc (1.0)`；
   - 由于浏览器无法直接从硬解器读取宏块平均 `QP`，可以在页面提供一个 **「压缩伪影修复强度」滑块**（对应虚拟 `qp_mean ∈ [20, 40]`，默认可根据 `bpp` 自动反推：`qp_est = clamp(28 - 8 * log10(bpp / 0.05), 20, 40)`），以及一个 **「文字/屏幕内容优化」开关**（`is_screen_content = 0 或 1`）；
   - 当用户拖动滑块时，调用 `bake_csr1_to_ssbo` 在 CPU 上重新烘焙一次权重（仅耗时 **0.5 ~ 2 ms**）并 `device.queue.writeBuffer` 更新 `WeightBuf` 即可实时生效！
   - 若不想启用先验，也可直接加载 `*_noprior.bin`。

---

## 5. WebGPU (WGSL) 移植与色彩空间细节

### 5.1 从 GLSL (`client_sr_v3.comp`) 迁移到 WGSL 的一一对应关系
`reference_code/client_sr_v3.comp` 中已经把整网推理浓缩成单个 Compute Shader 里的 **7 个无状态函数**（工作组大小 `@workgroup_size(16, 16, 1)`）：
- `u_op == 0` (`op_bicubic`)：Keys $a = -0.75$ 双三次上采样基础路径（严格对齐 `F.interpolate(..., mode="bicubic", align_corners=False)`）
- `u_op == 1` (`op_unshuffle2`)：`PixelUnshuffle(2)` 空间转通道下采样
- `u_op == 2` (`op_conv3x3`)：通用 3×3 卷积 + 可选 `8×8` Mod-16 相位偏置（`u_has_pb`）+ 可选逐通道 `PReLU`（`u_has_act`）
- `u_op == 3` (`op_span_combine`)：SPAN-Lite 无参跳跃连接融合 `dst = dst + alpha * buf_a + beta * buf_b`
- `u_op == 4` (`op_shuffle_r`)：`PixelShuffle(r)` 亚像素重排上采样（`r = 2, 3, 4`）
- `u_op == 5` (`op_seam_lut`)：`Client-L / S` 专属 `17×17×17` 三线性 3D LUT 宏块接缝去块分支
- `u_op == 6` (`op_add_trust_clamp`)：残差相加 + 局部 3×3 邻域自适应抗振铃软信任域截断 `[y_min - delta, y_max + delta] ∩ [0, 1]`

> **WGSL 移植小贴士**：
> 1. GLSL 里的 `unpackHalf2x16(uint_word)` 在 WGSL 中有完全等价的原生内置函数 **`unpack2x16float(word: u32) -> vec2<f32>`**，因此**不需要**开启 WebGPU 的 `shader-f16` 扩展即可在所有支持 WebGPU 的手机/电脑上直接读取 FP16 `WeightBuf`！
> 2. 不同模型档位（标准流 `L / S` vs 沙漏流 `P`，以及 `×2.0` vs `×1.5`）的逐层 Dispatch 顺序，请直接照搬 `reference_code/verify_compute_shader.py` 第 **470 ~ 559 行**（总共仅约 80 行代码）。

### 5.2 视频帧 RGB ↔ YCbCr 色彩处理说明
模型主干输入 `x` 为归一化到 **`[0.0, 1.0]` 的全范围亮度通道（Full-Range Luma $Y$）**：
1. **前处理 Pass（RGB $\to$ YCbCr）**：从 `<video>` / `VideoFrame` 采样线性化前的 sRGB 像素 `(R, G, B) ∈ [0, 1]`，按标准全范围转换提取亮度 $Y$ 与色度 $(Cb, Cr)$：
   ```text
   Y  =  0.299 * R + 0.587 * G + 0.114 * B
   Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
   Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
   ```
   *(注：训练时 `frame.to_ndarray(format="gray")` 默认采用上述 ITU-R BT.601 全范围灰度系数，按此公式提取 $Y$ 与训练分布完全一致)*
2. **亮度超分**：将 `Y`（尺寸 `H × W`）送入 `client_sr_v3` WebGPU 管线，得到高分辨率 `Y_sr`（尺寸 `out_H × out_W`）。
3. **色度上采样与后处理（YCbCr $\to$ RGB）**：
   - **轻量推荐做法**：在输出写出像素的最后一个 Pass 里，直接对原图的 `(Cb, Cr)` 做双线性/双三次纹理采样，与 `Y_sr` 合回 RGB：
     ```text
     R = clamp(Y_sr + 1.402 * (Cr - 0.5), 0.0, 1.0)
     G = clamp(Y_sr - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5), 0.0, 1.0)
     B = clamp(Y_sr + 1.772 * (Cb - 0.5), 0.0, 1.0)
     ```
   - **进阶可选做法**：若需要极致色度边缘对齐，可额外加载 `weights/chroma_S_{540p,720p}.bin`（结构见 `reference_code/models.py` 第 723 行 `YGuidedChromaHead`）。

---

## 6. 本地精度对拍验收指南（`golden_vectors/`）

在本地写完 TypeScript 核烘焙和 WGSL 着色器后，**不需要装 PyTorch**，直接用 `golden_vectors/golden_manifest.json` 里的 6 组测试用例做两步自动化单元测试：

1. **Step 1 — 校验 TypeScript 动态核烘焙**：
   - 加载 `weights/<weight_file>` 并传入用例的 `prior_7d`；
   - 检查你生成的 `half_offsets` 字典与 `baked_ssbo_half_offsets` **100% 完全相等**；
   - 检查你生成的 FP16 `WeightBuf` 与 `baked_ssbo_*.bin` 逐元素最大误差 `< 2e-4`（1 个 FP16 ULP 以内），`trust_delta` 误差 `< 5e-5`。
2. **Step 2 — 校验 WebGPU WGSL 超分管线**：
   - 读取 `input_{540p,720p}_64x64_f32.bin`（`Float32Array`，长度 `64 * 64 = 4096`）作为输入 `x_ssbo`；
   - 运行 WebGPU Compute Pipeline 读回输出 `Float32Array`（540p 输出 `128×128`，720p 输出 `96×96`）；
   - 与对应的 `output_{L,S,P}_{540p,720p}_prior_f32.bin` 计算 PSNR：
     - **验收通过标准**：`PSNR >= 70.0 dB`（最大像素绝对误差 `< 0.002`）。只要这 6 个用例全部通过，你的前端超分引擎就与论文正式版端侧推理 **100% 数学等价**！
