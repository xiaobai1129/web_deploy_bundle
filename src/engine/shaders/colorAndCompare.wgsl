// RGB <-> Full-Range BT.601 YCbCr compute kernels + Split-Screen / Heatmap display shader

struct RgbToYParams {
  src_w: u32,
  src_h: u32,
  pad_w: u32,
  pad_h: u32,
};

@group(0) @binding(0) var src_rgb_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst_y_buf: array<f32>;
@group(0) @binding(2) var<uniform> rgb2y_u: RgbToYParams;

@compute @workgroup_size(16, 16, 1)
fn rgb_to_y_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= rgb2y_u.pad_w || gid.y >= rgb2y_u.pad_h) {
    return;
  }
  let cx = min(gid.x, rgb2y_u.src_w - 1u);
  let cy = min(gid.y, rgb2y_u.src_h - 1u);
  let rgb = textureLoad(src_rgb_tex, vec2<i32>(i32(cx), i32(cy)), 0).rgb;
  let y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
  dst_y_buf[gid.y * rgb2y_u.pad_w + gid.x] = clamp(y, 0.0, 1.0);
}

struct YcbcrToRgbParams {
  out_w: u32,
  out_h: u32,
  y_stride: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<storage, read> in_y_buf: array<f32>;
@group(0) @binding(1) var chroma_src_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var dst_rgba_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> y2rgb_u: YcbcrToRgbParams;

@compute @workgroup_size(16, 16, 1)
fn ycbcr_to_rgba_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= y2rgb_u.out_w || gid.y >= y2rgb_u.out_h) {
    return;
  }
  let uv = vec2<f32>(
    (f32(gid.x) + 0.5) / f32(y2rgb_u.out_w),
    (f32(gid.y) + 0.5) / f32(y2rgb_u.out_h),
  );
  let rgb_lr = textureSampleLevel(chroma_src_tex, linear_sampler, uv, 0.0).rgb;
  let cb = -0.168736 * rgb_lr.r - 0.331264 * rgb_lr.g + 0.5 * rgb_lr.b + 0.5;
  let cr =  0.5 * rgb_lr.r - 0.418688 * rgb_lr.g - 0.081312 * rgb_lr.b + 0.5;

  let y_val = in_y_buf[gid.y * y2rgb_u.y_stride + gid.x];
  let r = clamp(y_val + 1.402 * (cr - 0.5), 0.0, 1.0);
  let g = clamp(y_val - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5), 0.0, 1.0);
  let b = clamp(y_val + 1.772 * (cb - 0.5), 0.0, 1.0);
  textureStore(dst_rgba_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, 1.0));
}

// Full-screen render shader for interactive comparison viewport and export canvas
struct DisplayParams {
  split_x: f32,    // [0..1] divider position
  view_mode: u32,  // 0=split, 1=sr_only, 2=bicubic_only, 3=diff_heatmap
  zoom: f32,       // >= 1.0
  _pad0: f32,
  center_x: f32,   // [0..1]
  center_y: f32,   // [0..1]
  _pad1: f32,
  _pad2: f32,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn display_vs(@builtin(vertex_index) vid: u32) -> VsOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0),
  );
  let p = positions[vid];
  var out: VsOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}

@group(0) @binding(0) var sr_tex: texture_2d<f32>;
@group(0) @binding(1) var bicubic_tex: texture_2d<f32>;
@group(0) @binding(2) var disp_sampler: sampler;
@group(0) @binding(3) var<uniform> disp_u: DisplayParams;

fn heatmap_color(t_in: f32) -> vec3<f32> {
  let t = clamp(t_in, 0.0, 1.0);
  let c0 = vec3<f32>(0.04, 0.07, 0.14);
  let c1 = vec3<f32>(0.06, 0.48, 0.78);
  let c2 = vec3<f32>(0.22, 0.90, 0.65);
  let c3 = vec3<f32>(0.99, 0.78, 0.18);
  let c4 = vec3<f32>(0.98, 0.24, 0.28);
  if (t < 0.25) { return mix(c0, c1, t * 4.0); }
  if (t < 0.50) { return mix(c1, c2, (t - 0.25) * 4.0); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.50) * 4.0); }
  return mix(c3, c4, (t - 0.75) * 4.0);
}

@fragment
fn display_fs(in: VsOut) -> @location(0) vec4<f32> {
  let z = max(disp_u.zoom, 1.0);
  let sample_uv = clamp(
    (in.uv - vec2<f32>(0.5, 0.5)) / z + vec2<f32>(disp_u.center_x, disp_u.center_y),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
  );

  let sr_col = textureSample(sr_tex, disp_sampler, sample_uv);
  let bi_col = textureSample(bicubic_tex, disp_sampler, sample_uv);

  if (disp_u.view_mode == 1u) {
    return sr_col;
  }
  if (disp_u.view_mode == 2u) {
    return bi_col;
  }
  if (disp_u.view_mode == 3u) {
    let diff = abs(sr_col.rgb - bi_col.rgb);
    let mag = (0.299 * diff.r + 0.587 * diff.g + 0.114 * diff.b) * 10.0;
    return vec4<f32>(heatmap_color(mag), 1.0);
  }
  // Mode 0: Split-screen comparison (left = Bicubic, right = CodecSR)
  if (in.uv.x < disp_u.split_x) {
    return bi_col;
  }
  return sr_col;
}
