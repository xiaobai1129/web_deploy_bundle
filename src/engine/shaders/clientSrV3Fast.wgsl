// CodecSR V3.5 High-Performance WebGPU Compute Kernels
// - NC4HW4 (vec4<f32>) coalesced feature layout
// - 18x18 Workgroup Shared Memory tiled Conv3x3 with Z-parallel 8-outCh SIMD FMA
// - LR-resolution Seam3DLUT + Fused Bicubic-Residual-Seam-TrustClamp final pass
// - 100% mathematically equivalent to client_sr_v3.comp (PSNR > 73 dB vs PyTorch FP16)

struct FastParams {
  in_c4: i32,     // number of input vec4 channel groups (or 1 when scalar)
  in_h: i32,
  in_w: i32,
  out_c4: i32,    // number of output vec4 channel groups
  out_h: i32,
  out_w: i32,
  w_off: u32,     // u32 word offset into fast_weights
  b_off: u32,     // u32 word offset for bias (or seam mask)
  pb_off: u32,    // u32 word offset for phase_bias
  act_off: u32,   // u32 word offset for PReLU
  flags: u32,     // bit0=has_pb, bit1=has_act, bit3=add_b, bit4=has_seam
  scale_r: i32,
  alpha: f32,
  beta: f32,
  delta: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<storage, read> f_weights: array<u32>;
@group(0) @binding(0) var<storage, read> s_seam_lr: array<f32>;

@group(0) @binding(1) var<storage, read> s_buf_a: array<f32>;
@group(0) @binding(1) var<storage, read> v_buf_a: array<vec4<f32>>;

@group(0) @binding(2) var<storage, read> s_buf_b: array<f32>;
@group(0) @binding(2) var<storage, read> v_buf_b: array<vec4<f32>>;

@group(0) @binding(3) var<storage, read_write> s_dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_dst: array<vec4<f32>>;

@group(0) @binding(4) var<uniform> u: FastParams;

fn unpack4(off_u32: u32) -> vec4<f32> {
  let lo = unpack2x16float(f_weights[off_u32]);
  let hi = unpack2x16float(f_weights[off_u32 + 1u]);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}

fn read_fp16_fast(base_u32: u32, elem_idx: u32) -> f32 {
  let word = f_weights[base_u32 + (elem_idx >> 1u)];
  let pair = unpack2x16float(word);
  return select(pair.y, pair.x, (elem_idx & 1u) == 0u);
}

fn cubic_w(t: f32) -> f32 {
  let x = abs(t);
  let a = -0.75;
  if (x <= 1.0) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  } else if (x < 2.0) {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
  }
  return 0.0;
}

// ============================================================================
// Kernel 0: Bicubic upsample (Keys a = -0.75, scalar f32 -> scalar f32)
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_bicubic_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  if (ox >= u.out_w || oy >= u.out_h) {
    return;
  }
  let sx = (f32(ox) + 0.5) * (f32(u.in_w) / f32(u.out_w)) - 0.5;
  let sy = (f32(oy) + 0.5) * (f32(u.in_h) / f32(u.out_h)) - 0.5;
  let ix0 = i32(floor(sx));
  let iy0 = i32(floor(sy));

  var acc: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 2; dy = dy + 1) {
    let ry = clamp(iy0 + dy, 0, u.in_h - 1);
    let wy = cubic_w(sy - f32(iy0 + dy));
    let row_off = ry * u.in_w;
    for (var dx: i32 = -1; dx <= 2; dx = dx + 1) {
      let rx = clamp(ix0 + dx, 0, u.in_w - 1);
      let wx = cubic_w(sx - f32(ix0 + dx));
      acc = fma(wy * wx, s_buf_a[row_off + rx], acc);
    }
  }
  s_dst[oy * u.out_w + ox] = acc;
}

// ============================================================================
// Kernel 1a: PixelUnshuffle(2) from scalar f32 (1ch) -> NC4HW4 (1 vec4)
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_unshuffle2_scalar_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  if (ox >= u.out_w || oy >= u.out_h) {
    return;
  }
  let iy = oy * 2;
  let ix = ox * 2;
  let r0 = iy * u.in_w + ix;
  let r1 = (iy + 1) * u.in_w + ix;
  v_dst[oy * u.out_w + ox] = vec4<f32>(
    s_buf_a[r0],
    s_buf_a[r0 + 1],
    s_buf_a[r1],
    s_buf_a[r1 + 1],
  );
}

// ============================================================================
// Kernel 1b: PixelUnshuffle(2) from NC4HW4 -> NC4HW4 (4x4 transpose per ic4)
// Dispatched with (ceil(W/16), ceil(H/16), in_c4)
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_unshuffle2_c4_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  let ic4 = i32(gid.z);
  if (ox >= u.out_w || oy >= u.out_h || ic4 >= u.in_c4) {
    return;
  }
  let in_hw = u.in_h * u.in_w;
  let out_hw = u.out_h * u.out_w;
  let base_in = ic4 * in_hw;
  let iy = oy * 2;
  let ix = ox * 2;
  let r0 = base_in + iy * u.in_w + ix;
  let r1 = base_in + (iy + 1) * u.in_w + ix;

  let v00 = v_buf_a[r0];
  let v01 = v_buf_a[r0 + 1];
  let v10 = v_buf_a[r1];
  let v11 = v_buf_a[r1 + 1];

  let out_pix = oy * u.out_w + ox;
  let oc4 = ic4 * 4;
  v_dst[(oc4 + 0) * out_hw + out_pix] = vec4<f32>(v00.x, v01.x, v10.x, v11.x);
  v_dst[(oc4 + 1) * out_hw + out_pix] = vec4<f32>(v00.y, v01.y, v10.y, v11.y);
  v_dst[(oc4 + 2) * out_hw + out_pix] = vec4<f32>(v00.z, v01.z, v10.z, v11.z);
  v_dst[(oc4 + 3) * out_hw + out_pix] = vec4<f32>(v00.w, v01.w, v10.w, v11.w);
}

// ============================================================================
// Kernel 2a: Tiled Shared-Memory Conv3x3 (1ch scalar f32 -> NC4HW4, for Client-P stem1)
// Dispatched with (ceil(W/16), ceil(H/16), ceil(out_c4 / 2))
// ============================================================================
var<workgroup> smem_scalar: array<f32, 324>; // 18 * 18

@compute @workgroup_size(16, 16, 1)
fn op_conv3x3_scalar_to_c4_main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let lx = i32(lid.x);
  let ly = i32(lid.y);
  let ox = i32(wid.x) * 16 + lx;
  let oy = i32(wid.y) * 16 + ly;
  let co4_0 = i32(wid.z) * 2;
  let co4_1 = co4_0 + 1;
  let has_second = co4_1 < u.out_c4;
  let in_bounds = (ox < u.out_w) && (oy < u.out_h);

  let tid = ly * 16 + lx;
  let tile_x0 = i32(wid.x) * 16 - 1;
  let tile_y0 = i32(wid.y) * 16 - 1;

  for (var idx: i32 = tid; idx < 324; idx = idx + 256) {
    let sy = idx / 18;
    let sx = idx - sy * 18;
    let gy = tile_y0 + sy;
    let gx = tile_x0 + sx;
    if (gx >= 0 && gx < u.in_w && gy >= 0 && gy < u.in_h) {
      smem_scalar[idx] = s_buf_a[gy * u.in_w + gx];
    } else {
      smem_scalar[idx] = 0.0;
    }
  }
  workgroupBarrier();

  if (!in_bounds) {
    return;
  }

  var acc0 = unpack4(u.b_off + u32(co4_0) * 2u);
  var acc1 = vec4<f32>(0.0);
  if (has_second) {
    acc1 = unpack4(u.b_off + u32(co4_1) * 2u);
  }

  let w_base_0 = u.w_off + u32(co4_0) * 72u;
  let w_base_1 = u.w_off + u32(co4_1) * 72u;
  let smem_row0 = ly * 18 + lx;

  for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
    let row = smem_row0 + ky * 18;
    let wk = u32(ky * 3);
    for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
      let vx = vec4<f32>(smem_scalar[row + kx]);
      let k_u32 = (wk + u32(kx)) * 8u;
      acc0 = fma(vx, unpack4(w_base_0 + k_u32), acc0);
      if (has_second) {
        acc1 = fma(vx, unpack4(w_base_1 + k_u32), acc1);
      }
    }
  }

  if ((u.flags & 2u) != 0u) {
    let slope0 = unpack4(u.act_off + u32(co4_0) * 2u);
    acc0 = select(acc0, acc0 * slope0, acc0 < vec4<f32>(0.0));
    if (has_second) {
      let slope1 = unpack4(u.act_off + u32(co4_1) * 2u);
      acc1 = select(acc1, acc1 * slope1, acc1 < vec4<f32>(0.0));
    }
  }

  let out_hw = u.out_h * u.out_w;
  let out_pix = oy * u.out_w + ox;
  v_dst[co4_0 * out_hw + out_pix] = acc0;
  if (has_second) {
    v_dst[co4_1 * out_hw + out_pix] = acc1;
  }
}

// ============================================================================
// Kernel 2b: Tiled Shared-Memory Conv3x3 (NC4HW4 -> NC4HW4)
// 16x16 spatial tile + 1px halo = 18x18 shared memory (5.18 KB)
// Each thread computes 2 vec4s = 8 output channels in parallel!
// Dispatched with (ceil(W/16), ceil(H/16), ceil(out_c4 / 2))
// ============================================================================
var<workgroup> smem_tile: array<vec4<f32>, 324>; // 18 * 18

@compute @workgroup_size(16, 16, 1)
fn op_conv3x3_c4_main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let lx = i32(lid.x);
  let ly = i32(lid.y);
  let ox = i32(wid.x) * 16 + lx;
  let oy = i32(wid.y) * 16 + ly;
  let co4_0 = i32(wid.z) * 2;
  let co4_1 = co4_0 + 1;
  let has_second = co4_1 < u.out_c4;
  let in_bounds = (ox < u.out_w) && (oy < u.out_h);

  let tid = ly * 16 + lx;
  let tile_x0 = i32(wid.x) * 16 - 1;
  let tile_y0 = i32(wid.y) * 16 - 1;
  let in_hw = u.in_h * u.in_w;
  let out_hw = u.out_h * u.out_w;

  var acc0 = unpack4(u.b_off + u32(co4_0) * 2u);
  var acc1 = vec4<f32>(0.0);
  if (has_second) {
    acc1 = unpack4(u.b_off + u32(co4_1) * 2u);
  }

  let layer_stride_u32 = u32(u.in_c4) * 72u; // in_c4 * 9 * 4 * 2 u32
  let w_base_0 = u.w_off + u32(co4_0) * layer_stride_u32;
  let w_base_1 = u.w_off + u32(co4_1) * layer_stride_u32;

  for (var ci4: i32 = 0; ci4 < u.in_c4; ci4 = ci4 + 1) {
    let ch_off = ci4 * in_hw;
    for (var idx: i32 = tid; idx < 324; idx = idx + 256) {
      let sy = idx / 18;
      let sx = idx - sy * 18;
      let gy = tile_y0 + sy;
      let gx = tile_x0 + sx;
      if (gx >= 0 && gx < u.in_w && gy >= 0 && gy < u.in_h) {
        smem_tile[idx] = v_buf_a[ch_off + gy * u.in_w + gx];
      } else {
        smem_tile[idx] = vec4<f32>(0.0);
      }
    }
    workgroupBarrier();

    if (in_bounds) {
      let w0_ci = w_base_0 + u32(ci4) * 72u;
      let w1_ci = w_base_1 + u32(ci4) * 72u;
      let smem_row0 = ly * 18 + lx;

      for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
        let row = smem_row0 + ky * 18;
        let wk = u32(ky * 3);
        for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
          let v = smem_tile[row + kx];
          let k_u32 = (wk + u32(kx)) * 8u;
          let vx = vec4<f32>(v.x);
          let vy = vec4<f32>(v.y);
          let vz = vec4<f32>(v.z);
          let vw = vec4<f32>(v.w);

          acc0 = fma(vx, unpack4(w0_ci + k_u32 + 0u), acc0);
          acc0 = fma(vy, unpack4(w0_ci + k_u32 + 2u), acc0);
          acc0 = fma(vz, unpack4(w0_ci + k_u32 + 4u), acc0);
          acc0 = fma(vw, unpack4(w0_ci + k_u32 + 6u), acc0);

          if (has_second) {
            acc1 = fma(vx, unpack4(w1_ci + k_u32 + 0u), acc1);
            acc1 = fma(vy, unpack4(w1_ci + k_u32 + 2u), acc1);
            acc1 = fma(vz, unpack4(w1_ci + k_u32 + 4u), acc1);
            acc1 = fma(vw, unpack4(w1_ci + k_u32 + 6u), acc1);
          }
        }
      }
    }
    workgroupBarrier();
  }

  if (!in_bounds) {
    return;
  }

  if ((u.flags & 1u) != 0u) {
    let pb_pix = u32(((oy & 7) * 8) + (ox & 7));
    acc0 = acc0 + unpack4(u.pb_off + (u32(co4_0) * 64u + pb_pix) * 2u);
    if (has_second) {
      acc1 = acc1 + unpack4(u.pb_off + (u32(co4_1) * 64u + pb_pix) * 2u);
    }
  }

  if ((u.flags & 2u) != 0u) {
    let slope0 = unpack4(u.act_off + u32(co4_0) * 2u);
    acc0 = select(acc0, acc0 * slope0, acc0 < vec4<f32>(0.0));
    if (has_second) {
      let slope1 = unpack4(u.act_off + u32(co4_1) * 2u);
      acc1 = select(acc1, acc1 * slope1, acc1 < vec4<f32>(0.0));
    }
  }

  let out_pix = oy * u.out_w + ox;
  v_dst[co4_0 * out_hw + out_pix] = acc0;
  if (has_second) {
    v_dst[co4_1 * out_hw + out_pix] = acc1;
  }
}

// ============================================================================
// Kernel 3: SPAN-Lite feature highway combine (NC4HW4, Z-parallel)
// Dispatched with (ceil(W/16), ceil(H/16), out_c4)
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_span_combine_c4_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  let c4 = i32(gid.z);
  if (ox >= u.out_w || oy >= u.out_h || c4 >= u.out_c4) {
    return;
  }
  let idx = c4 * (u.out_h * u.out_w) + oy * u.out_w + ox;
  let cur = v_dst[idx];
  v_dst[idx] = fma(vec4<f32>(u.alpha), v_buf_a[idx], fma(vec4<f32>(u.beta), v_buf_b[idx], cur));
}

// ============================================================================
// Kernel 4a: PixelShuffle(r) NC4HW4 -> scalar f32 (final 1ch residual output)
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_shuffle_to_scalar_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  if (ox >= u.out_w || oy >= u.out_h) {
    return;
  }
  let r = u.scale_r;
  let iy = oy / r;
  let dy = oy - iy * r;
  let ix = ox / r;
  let dx = ox - ix * r;
  let ic = dy * r + dx;
  let ic4 = ic >> 2;
  let lane = ic & 3;
  let v = v_buf_a[ic4 * (u.in_h * u.in_w) + iy * u.in_w + ix];
  var val: f32 = v.x;
  if (lane == 1) {
    val = v.y;
  } else if (lane == 2) {
    val = v.z;
  } else if (lane == 3) {
    val = v.w;
  }
  s_dst[oy * u.out_w + ox] = val;
}

// ============================================================================
// Kernel 4b: PixelShuffle(2) NC4HW4 -> NC4HW4 (+ optional skip add) for Client-P
// Dispatched with (ceil(out_w/16), ceil(out_h/16), out_c4)
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_shuffle2_c4_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  let oc4 = i32(gid.z);
  if (ox >= u.out_w || oy >= u.out_h || oc4 >= u.out_c4) {
    return;
  }
  let iy = oy >> 1;
  let dy = oy & 1;
  let ix = ox >> 1;
  let dx = ox & 1;
  let sub = dy * 2 + dx;
  let in_hw = u.in_h * u.in_w;
  let in_pix = iy * u.in_w + ix;
  let base_ic4 = oc4 * 4;

  let v0 = v_buf_a[(base_ic4 + 0) * in_hw + in_pix];
  let v1 = v_buf_a[(base_ic4 + 1) * in_hw + in_pix];
  let v2 = v_buf_a[(base_ic4 + 2) * in_hw + in_pix];
  let v3 = v_buf_a[(base_ic4 + 3) * in_hw + in_pix];

  var res = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
  if (sub == 1) {
    res = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
  } else if (sub == 2) {
    res = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
  } else if (sub == 3) {
    res = vec4<f32>(v0.w, v1.w, v2.w, v3.w);
  }

  let out_idx = oc4 * (u.out_h * u.out_w) + oy * u.out_w + ox;
  if ((u.flags & 8u) != 0u) {
    res = res + v_buf_b[out_idx];
  }
  v_dst[out_idx] = res;
}

// ============================================================================
// Kernel 5: Seam3DLUT computed ONCE at LR resolution (4x less work than HR!)
// ============================================================================
fn lut_sample_fast(d_norm: f32, h_norm: f32, w_norm: f32) -> f32 {
  let fd = clamp(d_norm, 0.0, 1.0) * 16.0;
  let fh = clamp(h_norm, 0.0, 1.0) * 16.0;
  let fw = clamp(w_norm, 0.0, 1.0) * 16.0;
  let d0 = i32(floor(fd));
  let h0 = i32(floor(fh));
  let w0 = i32(floor(fw));
  let d1 = min(d0 + 1, 16);
  let h1 = min(h0 + 1, 16);
  let w1 = min(w0 + 1, 16);
  let td = fd - f32(d0);
  let th = fh - f32(h0);
  let tw = fw - f32(w0);

  let c000 = read_fp16_fast(u.w_off, u32((d0 * 17 + h0) * 17 + w0));
  let c001 = read_fp16_fast(u.w_off, u32((d0 * 17 + h0) * 17 + w1));
  let c010 = read_fp16_fast(u.w_off, u32((d0 * 17 + h1) * 17 + w0));
  let c011 = read_fp16_fast(u.w_off, u32((d0 * 17 + h1) * 17 + w1));
  let c100 = read_fp16_fast(u.w_off, u32((d1 * 17 + h0) * 17 + w0));
  let c101 = read_fp16_fast(u.w_off, u32((d1 * 17 + h0) * 17 + w1));
  let c110 = read_fp16_fast(u.w_off, u32((d1 * 17 + h1) * 17 + w0));
  let c111 = read_fp16_fast(u.w_off, u32((d1 * 17 + h1) * 17 + w1));

  let c00 = mix(c000, c001, tw);
  let c01 = mix(c010, c011, tw);
  let c10 = mix(c100, c101, tw);
  let c11 = mix(c110, c111, tw);
  let c0 = mix(c00, c01, th);
  let c1 = mix(c10, c11, th);
  return mix(c0, c1, td);
}

@compute @workgroup_size(16, 16, 1)
fn op_seam_lut_lr_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cx = i32(gid.x);
  let cy = i32(gid.y);
  if (cx >= u.in_w || cy >= u.in_h) {
    return;
  }
  let xm = max(cx - 1, 0);
  let xp = min(cx + 1, u.in_w - 1);
  let ym = max(cy - 1, 0);
  let yp = min(cy + 1, u.in_h - 1);

  let xc = s_buf_a[cy * u.in_w + cx];
  let dh_l = clamp(0.5 * (s_buf_a[cy * u.in_w + xm] - xc) + 0.5, 0.0, 1.0);
  let dh_r = clamp(0.5 * (s_buf_a[cy * u.in_w + xp] - xc) + 0.5, 0.0, 1.0);
  let dv_u = clamp(0.5 * (s_buf_a[ym * u.in_w + cx] - xc) + 0.5, 0.0, 1.0);
  let dv_d = clamp(0.5 * (s_buf_a[yp * u.in_w + cx] - xc) + 0.5, 0.0, 1.0);

  let lh = lut_sample_fast(xc, dh_l, dh_r);
  let lv = lut_sample_fast(xc, dv_u, dv_d);
  let mask = read_fp16_fast(u.b_off, u32((cy & 7) * 8 + (cx & 7)));
  s_dst[cy * u.in_w + cx] = 0.5 * (lh + lv) * mask;
}

// ============================================================================
// Kernel 6: Fused Residual + Optional Bilinear SeamLUT + 3x3 TrustClamp
// Binding 0: seam_lr_buf (read), Binding 1: base_buf (read),
// Binding 2: x_buf (read), Binding 3: res_buf (read_write), Binding 4: params
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn op_add_seam_trust_clamp_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  if (ox >= u.out_w || oy >= u.out_h) {
    return;
  }
  let out_idx = oy * u.out_w + ox;
  var res_val = s_dst[out_idx];

  if ((u.flags & 16u) != 0u) {
    let sx = clamp((f32(ox) + 0.5) * (f32(u.in_w) / f32(u.out_w)) - 0.5, 0.0, f32(u.in_w - 1));
    let sy = clamp((f32(oy) + 0.5) * (f32(u.in_h) / f32(u.out_h)) - 0.5, 0.0, f32(u.in_h - 1));
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let x1 = min(x0 + 1, u.in_w - 1);
    let y1 = min(y0 + 1, u.in_h - 1);
    let fx = sx - f32(x0);
    let fy = sy - f32(y0);

    let v00 = s_seam_lr[y0 * u.in_w + x0];
    let v01 = s_seam_lr[y0 * u.in_w + x1];
    let v10 = s_seam_lr[y1 * u.in_w + x0];
    let v11 = s_seam_lr[y1 * u.in_w + x1];
    res_val = res_val + mix(mix(v00, v01, fx), mix(v10, v11, fx), fy);
  }

  let y_raw = s_buf_a[out_idx] + res_val;

  let lcx = clamp((ox * u.in_w) / u.out_w, 0, u.in_w - 1);
  let lcy = clamp((oy * u.in_h) / u.out_h, 0, u.in_h - 1);

  var y_min: f32 = 1e9;
  var y_max: f32 = -1e9;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    let ry = clamp(lcy + dy, 0, u.in_h - 1);
    let row_off = ry * u.in_w;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let rx = clamp(lcx + dx, 0, u.in_w - 1);
      let v = s_buf_b[row_off + rx];
      y_min = min(y_min, v);
      y_max = max(y_max, v);
    }
  }

  let lo = y_min - u.delta;
  let hi = y_max + u.delta;
  s_dst[out_idx] = clamp(clamp(y_raw, lo, hi), 0.0, 1.0);
}
