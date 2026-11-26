// CodecSR V3.5 WebGPU Compute Shader (1:1 port of reference_code/client_sr_v3.comp)
// Uses native WGSL unpack2x16float() — no shader-f16 extension required.

struct DispatchParams {
  op: i32,         // 0=bicubic, 1=unshuffle2, 2=conv3x3, 3=span_combine, 4=shuffle_r, 5=seam_lut, 6=add_trust_clamp
  in_c: i32,
  in_h: i32,
  in_w: i32,
  out_c: i32,
  out_h: i32,
  out_w: i32,
  w_off: u32,
  b_off: u32,
  pb_off: u32,
  act_off: u32,
  has_pb: i32,
  has_act: i32,
  scale_r: i32,
  add_b: i32,
  _pad0: i32,
  alpha: f32,
  beta: f32,
  delta: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<storage, read> weights_u32: array<u32>;
@group(0) @binding(1) var<storage, read> buf_a: array<f32>;
@group(0) @binding(2) var<storage, read> buf_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_buf: array<f32>;
@group(0) @binding(4) var<uniform> u: DispatchParams;

fn read_fp16(elem_idx: u32) -> f32 {
  let word = weights_u32[elem_idx >> 1u];
  let pair = unpack2x16float(word);
  return select(pair.y, pair.x, (elem_idx & 1u) == 0u);
}

// PyTorch F.interpolate(mode="bicubic", align_corners=False) uses Keys a = -0.75
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

fn op_bicubic(ox: i32, oy: i32) {
  let sx = (f32(ox) + 0.5) * (f32(u.in_w) / f32(u.out_w)) - 0.5;
  let sy = (f32(oy) + 0.5) * (f32(u.in_h) / f32(u.out_h)) - 0.5;
  let ix0 = i32(floor(sx));
  let iy0 = i32(floor(sy));

  var acc: f32 = 0.0;
  for (var dy: i32 = -1; dy <= 2; dy = dy + 1) {
    let ry = clamp(iy0 + dy, 0, u.in_h - 1);
    let wy = cubic_w(sy - f32(iy0 + dy));
    for (var dx: i32 = -1; dx <= 2; dx = dx + 1) {
      let rx = clamp(ix0 + dx, 0, u.in_w - 1);
      let wx = cubic_w(sx - f32(ix0 + dx));
      acc = acc + (wy * wx) * buf_a[ry * u.in_w + rx];
    }
  }
  dst_buf[oy * u.out_w + ox] = acc;
}

fn op_unshuffle2(ox: i32, oy: i32) {
  let in_hw = u.in_h * u.in_w;
  for (var c: i32 = 0; c < u.in_c; c = c + 1) {
    let base_in = c * in_hw;
    for (var dy: i32 = 0; dy < 2; dy = dy + 1) {
      let iy = oy * 2 + dy;
      for (var dx: i32 = 0; dx < 2; dx = dx + 1) {
        let ix = ox * 2 + dx;
        let oc = c * 4 + dy * 2 + dx;
        let out_idx = (oc * u.out_h + oy) * u.out_w + ox;
        dst_buf[out_idx] = buf_a[base_in + iy * u.in_w + ix];
      }
    }
  }
}

fn op_conv3x3(ox: i32, oy: i32) {
  let in_hw = u.in_h * u.in_w;
  let out_hw = u.out_h * u.out_w;
  let pb_y = oy & 7;
  let pb_x = ox & 7;

  for (var co: i32 = 0; co < u.out_c; co = co + 1) {
    var acc: f32 = read_fp16(u.b_off + u32(co));
    let w_co_base = u.w_off + u32(co * u.in_c * 9);

    for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
      let iy = oy + ky - 1;
      if (iy < 0 || iy >= u.in_h) {
        continue;
      }
      for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
        let ix = ox + kx - 1;
        if (ix < 0 || ix >= u.in_w) {
          continue;
        }
        let k_off = ky * 3 + kx;
        let in_pix = iy * u.in_w + ix;
        for (var ci: i32 = 0; ci < u.in_c; ci = ci + 1) {
          let w = read_fp16(w_co_base + u32(ci * 9 + k_off));
          acc = acc + w * buf_a[ci * in_hw + in_pix];
        }
      }
    }

    if (u.has_pb != 0) {
      acc = acc + read_fp16(u.pb_off + u32((co * 8 + pb_y) * 8 + pb_x));
    }
    if (u.has_act != 0 && acc < 0.0) {
      acc = acc * read_fp16(u.act_off + u32(co));
    }
    dst_buf[co * out_hw + oy * u.out_w + ox] = acc;
  }
}

fn op_span_combine(ox: i32, oy: i32) {
  let hw = u.out_h * u.out_w;
  let pix = oy * u.out_w + ox;
  for (var c: i32 = 0; c < u.out_c; c = c + 1) {
    let idx = c * hw + pix;
    dst_buf[idx] = dst_buf[idx] + u.alpha * buf_a[idx] + u.beta * buf_b[idx];
  }
}

fn op_shuffle_r(ox: i32, oy: i32) {
  let r = u.scale_r;
  let iy = oy / r;
  let dy = oy - iy * r;
  let ix = ox / r;
  let dx = ox - ix * r;
  let sub = dy * r + dx;
  let r2 = r * r;
  let in_hw = u.in_h * u.in_w;
  let out_hw = u.out_h * u.out_w;
  let in_pix = iy * u.in_w + ix;
  let out_pix = oy * u.out_w + ox;

  for (var c: i32 = 0; c < u.out_c; c = c + 1) {
    let ic = c * r2 + sub;
    var val: f32 = buf_a[ic * in_hw + in_pix];
    let out_idx = c * out_hw + out_pix;
    if (u.add_b != 0) {
      val = val + buf_b[out_idx];
    }
    dst_buf[out_idx] = val;
  }
}

fn lut_sample(d_norm: f32, h_norm: f32, w_norm: f32) -> f32 {
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

  let c000 = read_fp16(u.w_off + u32((d0 * 17 + h0) * 17 + w0));
  let c001 = read_fp16(u.w_off + u32((d0 * 17 + h0) * 17 + w1));
  let c010 = read_fp16(u.w_off + u32((d0 * 17 + h1) * 17 + w0));
  let c011 = read_fp16(u.w_off + u32((d0 * 17 + h1) * 17 + w1));
  let c100 = read_fp16(u.w_off + u32((d1 * 17 + h0) * 17 + w0));
  let c101 = read_fp16(u.w_off + u32((d1 * 17 + h0) * 17 + w1));
  let c110 = read_fp16(u.w_off + u32((d1 * 17 + h1) * 17 + w0));
  let c111 = read_fp16(u.w_off + u32((d1 * 17 + h1) * 17 + w1));

  let c00 = mix(c000, c001, tw);
  let c01 = mix(c010, c011, tw);
  let c10 = mix(c100, c101, tw);
  let c11 = mix(c110, c111, tw);
  let c0 = mix(c00, c01, th);
  let c1 = mix(c10, c11, th);
  return mix(c0, c1, td);
}

fn eval_seam_lr(lx: i32, ly: i32) -> f32 {
  let cx = clamp(lx, 0, u.in_w - 1);
  let cy = clamp(ly, 0, u.in_h - 1);
  let xm = max(cx - 1, 0);
  let xp = min(cx + 1, u.in_w - 1);
  let ym = max(cy - 1, 0);
  let yp = min(cy + 1, u.in_h - 1);

  let xc = buf_a[cy * u.in_w + cx];
  let dh_l = clamp(0.5 * (buf_a[cy * u.in_w + xm] - xc) + 0.5, 0.0, 1.0);
  let dh_r = clamp(0.5 * (buf_a[cy * u.in_w + xp] - xc) + 0.5, 0.0, 1.0);
  let dv_u = clamp(0.5 * (buf_a[ym * u.in_w + cx] - xc) + 0.5, 0.0, 1.0);
  let dv_d = clamp(0.5 * (buf_a[yp * u.in_w + cx] - xc) + 0.5, 0.0, 1.0);

  let lh = lut_sample(xc, dh_l, dh_r);
  let lv = lut_sample(xc, dv_u, dv_d);
  let mask = read_fp16(u.b_off + u32((cy & 7) * 8 + (cx & 7)));
  return 0.5 * (lh + lv) * mask;
}

fn op_seam_lut(ox: i32, oy: i32) {
  let sx = clamp((f32(ox) + 0.5) * (f32(u.in_w) / f32(u.out_w)) - 0.5, 0.0, f32(u.in_w - 1));
  let sy = clamp((f32(oy) + 0.5) * (f32(u.in_h) / f32(u.out_h)) - 0.5, 0.0, f32(u.in_h - 1));
  let x0 = i32(floor(sx));
  let y0 = i32(floor(sy));
  let x1 = min(x0 + 1, u.in_w - 1);
  let y1 = min(y0 + 1, u.in_h - 1);
  let fx = sx - f32(x0);
  let fy = sy - f32(y0);

  let v00 = eval_seam_lr(x0, y0);
  let v01 = eval_seam_lr(x1, y0);
  let v10 = eval_seam_lr(x0, y1);
  let v11 = eval_seam_lr(x1, y1);
  let delta = mix(mix(v00, v01, fx), mix(v10, v11, fx), fy);

  let out_idx = oy * u.out_w + ox;
  dst_buf[out_idx] = dst_buf[out_idx] + delta;
}

fn op_add_trust_clamp(ox: i32, oy: i32) {
  let out_idx = oy * u.out_w + ox;
  let y_raw = buf_a[out_idx] + dst_buf[out_idx];

  let lcx = clamp((ox * u.in_w) / u.out_w, 0, u.in_w - 1);
  let lcy = clamp((oy * u.in_h) / u.out_h, 0, u.in_h - 1);

  var y_min: f32 = 1e9;
  var y_max: f32 = -1e9;
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    let ry = clamp(lcy + dy, 0, u.in_h - 1);
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let rx = clamp(lcx + dx, 0, u.in_w - 1);
      let v = buf_b[ry * u.in_w + rx];
      y_min = min(y_min, v);
      y_max = max(y_max, v);
    }
  }

  let lo = y_min - u.delta;
  let hi = y_max + u.delta;
  dst_buf[out_idx] = clamp(clamp(y_raw, lo, hi), 0.0, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x);
  let oy = i32(gid.y);
  if (ox >= u.out_w || oy >= u.out_h) {
    return;
  }

  if (u.op == 0) {
    op_bicubic(ox, oy);
  } else if (u.op == 1) {
    op_unshuffle2(ox, oy);
  } else if (u.op == 2) {
    op_conv3x3(ox, oy);
  } else if (u.op == 3) {
    op_span_combine(ox, oy);
  } else if (u.op == 4) {
    op_shuffle_r(ox, oy);
  } else if (u.op == 5) {
    op_seam_lut(ox, oy);
  } else if (u.op == 6) {
    op_add_trust_clamp(ox, oy);
  }
}
