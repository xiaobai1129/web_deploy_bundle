export type ModelTier = 'L' | 'S' | 'P';
export type ModelScale = 2.0 | 1.5;

export interface TensorMeta {
  name: string;
  shape: number[];
  dtype: string;
  offset: number;
  nbytes: number;
}

export interface Csr1Meta {
  tier?: string;
  scale: number;
  use_prior?: boolean;
  arch_version?: string;
  prior_dim?: number;
  tensors: TensorMeta[];
}

export interface ParsedTensor {
  data: Float32Array;
  shape: number[];
}

export interface ParsedBlob {
  magic: 'CSR1' | 'CHR1';
  meta: Csr1Meta;
  tensors: Map<string, ParsedTensor>;
}

export interface StreamPriorParams {
  width: number;
  height: number;
  fps: number;
  bitrateKbps: number;
  qpMean: number;
  codec: string;
  isLive: boolean;
  isScreenContent: boolean;
  hasPrior: boolean;
}

export interface BakedScalars {
  tier: ModelTier;
  scale: number;
  usePrior: boolean;
  skipAlpha: number;
  skipBeta: number;
  skipHr: number;
  trustDelta: number;
  isHourglass: boolean;
  hgHasUp: boolean;
  hgHasHrSkip: boolean;
  numBody: number;
  stemCh: number;
  midCh: number;
  upOutCh: number;
  hasSeamLut: boolean;
}

export interface BakedSsboResult {
  ssboBytes: Uint8Array;
  halfOffsets: Record<string, number>;
  scalars: BakedScalars;
  bakeMs: number;
}

const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);

/** IEEE-754 binary16 (uint16) -> float32 */
export function fp16ToFp32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 0x1f) {
    return f ? Number.NaN : s ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Exact IEEE-754 float32 -> binary16 (uint16) with round-to-nearest-even */
export function fp32ToFp16(val: number): number {
  f32View[0] = val;
  const x = u32View[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x007fffff;

  if (exp === 0xff) {
    // Inf or NaN
    if (mant === 0) return sign | 0x7c00;
    return sign | 0x7e00 | (mant >>> 13);
  }

  const halfExp = exp - 127 + 15;
  if (halfExp >= 0x1f) {
    // Overflow to Inf
    return sign | 0x7c00;
  }
  if (halfExp <= 0) {
    if (halfExp < -10) {
      return sign;
    }
    // Subnormal half
    const fullMant = mant | 0x00800000;
    const shift = 14 - halfExp;
    const halfMant = fullMant >>> shift;
    const rem = fullMant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (halfMant & 1) !== 0)) {
      return sign | (halfMant + 1);
    }
    return sign | halfMant;
  }

  // Normal half with round-to-nearest-even
  let halfMant = mant >>> 13;
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (halfMant & 1) !== 0)) {
    halfMant += 1;
    if (halfMant === 0x400) {
      halfMant = 0;
      const nextExp = halfExp + 1;
      if (nextExp >= 0x1f) return sign | 0x7c00;
      return sign | (nextExp << 10);
    }
  }
  return sign | (halfExp << 10) | halfMant;
}

export function roundToFp16AsNumber(val: number): number {
  return fp16ToFp32(fp32ToFp16(val));
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function silu(x: number): number {
  return x / (1.0 + Math.exp(-x));
}

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

function softplus(x: number): number {
  return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0.0);
}

/**
 * Estimate reasonable virtual QP in [20, 40] from bits-per-pixel when bitstream QP is unavailable.
 */
export function estimateQpFromVideo(width: number, height: number, fps: number, bitrateKbps: number): number {
  const w = Math.max(1, width);
  const h = Math.max(1, height);
  const f = Math.max(1, fps);
  const bpp = (Math.max(1, bitrateKbps) * 1000) / (w * h * f);
  const est = 28 - 8 * Math.log10(Math.max(bpp, 1e-4) / 0.05);
  return Math.round(clamp(est, 20, 40) * 10) / 10;
}

/**
 * Exact 7D stream prior vector:
 * [qp_norm, codec_id, is_live_vbv, log_bpp_norm, rd_complexity, is_screen_content, has_prior]
 */
export function encodeStreamPrior7d(params: Partial<StreamPriorParams> = {}): number[] {
  const w = Math.max(1, Math.trunc(params.width ?? 960));
  const h = Math.max(1, Math.trunc(params.height ?? 540));
  const fpsF = Math.max(1.0, params.fps ?? 30.0);
  const kbps = Math.max(1.0, params.bitrateKbps ?? 800.0);
  const qpMean = params.qpMean ?? 28.0;
  const codec = (params.codec ?? 'h264').toLowerCase();
  const isLive = Boolean(params.isLive ?? false);
  const isScreenContent = Boolean(params.isScreenContent ?? false);
  const hasPrior = params.hasPrior ?? true;

  const bpp = (kbps * 1000.0) / (w * h * fpsF);
  const logBppNorm = clamp((Math.log10(Math.max(bpp, 1e-4)) + 2.5) / 2.0, 0.0, 1.0);
  if (!hasPrior) {
    return [0.0, 0.0, 0.0, logBppNorm, 0.0, 0.0, 0.0];
  }

  const qpNorm = qpMean / 51.0;
  const codecId =
    codec === 'hevc' || codec === 'h265' || codec === 'libx265' || codec.startsWith('hev1') || codec.startsWith('hvc1')
      ? 1.0
      : 0.0;
  const isLiveVbv = isLive ? 1.0 : 0.0;
  const resScale = Math.pow((w * h) / (960.0 * 540.0), 0.28);
  const bppCal = bpp * resScale;
  const logBppCal = clamp((Math.log10(Math.max(bppCal, 1e-4)) + 2.5) / 2.0, 0.0, 1.0);
  const rdComplexity = clamp(logBppCal + 1.8 * qpNorm - 1.1 - 0.06 * codecId, -1.0, 1.0);
  const isScreen = isScreenContent ? 1.0 : 0.0;
  return [qpNorm, codecId, isLiveVbv, logBppNorm, rdComplexity, isScreen, 1.0];
}

/**
 * Parse CSR1 (luma) or CHR1 (chroma) binary weight blob into FP32 tensors preserving insertion order.
 */
export function parseBinaryBlob(buf: ArrayBuffer): ParsedBlob {
  const view = new DataView(buf);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== 'CSR1' && magic !== 'CHR1') {
    throw new Error(`Unexpected weight blob magic header: ${JSON.stringify(magic)}`);
  }
  const metaLen = view.getUint32(4, true);
  const metaBytes = new Uint8Array(buf, 8, metaLen);
  const meta = JSON.parse(new TextDecoder().decode(metaBytes)) as Csr1Meta;
  const dataStart = 8 + metaLen;

  const tensors = new Map<string, ParsedTensor>();
  for (const t of meta.tensors) {
    const count = t.nbytes >>> 1;
    const out = new Float32Array(count);
    const base = dataStart + t.offset;
    for (let i = 0; i < count; i++) {
      out[i] = fp16ToFp32(view.getUint16(base + (i << 1), true));
    }
    tensors.set(t.name, { data: out, shape: [...t.shape] });
  }
  return { magic, meta, tensors };
}

function matVecAdd(w: Float32Array, rows: number, cols: number, x: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = b[r];
    const rowOff = r * cols;
    for (let c = 0; c < cols; c++) {
      acc += w[rowOff + c] * x[c];
    }
    out[r] = Math.fround(acc);
  }
  return out;
}

/**
 * Pure-CPU dynamic kernel baking (1:1 port of `bake_csr1_to_ssbo` in `standalone_numpy_reference.py`).
 * Typical latency: 0.4 ~ 1.8 ms in JS.
 */
export function bakeCsr1ToSsbo(blob: ParsedBlob, prior7d: number[] | null = null): BakedSsboResult {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const { meta, tensors } = blob;
  const tier = String(meta.tier ?? 'S').toUpperCase() as ModelTier;
  const scale = Number(meta.scale);
  const usePrior = Boolean(meta.use_prior);
  const isHourglass = tier === 'P';
  const hgHasUp = isHourglass && Math.abs(scale - 2.0) < 1e-4;
  const hgHasHrSkip = isHourglass && Math.abs(scale - 1.5) < 1e-4;

  let numBody: number;
  let layerPrefixes: string[];
  if (isHourglass) {
    numBody = Math.abs(scale - 2.0) < 1e-4 ? 4 : 3;
    layerPrefixes = ['stem1', 'stem2', 'down'];
    for (let i = 0; i < numBody; i++) layerPrefixes.push(`body.${i}`);
    if (hgHasUp) layerPrefixes.push('up', 'refine');
    layerPrefixes.push('tail');
  } else {
    numBody = tier === 'L' ? 4 : 5;
    layerPrefixes = ['head'];
    for (let i = 0; i < numBody; i++) layerPrefixes.push(`body.${i}`);
    layerPrefixes.push('tail');
  }

  // Deep-copy tensors preserving exact insertion order
  const outSd = new Map<string, ParsedTensor>();
  for (const [k, v] of tensors.entries()) {
    outSd.set(k, { data: new Float32Array(v.data), shape: [...v.shape] });
  }

  // 1. Predict per-layer (alphas, gamma, beta) from 7D prior if enabled
  interface LayerParam {
    alphas: Float32Array | null;
    gamma: Float32Array | null;
    beta: Float32Array | null;
  }
  let layerParams: LayerParam[] = layerPrefixes.map(() => ({ alphas: null, gamma: null, beta: null }));

  if (usePrior && prior7d !== null && outSd.has('prior_mod.mlp.0.weight')) {
    const p = new Float32Array(7);
    for (let i = 0; i < 7; i++) p[i] = prior7d[i] ?? 0;

    const mlp0W = outSd.get('prior_mod.mlp.0.weight')!;
    const mlp0B = outSd.get('prior_mod.mlp.0.bias')!;
    const h = matVecAdd(mlp0W.data, mlp0W.shape[0], mlp0W.shape[1], p, mlp0B.data);
    for (let i = 0; i < h.length; i++) h[i] = silu(h[i]);

    const mlp2W = outSd.get('prior_mod.mlp.2.weight')!;
    const mlp2B = outSd.get('prior_mod.mlp.2.bias')!;
    const raw = matVecAdd(mlp2W.data, mlp2W.shape[0], mlp2W.shape[1], h, mlp2B.data);

    let off = 0;
    layerParams = [];
    for (const pref of layerPrefixes) {
      const outCh = outSd.get(`${pref}.base.reparam_conv.bias`)!.shape[0];
      const alphas = new Float32Array(4);
      for (let k = 0; k < 4; k++) {
        alphas[k] = 0.2 * Math.tanh(raw[off + k]);
      }
      off += 4;
      const gamma = new Float32Array(outCh);
      for (let c = 0; c < outCh; c++) {
        gamma[c] = 0.1 * raw[off + c];
      }
      off += outCh;
      const beta = new Float32Array(outCh);
      for (let c = 0; c < outCh; c++) {
        beta[c] = 0.1 * raw[off + c];
      }
      off += outCh;
      layerParams.push({ alphas, gamma, beta });
    }

    // Bake Seam-3DLUT prior gain into 8x8 seam_mask
    if (outSd.has('seam_lut.seam_mask') && outSd.has('prior_mod.lut_head.weight')) {
      const lutW = outSd.get('prior_mod.lut_head.weight')!;
      const lutB = outSd.get('prior_mod.lut_head.bias')!;
      let lutRaw = lutB.data[0];
      for (let i = 0; i < 7; i++) lutRaw += lutW.data[i] * p[i];
      const gLut = 2.0 * sigmoid(lutRaw);
      const mask = outSd.get('seam_lut.seam_mask')!;
      for (let i = 0; i < mask.data.length; i++) {
        mask.data[i] *= gLut;
      }
    }
  }

  // 2. Fold K=4 dynamic bases + FiLM (gamma, beta) into each static 3x3 conv
  for (let li = 0; li < layerPrefixes.length; li++) {
    const pref = layerPrefixes[li];
    const { alphas, gamma, beta } = layerParams[li];
    const wEntry = outSd.get(`${pref}.base.reparam_conv.weight`)!;
    const bEntry = outSd.get(`${pref}.base.reparam_conv.bias`)!;
    const w = wEntry.data;
    const b = bEntry.data;
    const outCh = wEntry.shape[0];
    const inCh = wEntry.shape[1];

    const bwKey = `${pref}.basis_weight`;
    const bbKey = `${pref}.basis_bias`;
    const bwEntry = outSd.get(bwKey);
    const bbEntry = outSd.get(bbKey);

    if (alphas !== null && bwEntry && bbEntry) {
      const bw = bwEntry.data; // [4, out_ch, in_ch // groups, 3, 3]
      const bb = bbEntry.data; // [4, out_ch]
      const inChPerGroup = bwEntry.shape[2];
      const groups = Math.trunc(inCh / inChPerGroup);
      const og = Math.trunc(outCh / groups);
      const ig = inChPerGroup;
      const basisStride = outCh * inChPerGroup * 9;

      for (let k = 0; k < 4; k++) {
        const kOff = k * basisStride;
        let sqSum = 0;
        for (let idx = 0; idx < basisStride; idx++) {
          const v = bw[kOff + idx];
          sqSum += v * v;
        }
        const norm = Math.fround(Math.sqrt(sqSum) + 1e-6);
        const ak = alphas[k];

        if (groups <= 1) {
          for (let idx = 0; idx < basisStride; idx++) {
            const bNorm = Math.fround(bw[kOff + idx] / norm);
            w[idx] = Math.fround(w[idx] + Math.fround(ak * bNorm));
          }
        } else {
          for (let g = 0; g < groups; g++) {
            const coStart = g * og;
            const ciStart = g * ig;
            for (let coLocal = 0; coLocal < og; coLocal++) {
              const co = coStart + coLocal;
              const dstCoBase = (co * inCh + ciStart) * 9;
              const srcCoBase = kOff + co * ig * 9;
              const len = ig * 9;
              for (let j = 0; j < len; j++) {
                const bNorm = Math.fround(bw[srcCoBase + j] / norm);
                w[dstCoBase + j] = Math.fround(w[dstCoBase + j] + Math.fround(ak * bNorm));
              }
            }
          }
        }

        const bbBase = k * outCh;
        for (let co = 0; co < outCh; co++) {
          b[co] = Math.fround(b[co] + Math.fround(ak * bb[bbBase + co]));
        }
      }
    }

    if (gamma !== null && beta !== null) {
      const chStride = inCh * 9;
      for (let co = 0; co < outCh; co++) {
        const scaleC = Math.fround(1.0 + gamma[co]);
        const wBase = co * chStride;
        for (let j = 0; j < chStride; j++) {
          w[wBase + j] = Math.fround(w[wBase + j] * scaleC);
        }
        b[co] = Math.fround(Math.fround(b[co] * scaleC) + beta[co]);
      }
    }

    outSd.delete(bwKey);
    outSd.delete(bbKey);
  }

  // Remove prior_mod weights after baking
  for (const key of Array.from(outSd.keys())) {
    if (key.startsWith('prior_mod.')) {
      outSd.delete(key);
    }
  }

  // 3. Bake QPAdaptiveTrustClamp delta scalar
  let trustDelta: number;
  if (usePrior && prior7d !== null && outSd.has('trust_clamp.delta_head.weight')) {
    const dw = outSd.get('trust_clamp.delta_head.weight')!.data;
    const db = outSd.get('trust_clamp.delta_head.bias')!.data;
    let rawD = db[0];
    for (let i = 0; i < 7; i++) rawD += dw[i] * (prior7d[i] ?? 0);
    trustDelta = 0.004 + 0.076 * sigmoid(rawD);
  } else {
    const th = outSd.get('trust_clamp.theta')!.data;
    trustDelta = 0.004 + softplus(th[0]) + softplus(th[1]);
  }
  outSd.set('trust_clamp.baked_delta', { data: new Float32Array([trustDelta]), shape: [] });

  // 4. Pack remaining tensors into 4-byte aligned FP16 WeightBuf in exact state_dict order
  let totalHalves = 0;
  const halfOffsets: Record<string, number> = {};
  for (const [name, entry] of outSd.entries()) {
    halfOffsets[name] = totalHalves;
    totalHalves += entry.data.length;
  }

  const paddedHalves = (totalHalves & 1) === 1 ? totalHalves + 1 : totalHalves;
  const ssboU16 = new Uint16Array(paddedHalves);
  let cursor = 0;
  for (const entry of outSd.values()) {
    const arr = entry.data;
    for (let i = 0; i < arr.length; i++) {
      ssboU16[cursor++] = fp32ToFp16(arr[i]);
    }
  }

  const stemCh = isHourglass ? outSd.get('stem1.base.reparam_conv.bias')!.shape[0] : 0;
  const midCh = isHourglass
    ? outSd.get('down.base.reparam_conv.bias')!.shape[0]
    : outSd.get('head.base.reparam_conv.bias')!.shape[0];
  const upOutCh = hgHasUp ? outSd.get('up.base.reparam_conv.bias')!.shape[0] : 0;

  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return {
    ssboBytes: new Uint8Array(ssboU16.buffer),
    halfOffsets,
    scalars: {
      tier,
      scale,
      usePrior,
      skipAlpha: roundToFp16AsNumber(outSd.get('skip_alpha')!.data[0]),
      skipBeta: roundToFp16AsNumber(outSd.get('skip_beta')!.data[0]),
      skipHr: outSd.has('skip_hr') ? roundToFp16AsNumber(outSd.get('skip_hr')!.data[0]) : 0.0,
      trustDelta,
      isHourglass,
      hgHasUp,
      hgHasHrSkip,
      numBody,
      stemCh,
      midCh,
      upOutCh,
      hasSeamLut: outSd.has('seam_lut.table'),
    },
    bakeMs: t1 - t0,
  };
}
