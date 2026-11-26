import type { BakedSsboResult } from './csr1';

export interface ConvLayerFastMeta {
  wOffU32: number;      // offset in u32 words inside fastWeightBuf
  bOffU32: number;      // bias [outC4] packed as 2 u32 per vec4 (fp16x4)
  actOffU32: number;    // prelu [outC4] packed as 2 u32 per vec4 (fp16x4), 0 if none
  pbOffU32: number;     // phase_bias [outC4 * 64] packed as 2 u32 per vec4, 0 if none
  inC: number;
  outC: number;
  inC4: number;
  outC4: number;
  hasAct: boolean;
  hasPb: boolean;
}

export interface PackedFastWeights {
  weightU32: Uint32Array;
  layers: Record<string, ConvLayerFastMeta>;
  seamTableOffU32: number; // offset in u32 words (stored as fp16 pairs, element-indexed via read_fp16)
  seamMaskOffU32: number;
  packMs: number;
}

/**
 * Repacks exact baked FP16 weights from `BakedSsboResult` into a GPU-optimal C4-aligned
 * layout for `clientSrV3Fast.wgsl` without altering a single FP16 bit of precision.
 *
 * Conv weight layout (in pairs of u32 = 4×FP16 = vec4<f32> for 4 output channels):
 *   index(co4, ci4, k, ciSub) = (((co4 * inC4 + ci4) * 9 + k) * 4 + ciSub) * 2  [in u32 words]
 *   stores [w(co4*4+0), w(co4*4+1), w(co4*4+2), w(co4*4+3)] as two u32s (fp16x2 + fp16x2).
 */
export function packFastWeights(baked: BakedSsboResult): PackedFastWeights {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const srcU16 = new Uint16Array(
    baked.ssboBytes.buffer,
    baked.ssboBytes.byteOffset,
    baked.ssboBytes.byteLength >>> 1,
  );
  const offs = baked.halfOffsets;
  const sc = baked.scalars;

  interface LayerSpec {
    prefix: string;
    actPrefix: string | null;
    inC: number;
    outC: number;
    hasPb: boolean;
  }

  const specs: LayerSpec[] = [];
  if (sc.isHourglass) {
    specs.push({ prefix: 'stem1', actPrefix: 'act_stem1', inC: 1, outC: sc.stemCh, hasPb: false });
    specs.push({ prefix: 'stem2', actPrefix: 'act_stem2', inC: sc.stemCh, outC: sc.stemCh, hasPb: false });
    specs.push({ prefix: 'down', actPrefix: 'act_down', inC: 4 * sc.stemCh, outC: sc.midCh, hasPb: true });
    for (let i = 0; i < sc.numBody; i++) {
      specs.push({ prefix: `body.${i}`, actPrefix: `body_acts.${i}`, inC: sc.midCh, outC: sc.midCh, hasPb: false });
    }
    if (sc.hgHasUp) {
      specs.push({ prefix: 'up', actPrefix: 'act_up', inC: sc.midCh, outC: sc.upOutCh, hasPb: false });
      specs.push({ prefix: 'refine', actPrefix: 'act_refine', inC: sc.stemCh, outC: sc.stemCh, hasPb: false });
      specs.push({ prefix: 'tail', actPrefix: null, inC: sc.stemCh, outC: 4, hasPb: false });
    } else {
      specs.push({ prefix: 'tail', actPrefix: null, inC: sc.midCh, outC: 9, hasPb: false });
    }
  } else {
    const rTail = Math.abs(sc.scale - 2.0) < 1e-4 ? 4 : 3;
    specs.push({ prefix: 'head', actPrefix: 'act_head', inC: 4, outC: sc.midCh, hasPb: true });
    for (let i = 0; i < sc.numBody; i++) {
      specs.push({ prefix: `body.${i}`, actPrefix: `body_acts.${i}`, inC: sc.midCh, outC: sc.midCh, hasPb: false });
    }
    specs.push({ prefix: 'tail', actPrefix: null, inC: sc.midCh, outC: rTail * rTail, hasPb: false });
  }

  // Pass 1: compute total u32 words needed
  let totalU32 = 0;
  for (const sp of specs) {
    const inC4 = Math.ceil(sp.inC / 4);
    const outC4 = Math.ceil(sp.outC / 4);
    totalU32 += outC4 * inC4 * 9 * 4 * 2; // weights
    totalU32 += outC4 * 2;                // bias
    if (sp.actPrefix) totalU32 += outC4 * 2;
    if (sp.hasPb) totalU32 += outC4 * 64 * 2;
  }
  const hasSeam = sc.hasSeamLut && offs['seam_lut.table'] !== undefined;
  if (hasSeam) {
    totalU32 += Math.ceil((17 * 17 * 17) / 2); // 4913 halves -> 2457 u32
    totalU32 += 32;                            // 64 halves -> 32 u32
  }

  const dstU32 = new Uint32Array(Math.max(4, totalU32));
  const dstU16 = new Uint16Array(dstU32.buffer);
  const layers: Record<string, ConvLayerFastMeta> = {};
  let cursorU32 = 0;

  for (const sp of specs) {
    const inC = sp.inC;
    const outC = sp.outC;
    const inC4 = Math.ceil(inC / 4);
    const outC4 = Math.ceil(outC / 4);

    // 1. Pack conv3x3 weights [co4][ci4][k=0..8][ciSub=0..3][coSub=0..3]
    const wOffU32 = cursorU32;
    const srcWBase = offs[`${sp.prefix}.base.reparam_conv.weight`];
    let outHalf = cursorU32 << 1;
    for (let co4 = 0; co4 < outC4; co4++) {
      const coBase = co4 * 4;
      for (let ci4 = 0; ci4 < inC4; ci4++) {
        const ciBase = ci4 * 4;
        for (let k = 0; k < 9; k++) {
          for (let ciSub = 0; ciSub < 4; ciSub++) {
            const ci = ciBase + ciSub;
            if (ci < inC) {
              for (let coSub = 0; coSub < 4; coSub++) {
                const co = coBase + coSub;
                dstU16[outHalf++] = co < outC ? srcU16[srcWBase + (co * inC + ci) * 9 + k] : 0;
              }
            } else {
              dstU16[outHalf++] = 0;
              dstU16[outHalf++] = 0;
              dstU16[outHalf++] = 0;
              dstU16[outHalf++] = 0;
            }
          }
        }
      }
    }
    cursorU32 = outHalf >>> 1;

    // 2. Pack bias [outC4][4]
    const bOffU32 = cursorU32;
    const srcBBase = offs[`${sp.prefix}.base.reparam_conv.bias`];
    outHalf = cursorU32 << 1;
    for (let co = 0; co < outC4 * 4; co++) {
      dstU16[outHalf++] = co < outC ? srcU16[srcBBase + co] : 0;
    }
    cursorU32 = outHalf >>> 1;

    // 3. Pack PReLU slopes [outC4][4]
    let actOffU32 = 0;
    if (sp.actPrefix) {
      actOffU32 = cursorU32;
      const srcActBase = offs[`${sp.actPrefix}.weight`];
      outHalf = cursorU32 << 1;
      for (let co = 0; co < outC4 * 4; co++) {
        dstU16[outHalf++] = co < outC ? srcU16[srcActBase + co] : 0;
      }
      cursorU32 = outHalf >>> 1;
    }

    // 4. Pack 8x8 Mod-16 phase_bias [outC4][y=0..7][x=0..7][coSub=0..3]
    let pbOffU32 = 0;
    if (sp.hasPb) {
      pbOffU32 = cursorU32;
      const srcPbBase = offs['phase_bias'];
      outHalf = cursorU32 << 1;
      for (let co4 = 0; co4 < outC4; co4++) {
        const coBase = co4 * 4;
        for (let py = 0; py < 8; py++) {
          for (let px = 0; px < 8; px++) {
            for (let coSub = 0; coSub < 4; coSub++) {
              const co = coBase + coSub;
              dstU16[outHalf++] = co < outC ? srcU16[srcPbBase + (co * 8 + py) * 8 + px] : 0;
            }
          }
        }
      }
      cursorU32 = outHalf >>> 1;
    }

    layers[sp.prefix] = {
      wOffU32,
      bOffU32,
      actOffU32,
      pbOffU32,
      inC,
      outC,
      inC4,
      outC4,
      hasAct: Boolean(sp.actPrefix),
      hasPb: sp.hasPb,
    };
  }

  // 5. Copy Seam3DLUT table & 8x8 mask (kept at u32-aligned start)
  let seamTableOffU32 = 0;
  let seamMaskOffU32 = 0;
  if (hasSeam) {
    seamTableOffU32 = cursorU32;
    const srcTbl = offs['seam_lut.table'];
    const tblCount = 17 * 17 * 17;
    let outHalf = cursorU32 << 1;
    for (let i = 0; i < tblCount; i++) {
      dstU16[outHalf++] = srcU16[srcTbl + i];
    }
    if ((outHalf & 1) === 1) dstU16[outHalf++] = 0;
    cursorU32 = outHalf >>> 1;

    seamMaskOffU32 = cursorU32;
    const srcMask = offs['seam_lut.seam_mask'];
    outHalf = cursorU32 << 1;
    for (let i = 0; i < 64; i++) {
      dstU16[outHalf++] = srcU16[srcMask + i];
    }
    cursorU32 = outHalf >>> 1;
  }

  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  return {
    weightU32: dstU32,
    layers,
    seamTableOffU32,
    seamMaskOffU32,
    packMs: t1 - t0,
  };
}
