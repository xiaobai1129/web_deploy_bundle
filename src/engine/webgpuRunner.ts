import { bakeCsr1ToSsbo, parseBinaryBlob, type BakedSsboResult, type ParsedBlob } from '../core/csr1';
import clientSrWgsl from './shaders/clientSrV3.wgsl?raw';
import colorCompareWgsl from './shaders/colorAndCompare.wgsl?raw';

export type CompareViewMode = 0 | 1 | 2 | 3; // 0=split, 1=sr, 2=bicubic, 3=heatmap

export interface DisplayOptions {
  splitX: number;
  viewMode: CompareViewMode;
  zoom: number;
  centerX: number;
  centerY: number;
}

export interface GpuDeviceSummary {
  vendor: string;
  architecture: string;
  description: string;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

interface DispatchSpec {
  op: number;
  inC: number;
  inH: number;
  inW: number;
  outC: number;
  outH: number;
  outW: number;
  bufA: GPUBuffer;
  bufB: GPUBuffer;
  dst: GPUBuffer;
  wOff?: number;
  bOff?: number;
  pbOff?: number;
  actOff?: number;
  hasPb?: number;
  hasAct?: number;
  scaleR?: number;
  addB?: number;
  alpha?: number;
  beta?: number;
  delta?: number;
}

interface FrameWorkspace {
  srcW: number;
  srcH: number;
  padW: number;
  padH: number;
  srPadW: number;
  srPadH: number;
  outW: number;
  outH: number;
  scratchBytes: number;
  srcTex: GPUTexture;
  srTex: GPUTexture;
  bicubicTex: GPUTexture;
  xBuf: GPUBuffer;
  baseBuf: GPUBuffer;
  resBuf: GPUBuffer;
  fHeadBuf: GPUBuffer;
  fMidBuf: GPUBuffer;
  s2Buf: GPUBuffer;
  pingBuf: GPUBuffer;
  pongBuf: GPUBuffer;
  rgb2yUniform: GPUBuffer;
  y2rgbUniform: GPUBuffer;
  rgb2yBindGroup: GPUBindGroup;
  y2rgbSrBindGroup: GPUBindGroup;
  y2rgbBiBindGroup: GPUBindGroup;
}

const MAX_DISPATCH_SLOTS = 32;
const UNIFORM_STRIDE = 256; // WebGPU minUniformBufferOffsetAlignment

function computeRequiredScratchBytes(
  padH: number,
  padW: number,
  srPadH: number,
  srPadW: number,
  baked: BakedSsboResult,
): number {
  const uh = padH >>> 1;
  const uw = padW >>> 1;
  const s = baked.scalars;
  let maxFloats = srPadH * srPadW;
  if (s.isHourglass) {
    const maxMidC = Math.max(4 * s.stemCh, s.midCh, s.upOutCh, 9);
    maxFloats = Math.max(maxFloats, maxMidC * uh * uw, Math.max(s.stemCh, 4) * padH * padW);
  } else {
    const rTail = Math.abs(s.scale - 2.0) < 1e-4 ? 4 : 3;
    const maxMidC = Math.max(4, s.midCh, rTail * rTail);
    maxFloats = Math.max(maxFloats, maxMidC * uh * uw);
  }
  return Math.max(256, maxFloats * 4);
}

export class CodecSREngine {
  readonly device: GPUDevice;
  readonly canvasFormat: GPUTextureFormat;
  readonly gpuInfo: GpuDeviceSummary;

  private readonly srBindGroupLayout: GPUBindGroupLayout;
  private readonly srPipeline: GPUComputePipeline;
  private readonly rgb2yPipeline: GPUComputePipeline;
  private readonly y2rgbPipeline: GPUComputePipeline;
  private readonly displayPipeline: GPURenderPipeline;
  private readonly linearSampler: GPUSampler;
  private readonly nearestSampler: GPUSampler;

  private readonly paramsBuf: GPUBuffer;
  private readonly paramsStaging: ArrayBuffer;
  private readonly paramsI32: Int32Array;
  private readonly paramsU32: Uint32Array;
  private readonly paramsF32: Float32Array;
  private readonly displayUniformBuf: GPUBuffer;

  private readonly blobCache = new Map<string, ParsedBlob>();
  private weightBuf: GPUBuffer | null = null;
  private activeBaked: BakedSsboResult | null = null;
  private activeBlob: ParsedBlob | null = null;
  private workspace: FrameWorkspace | null = null;
  private fallback2dCanvas: HTMLCanvasElement | null = null;
  private fallback2dCtx: CanvasRenderingContext2D | null = null;

  private constructor(
    device: GPUDevice,
    canvasFormat: GPUTextureFormat,
    gpuInfo: GpuDeviceSummary,
  ) {
    this.device = device;
    this.canvasFormat = canvasFormat;
    this.gpuInfo = gpuInfo;

    const srModule = device.createShaderModule({
      label: 'client_sr_v3.wgsl',
      code: clientSrWgsl,
    });
    const colorModule = device.createShaderModule({
      label: 'color_and_compare.wgsl',
      code: colorCompareWgsl,
    });

    this.srBindGroupLayout = device.createBindGroupLayout({
      label: 'sr_bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 80 },
        },
      ],
    });

    this.srPipeline = device.createComputePipeline({
      label: 'sr_compute_pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.srBindGroupLayout] }),
      compute: { module: srModule, entryPoint: 'main' },
    });

    this.rgb2yPipeline = device.createComputePipeline({
      label: 'rgb_to_y_pipeline',
      layout: 'auto',
      compute: { module: colorModule, entryPoint: 'rgb_to_y_main' },
    });

    this.y2rgbPipeline = device.createComputePipeline({
      label: 'ycbcr_to_rgba_pipeline',
      layout: 'auto',
      compute: { module: colorModule, entryPoint: 'ycbcr_to_rgba_main' },
    });

    this.displayPipeline = device.createRenderPipeline({
      label: 'compare_display_pipeline',
      layout: 'auto',
      vertex: { module: colorModule, entryPoint: 'display_vs' },
      fragment: {
        module: colorModule,
        entryPoint: 'display_fs',
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.linearSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.nearestSampler = device.createSampler({
      magFilter: 'nearest',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.paramsStaging = new ArrayBuffer(MAX_DISPATCH_SLOTS * UNIFORM_STRIDE);
    this.paramsI32 = new Int32Array(this.paramsStaging);
    this.paramsU32 = new Uint32Array(this.paramsStaging);
    this.paramsF32 = new Float32Array(this.paramsStaging);
    this.paramsBuf = device.createBuffer({
      label: 'sr_dispatch_uniforms',
      size: MAX_DISPATCH_SLOTS * UNIFORM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.displayUniformBuf = device.createBuffer({
      label: 'display_uniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  static async create(): Promise<CodecSREngine> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('当前浏览器不支持 WebGPU。请使用 Chrome 113+、Edge 113+ 或 Safari 26+ 桌面/移动浏览器。');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('未能获取到可用的 WebGPU 图形适配器，请检查浏览器硬件加速设置。');
    }

    const maxStorage = Math.min(adapter.limits.maxStorageBufferBindingSize, 512 * 1024 * 1024);
    const maxBuffer = Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024);
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: maxStorage,
        maxBufferSize: maxBuffer,
      },
    });

    const adapterInfo = (adapter as unknown as { info?: Partial<GPUAdapterInfo> }).info;
    const gpuInfo: GpuDeviceSummary = {
      vendor: adapterInfo?.vendor || 'WebGPU',
      architecture: adapterInfo?.architecture || '',
      description: adapterInfo?.description || adapterInfo?.device || 'Local GPU',
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
    };

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    return new CodecSREngine(device, canvasFormat, gpuInfo);
  }

  async loadModel(weightUrl: string, prior7d: number[] | null): Promise<BakedSsboResult> {
    let blob = this.blobCache.get(weightUrl);
    if (!blob) {
      const resp = await fetch(weightUrl);
      if (!resp.ok) {
        throw new Error(`加载权重失败 (${resp.status}): ${weightUrl}`);
      }
      blob = parseBinaryBlob(await resp.arrayBuffer());
      this.blobCache.set(weightUrl, blob);
    }
    this.activeBlob = blob;
    return this.bakeAndUpload(blob, prior7d);
  }

  rebakePrior(prior7d: number[] | null): BakedSsboResult {
    if (!this.activeBlob) {
      throw new Error('尚未加载模型权重');
    }
    return this.bakeAndUpload(this.activeBlob, prior7d);
  }

  getActiveBaked(): BakedSsboResult | null {
    return this.activeBaked;
  }

  getCurrentOutputSize(): { outW: number; outH: number } | null {
    if (!this.workspace) return null;
    return { outW: this.workspace.outW, outH: this.workspace.outH };
  }

  private bakeAndUpload(blob: ParsedBlob, prior7d: number[] | null): BakedSsboResult {
    const baked = bakeCsr1ToSsbo(blob, prior7d);
    const neededSize = Math.max(4, baked.ssboBytes.byteLength);

    if (!this.weightBuf || this.weightBuf.size !== neededSize) {
      this.weightBuf?.destroy();
      this.weightBuf = this.device.createBuffer({
        label: 'csr1_baked_weights',
        size: neededSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(
      this.weightBuf,
      0,
      baked.ssboBytes.buffer,
      baked.ssboBytes.byteOffset,
      baked.ssboBytes.byteLength,
    );
    this.activeBaked = baked;
    return baked;
  }

  private writeDispatchSlot(slot: number, spec: DispatchSpec): void {
    const base = (slot * UNIFORM_STRIDE) >>> 2;
    this.paramsI32[base + 0] = spec.op;
    this.paramsI32[base + 1] = spec.inC;
    this.paramsI32[base + 2] = spec.inH;
    this.paramsI32[base + 3] = spec.inW;
    this.paramsI32[base + 4] = spec.outC;
    this.paramsI32[base + 5] = spec.outH;
    this.paramsI32[base + 6] = spec.outW;
    this.paramsU32[base + 7] = spec.wOff ?? 0;
    this.paramsU32[base + 8] = spec.bOff ?? 0;
    this.paramsU32[base + 9] = spec.pbOff ?? 0;
    this.paramsU32[base + 10] = spec.actOff ?? 0;
    this.paramsI32[base + 11] = spec.hasPb ?? 0;
    this.paramsI32[base + 12] = spec.hasAct ?? 0;
    this.paramsI32[base + 13] = spec.scaleR ?? 1;
    this.paramsI32[base + 14] = spec.addB ?? 0;
    this.paramsI32[base + 15] = 0;
    this.paramsF32[base + 16] = spec.alpha ?? 0.0;
    this.paramsF32[base + 17] = spec.beta ?? 0.0;
    this.paramsF32[base + 18] = spec.delta ?? 0.0;
    this.paramsF32[base + 19] = 0.0;
  }

  /**
   * Encodes all CodecSR V3.5 compute dispatches into `encoder` matching
   * `verify_compute_shader.py` lines 470-559 identically.
   * Returns number of uniform slots written to `this.paramsStaging`.
   */
  private encodeSrPipeline(
    encoder: GPUCommandEncoder,
    weightBuf: GPUBuffer,
    baked: BakedSsboResult,
    lrH: number,
    lrW: number,
    outH: number,
    outW: number,
    bufs: {
      xBuf: GPUBuffer;
      baseBuf: GPUBuffer;
      resBuf: GPUBuffer;
      fHeadBuf: GPUBuffer;
      fMidBuf: GPUBuffer;
      s2Buf: GPUBuffer;
      pingBuf: GPUBuffer;
      pongBuf: GPUBuffer;
    },
  ): number {
    const offs = baked.halfOffsets;
    const sc = baked.scalars;
    const uh = lrH >>> 1;
    const uw = lrW >>> 1;
    const rTail = Math.abs(sc.scale - 2.0) < 1e-4 ? 4 : 3;
    let slot = 0;

    const dispatch = (spec: DispatchSpec): void => {
      if (slot >= MAX_DISPATCH_SLOTS) {
        throw new Error(`Exceeded MAX_DISPATCH_SLOTS (${MAX_DISPATCH_SLOTS})`);
      }
      this.writeDispatchSlot(slot, spec);
      const bg = this.device.createBindGroup({
        layout: this.srBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: weightBuf } },
          { binding: 1, resource: { buffer: spec.bufA } },
          { binding: 2, resource: { buffer: spec.bufB } },
          { binding: 3, resource: { buffer: spec.dst } },
          { binding: 4, resource: { buffer: this.paramsBuf, size: 80 } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.srPipeline);
      pass.setBindGroup(0, bg, [slot * UNIFORM_STRIDE]);
      pass.dispatchWorkgroups(Math.ceil(spec.outW / 16), Math.ceil(spec.outH / 16), 1);
      pass.end();
      slot++;
    };

    const conv = (
      prefix: string,
      actPrefix: string | null,
      inC: number,
      outC: number,
      h: number,
      w: number,
      inBuf: GPUBuffer,
      outBuf: GPUBuffer,
      hasPb = false,
    ): void => {
      dispatch({
        op: 2,
        inC,
        inH: h,
        inW: w,
        outC,
        outH: h,
        outW: w,
        bufA: inBuf,
        bufB: bufs.xBuf,
        dst: outBuf,
        wOff: offs[`${prefix}.base.reparam_conv.weight`],
        bOff: offs[`${prefix}.base.reparam_conv.bias`],
        pbOff: hasPb ? offs['phase_bias'] : 0,
        actOff: actPrefix !== null ? offs[`${actPrefix}.weight`] : 0,
        hasPb: hasPb ? 1 : 0,
        hasAct: actPrefix !== null ? 1 : 0,
      });
    };

    // 0. Bicubic base (Keys a = -0.75)
    dispatch({
      op: 0,
      inC: 1,
      inH: lrH,
      inW: lrW,
      outC: 1,
      outH,
      outW,
      bufA: bufs.xBuf,
      bufB: bufs.xBuf,
      dst: bufs.baseBuf,
    });

    const midIdx = Math.floor(sc.numBody / 2);

    if (sc.isHourglass) {
      const stemCh = sc.stemCh;
      const midCh = sc.midCh;
      conv('stem1', 'act_stem1', 1, stemCh, lrH, lrW, bufs.xBuf, bufs.pongBuf, false);
      conv('stem2', 'act_stem2', stemCh, stemCh, lrH, lrW, bufs.pongBuf, bufs.s2Buf, false);
      dispatch({
        op: 1,
        inC: stemCh,
        inH: lrH,
        inW: lrW,
        outC: 4 * stemCh,
        outH: uh,
        outW: uw,
        bufA: bufs.s2Buf,
        bufB: bufs.xBuf,
        dst: bufs.pingBuf,
      });
      conv('down', 'act_down', 4 * stemCh, midCh, uh, uw, bufs.pingBuf, bufs.fHeadBuf, true);

      let curIn = bufs.fHeadBuf;
      let curOut = bufs.pingBuf;
      for (let idx = 0; idx < sc.numBody; idx++) {
        conv(`body.${idx}`, `body_acts.${idx}`, midCh, midCh, uh, uw, curIn, curOut, false);
        if (idx === midIdx) {
          encoder.copyBufferToBuffer(curOut, 0, bufs.fMidBuf, 0, midCh * uh * uw * 4);
        }
        curIn = curOut;
        curOut = curOut === bufs.pingBuf ? bufs.pongBuf : bufs.pingBuf;
      }

      dispatch({
        op: 3,
        inC: midCh,
        inH: uh,
        inW: uw,
        outC: midCh,
        outH: uh,
        outW: uw,
        bufA: bufs.fHeadBuf,
        bufB: bufs.fMidBuf,
        dst: curIn,
        alpha: sc.skipAlpha,
        beta: sc.skipBeta,
      });

      if (sc.hgHasUp) {
        const upOutC = sc.upOutCh;
        conv('up', 'act_up', midCh, upOutC, uh, uw, curIn, curOut, false);
        dispatch({
          op: 4,
          inC: upOutC,
          inH: uh,
          inW: uw,
          outC: stemCh,
          outH: lrH,
          outW: lrW,
          bufA: curOut,
          bufB: bufs.s2Buf,
          dst: bufs.fHeadBuf,
          scaleR: 2,
          addB: 1,
        });
        conv('refine', 'act_refine', stemCh, stemCh, lrH, lrW, bufs.fHeadBuf, bufs.pingBuf, false);
        conv('tail', null, stemCh, 4, lrH, lrW, bufs.pingBuf, bufs.pongBuf, false);
        dispatch({
          op: 4,
          inC: 4,
          inH: lrH,
          inW: lrW,
          outC: 1,
          outH,
          outW,
          bufA: bufs.pongBuf,
          bufB: bufs.xBuf,
          dst: bufs.resBuf,
          scaleR: 2,
          addB: 0,
        });
      } else {
        if (sc.hgHasHrSkip) {
          dispatch({
            op: 1,
            inC: stemCh,
            inH: lrH,
            inW: lrW,
            outC: 4 * stemCh,
            outH: uh,
            outW: uw,
            bufA: bufs.s2Buf,
            bufB: bufs.xBuf,
            dst: curOut,
          });
          dispatch({
            op: 3,
            inC: midCh,
            inH: uh,
            inW: uw,
            outC: midCh,
            outH: uh,
            outW: uw,
            bufA: curOut,
            bufB: curOut,
            dst: curIn,
            alpha: sc.skipHr,
            beta: 0.0,
          });
        }
        conv('tail', null, midCh, 9, uh, uw, curIn, curOut, false);
        dispatch({
          op: 4,
          inC: 9,
          inH: uh,
          inW: uw,
          outC: 1,
          outH,
          outW,
          bufA: curOut,
          bufB: bufs.xBuf,
          dst: bufs.resBuf,
          scaleR: 3,
          addB: 0,
        });
      }
    } else {
      const midCh = sc.midCh;
      dispatch({
        op: 1,
        inC: 1,
        inH: lrH,
        inW: lrW,
        outC: 4,
        outH: uh,
        outW: uw,
        bufA: bufs.xBuf,
        bufB: bufs.xBuf,
        dst: bufs.pingBuf,
      });
      conv('head', 'act_head', 4, midCh, uh, uw, bufs.pingBuf, bufs.fHeadBuf, true);

      let curIn = bufs.fHeadBuf;
      let curOut = bufs.pingBuf;
      for (let idx = 0; idx < sc.numBody; idx++) {
        conv(`body.${idx}`, `body_acts.${idx}`, midCh, midCh, uh, uw, curIn, curOut, false);
        if (idx === midIdx) {
          encoder.copyBufferToBuffer(curOut, 0, bufs.fMidBuf, 0, midCh * uh * uw * 4);
        }
        curIn = curOut;
        curOut = curOut === bufs.pingBuf ? bufs.pongBuf : bufs.pingBuf;
      }

      dispatch({
        op: 3,
        inC: midCh,
        inH: uh,
        inW: uw,
        outC: midCh,
        outH: uh,
        outW: uw,
        bufA: bufs.fHeadBuf,
        bufB: bufs.fMidBuf,
        dst: curIn,
        alpha: sc.skipAlpha,
        beta: sc.skipBeta,
      });

      conv('tail', null, midCh, rTail * rTail, uh, uw, curIn, curOut, false);
      dispatch({
        op: 4,
        inC: rTail * rTail,
        inH: uh,
        inW: uw,
        outC: 1,
        outH,
        outW,
        bufA: curOut,
        bufB: bufs.xBuf,
        dst: bufs.resBuf,
        scaleR: rTail,
        addB: 0,
      });
    }

    if (sc.hasSeamLut && offs['seam_lut.table'] !== undefined) {
      dispatch({
        op: 5,
        inC: 1,
        inH: lrH,
        inW: lrW,
        outC: 1,
        outH,
        outW,
        bufA: bufs.xBuf,
        bufB: bufs.xBuf,
        dst: bufs.resBuf,
        wOff: offs['seam_lut.table'],
        bOff: offs['seam_lut.seam_mask'],
      });
    }

    dispatch({
      op: 6,
      inC: 1,
      inH: lrH,
      inW: lrW,
      outC: 1,
      outH,
      outW,
      bufA: bufs.baseBuf,
      bufB: bufs.xBuf,
      dst: bufs.resBuf,
      delta: sc.trustDelta,
    });

    return slot;
  }

  /**
   * Direct float32 luma patch execution for pixel-exact Golden Vector verification (PSNR >= 70 dB).
   */
  async runLumaPatch(
    inputF32: Float32Array,
    lrH: number,
    lrW: number,
    baked: BakedSsboResult,
  ): Promise<Float32Array> {
    const outH = Math.round(lrH * baked.scalars.scale);
    const outW = Math.round(lrW * baked.scalars.scale);
    const scratchBytes = computeRequiredScratchBytes(lrH, lrW, outH, outW, baked);

    const wBuf = this.device.createBuffer({
      size: Math.max(4, baked.ssboBytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      wBuf,
      0,
      baked.ssboBytes.buffer,
      baked.ssboBytes.byteOffset,
      baked.ssboBytes.byteLength,
    );

    const xBuf = this.device.createBuffer({
      size: lrH * lrW * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      xBuf,
      0,
      inputF32.buffer,
      inputF32.byteOffset,
      inputF32.byteLength,
    );

    const makeBuf = (size: number, extra = 0) =>
      this.device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra,
      });

    const baseBuf = makeBuf(outH * outW * 4);
    const resBuf = makeBuf(outH * outW * 4);
    const fHeadBuf = makeBuf(scratchBytes);
    const fMidBuf = makeBuf(scratchBytes);
    const s2Buf = makeBuf(scratchBytes);
    const pingBuf = makeBuf(scratchBytes);
    const pongBuf = makeBuf(scratchBytes);
    const readbackBuf = this.device.createBuffer({
      size: outH * outW * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    try {
      const encoder = this.device.createCommandEncoder();
      const slots = this.encodeSrPipeline(encoder, wBuf, baked, lrH, lrW, outH, outW, {
        xBuf,
        baseBuf,
        resBuf,
        fHeadBuf,
        fMidBuf,
        s2Buf,
        pingBuf,
        pongBuf,
      });
      encoder.copyBufferToBuffer(resBuf, 0, readbackBuf, 0, outH * outW * 4);

      this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsStaging, 0, slots * UNIFORM_STRIDE);
      this.device.queue.submit([encoder.finish()]);

      await readbackBuf.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(readbackBuf.getMappedRange().slice(0));
      readbackBuf.unmap();
      return copy;
    } finally {
      wBuf.destroy();
      xBuf.destroy();
      baseBuf.destroy();
      resBuf.destroy();
      fHeadBuf.destroy();
      fMidBuf.destroy();
      s2Buf.destroy();
      pingBuf.destroy();
      pongBuf.destroy();
      readbackBuf.destroy();
    }
  }

  private ensureWorkspace(srcW: number, srcH: number, baked: BakedSsboResult): FrameWorkspace {
    const padW = srcW + (srcW & 1);
    const padH = srcH + (srcH & 1);
    const srPadW = Math.round(padW * baked.scalars.scale);
    const srPadH = Math.round(padH * baked.scalars.scale);
    const outW = Math.round(srcW * baked.scalars.scale);
    const outH = Math.round(srcH * baked.scalars.scale);
    const scratchBytes = computeRequiredScratchBytes(padH, padW, srPadH, srPadW, baked);

    if (
      this.workspace &&
      this.workspace.srcW === srcW &&
      this.workspace.srcH === srcH &&
      this.workspace.outW === outW &&
      this.workspace.outH === outH &&
      this.workspace.scratchBytes >= scratchBytes
    ) {
      return this.workspace;
    }

    if (scratchBytes > this.device.limits.maxStorageBufferBindingSize) {
      throw new Error(
        `当前视频分辨率 (${srcW}×${srcH}) 在此档位下所需显存缓冲 (${Math.round(scratchBytes / 1048576)} MB) 超出设备单 Buffer 上限 (${Math.round(this.device.limits.maxStorageBufferBindingSize / 1048576)} MB)，请切换 Client-L/S 或降低输入分辨率。`,
      );
    }

    this.destroyWorkspace();

    const srcTex = this.device.createTexture({
      label: 'src_video_tex',
      size: [srcW, srcH, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const srTex = this.device.createTexture({
      label: 'sr_rgba_tex',
      size: [outW, outH, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    const bicubicTex = this.device.createTexture({
      label: 'bicubic_rgba_tex',
      size: [outW, outH, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });

    const mkBuf = (label: string, size: number) =>
      this.device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

    const xBuf = mkBuf('x_luma', padW * padH * 4);
    const baseBuf = mkBuf('base_bicubic_luma', srPadW * srPadH * 4);
    const resBuf = mkBuf('res_sr_luma', srPadW * srPadH * 4);
    const fHeadBuf = mkBuf('f_head', scratchBytes);
    const fMidBuf = mkBuf('f_mid', scratchBytes);
    const s2Buf = mkBuf('s2', scratchBytes);
    const pingBuf = mkBuf('ping', scratchBytes);
    const pongBuf = mkBuf('pong', scratchBytes);

    const rgb2yUniform = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(rgb2yUniform, 0, new Uint32Array([srcW, srcH, padW, padH]));

    const y2rgbUniform = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(y2rgbUniform, 0, new Uint32Array([outW, outH, srPadW, 0]));

    const rgb2yBindGroup = this.device.createBindGroup({
      layout: this.rgb2yPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: rgb2yUniform } },
      ],
    });

    const y2rgbSrBindGroup = this.device.createBindGroup({
      layout: this.y2rgbPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: resBuf } },
        { binding: 1, resource: srcTex.createView() },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: srTex.createView() },
        { binding: 4, resource: { buffer: y2rgbUniform } },
      ],
    });

    const y2rgbBiBindGroup = this.device.createBindGroup({
      layout: this.y2rgbPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: baseBuf } },
        { binding: 1, resource: srcTex.createView() },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: bicubicTex.createView() },
        { binding: 4, resource: { buffer: y2rgbUniform } },
      ],
    });

    this.workspace = {
      srcW,
      srcH,
      padW,
      padH,
      srPadW,
      srPadH,
      outW,
      outH,
      scratchBytes,
      srcTex,
      srTex,
      bicubicTex,
      xBuf,
      baseBuf,
      resBuf,
      fHeadBuf,
      fMidBuf,
      s2Buf,
      pingBuf,
      pongBuf,
      rgb2yUniform,
      y2rgbUniform,
      rgb2yBindGroup,
      y2rgbSrBindGroup,
      y2rgbBiBindGroup,
    };
    return this.workspace;
  }

  private uploadExternalSource(
    source: GPUCopyExternalImageSource,
    dstTex: GPUTexture,
    srcW: number,
    srcH: number,
  ): void {
    try {
      this.device.queue.copyExternalImageToTexture(
        { source },
        { texture: dstTex },
        [srcW, srcH, 1],
      );
    } catch (err) {
      // Chromium can transiently reject paused/offscreen HTMLVideoElement before compositor backing exists.
      // Fall back transparently via a reusable 2D staging canvas so frame 0 / seeked frames always render.
      if (typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement) {
        if (!this.fallback2dCanvas) {
          this.fallback2dCanvas = document.createElement('canvas');
          this.fallback2dCtx = this.fallback2dCanvas.getContext('2d', { willReadFrequently: false });
        }
        if (this.fallback2dCanvas && this.fallback2dCtx) {
          if (this.fallback2dCanvas.width !== srcW || this.fallback2dCanvas.height !== srcH) {
            this.fallback2dCanvas.width = srcW;
            this.fallback2dCanvas.height = srcH;
          }
          this.fallback2dCtx.drawImage(source, 0, 0, srcW, srcH);
          this.device.queue.copyExternalImageToTexture(
            { source: this.fallback2dCanvas },
            { texture: dstTex },
            [srcW, srcH, 1],
          );
          return;
        }
      }
      throw err;
    }
  }

  /**
   * Uploads one video frame, runs full CodecSR V3.5 WebGPU inference, and writes SR (+ optional Bicubic) RGBA textures.
   */
  processFrame(
    source: GPUCopyExternalImageSource,
    srcW: number,
    srcH: number,
    includeBicubic = true,
  ): { outW: number; outH: number } {
    if (!this.activeBaked || !this.weightBuf) {
      throw new Error('模型尚未初始化');
    }
    const ws = this.ensureWorkspace(srcW, srcH, this.activeBaked);

    this.uploadExternalSource(source, ws.srcTex, srcW, srcH);

    const encoder = this.device.createCommandEncoder();

    // 1. RGB -> Full-Range BT.601 Luma Y
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.rgb2yPipeline);
      pass.setBindGroup(0, ws.rgb2yBindGroup);
      pass.dispatchWorkgroups(Math.ceil(ws.padW / 16), Math.ceil(ws.padH / 16), 1);
      pass.end();
    }

    // 2. CodecSR V3.5 super-resolution (computes baseBuf=bicubic Y and resBuf=SR Y)
    const slots = this.encodeSrPipeline(
      encoder,
      this.weightBuf,
      this.activeBaked,
      ws.padH,
      ws.padW,
      ws.srPadH,
      ws.srPadW,
      {
        xBuf: ws.xBuf,
        baseBuf: ws.baseBuf,
        resBuf: ws.resBuf,
        fHeadBuf: ws.fHeadBuf,
        fMidBuf: ws.fMidBuf,
        s2Buf: ws.s2Buf,
        pingBuf: ws.pingBuf,
        pongBuf: ws.pongBuf,
      },
    );

    // 3. Y_sr + upsampled CbCr -> RGBA
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.y2rgbPipeline);
      pass.setBindGroup(0, ws.y2rgbSrBindGroup);
      pass.dispatchWorkgroups(Math.ceil(ws.outW / 16), Math.ceil(ws.outH / 16), 1);
      pass.end();
    }

    // 4. Optional Y_bicubic + upsampled CbCr -> RGBA (for split-screen & heatmap comparison)
    if (includeBicubic) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.y2rgbPipeline);
      pass.setBindGroup(0, ws.y2rgbBiBindGroup);
      pass.dispatchWorkgroups(Math.ceil(ws.outW / 16), Math.ceil(ws.outH / 16), 1);
      pass.end();
    }

    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsStaging, 0, slots * UNIFORM_STRIDE);
    this.device.queue.submit([encoder.finish()]);

    return { outW: ws.outW, outH: ws.outH };
  }

  /**
   * Renders the most recently processed SR & Bicubic textures onto a WebGPU canvas context.
   * Zero re-inference when dragging the split slider or inspecting zoomed regions.
   */
  renderToCanvas(ctx: GPUCanvasContext, opts: DisplayOptions): void {
    const ws = this.workspace;
    if (!ws) return;

    const uBuf = new ArrayBuffer(32);
    const f32 = new Float32Array(uBuf);
    const u32 = new Uint32Array(uBuf);
    f32[0] = opts.splitX;
    u32[1] = opts.viewMode;
    f32[2] = opts.zoom;
    f32[3] = 0;
    f32[4] = opts.centerX;
    f32[5] = opts.centerY;
    f32[6] = 0;
    f32[7] = 0;
    this.device.queue.writeBuffer(this.displayUniformBuf, 0, uBuf);

    const sampler = opts.zoom >= 2.0 ? this.nearestSampler : this.linearSampler;
    const bg = this.device.createBindGroup({
      layout: this.displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: ws.srTex.createView() },
        { binding: 1, resource: ws.bicubicTex.createView() },
        { binding: 2, resource: sampler },
        { binding: 3, resource: { buffer: this.displayUniformBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.035, g: 0.043, b: 0.063, a: 1.0 },
        },
      ],
    });
    pass.setPipeline(this.displayPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3, 1, 0, 0);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Wait until GPU finishes all submitted work (used for accurate latency telemetry / export pacing). */
  async waitGpuIdle(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  private destroyWorkspace(): void {
    if (!this.workspace) return;
    const ws = this.workspace;
    ws.srcTex.destroy();
    ws.srTex.destroy();
    ws.bicubicTex.destroy();
    ws.xBuf.destroy();
    ws.baseBuf.destroy();
    ws.resBuf.destroy();
    ws.fHeadBuf.destroy();
    ws.fMidBuf.destroy();
    ws.s2Buf.destroy();
    ws.pingBuf.destroy();
    ws.pongBuf.destroy();
    ws.rgb2yUniform.destroy();
    ws.y2rgbUniform.destroy();
    this.workspace = null;
  }

  destroy(): void {
    this.destroyWorkspace();
    this.weightBuf?.destroy();
    this.paramsBuf.destroy();
    this.displayUniformBuf.destroy();
  }
}
