import { bakeCsr1ToSsbo, parseBinaryBlob, type BakedSsboResult, type ParsedBlob } from '../core/csr1';
import { packFastWeights, type PackedFastWeights } from '../core/packFastWeights';
import clientSrFastWgsl from './shaders/clientSrV3Fast.wgsl?raw';
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

type FastKernelKey =
  | 'bicubic'
  | 'unshuffle2_scalar'
  | 'unshuffle2_c4'
  | 'conv3x3_scalar'
  | 'conv3x3_c4'
  | 'span_combine_c4'
  | 'shuffle_to_scalar'
  | 'shuffle2_c4'
  | 'seam_lut_lr'
  | 'add_seam_trust_clamp';

interface CachedDispatch {
  kind: 'compute';
  kernel: FastKernelKey;
  bindGroup: GPUBindGroup;
  slotOffset: number;
  gx: number;
  gy: number;
  gz: number;
}

interface CachedCopy {
  kind: 'copy';
  src: GPUBuffer;
  dst: GPUBuffer;
  bytes: number;
}

type PlanStep = CachedDispatch | CachedCopy;

interface FrameWorkspace {
  srcW: number;
  srcH: number;
  workW: number;
  workH: number;
  padW: number;
  padH: number;
  srPadW: number;
  srPadH: number;
  outW: number;
  outH: number;
  scratchBytes: number;
  modelSignature: string;
  srcTex: GPUTexture;
  srTex: GPUTexture;
  bicubicTex: GPUTexture;
  xBuf: GPUBuffer;
  seamLrBuf: GPUBuffer;
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
  displayLinearBindGroup: GPUBindGroup;
  displayNearestBindGroup: GPUBindGroup;
  planSteps: PlanStep[];
  planSlotCount: number;
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
    const stemC4 = Math.ceil(s.stemCh / 4) * 4;
    const midC4 = Math.ceil(s.midCh / 4) * 4;
    const upC4 = Math.ceil(s.upOutCh / 4) * 4;
    const maxMidC = Math.max(4 * stemC4, midC4, upC4, 12);
    maxFloats = Math.max(maxFloats, maxMidC * uh * uw, Math.max(stemC4, 4) * padH * padW);
  } else {
    const rTail = Math.abs(s.scale - 2.0) < 1e-4 ? 4 : 3;
    const tailC4 = Math.ceil((rTail * rTail) / 4) * 4;
    const midC4 = Math.ceil(s.midCh / 4) * 4;
    const maxMidC = Math.max(4, midC4, tailC4);
    maxFloats = Math.max(maxFloats, maxMidC * uh * uw);
  }
  return Math.max(256, maxFloats * 4);
}

export class CodecSREngine {
  readonly device: GPUDevice;
  readonly canvasFormat: GPUTextureFormat;
  readonly gpuInfo: GpuDeviceSummary;

  private readonly srBindGroupLayout: GPUBindGroupLayout;
  private readonly kernels: Record<FastKernelKey, GPUComputePipeline>;
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
  private readonly displayUniformStaging = new ArrayBuffer(32);

  private readonly blobCache = new Map<string, ParsedBlob>();
  private weightBuf: GPUBuffer | null = null;
  private weightBufGeneration = 0;
  private activeBaked: BakedSsboResult | null = null;
  private activeFastWeights: PackedFastWeights | null = null;
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
      label: 'client_sr_v3_fast.wgsl',
      code: clientSrFastWgsl,
    });
    const colorModule = device.createShaderModule({
      label: 'color_and_compare.wgsl',
      code: colorCompareWgsl,
    });

    this.srBindGroupLayout = device.createBindGroupLayout({
      label: 'sr_fast_bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 },
        },
      ],
    });

    const srLayout = device.createPipelineLayout({ bindGroupLayouts: [this.srBindGroupLayout] });
    const mkKernel = (entryPoint: string) =>
      device.createComputePipeline({
        label: entryPoint,
        layout: srLayout,
        compute: { module: srModule, entryPoint },
      });

    this.kernels = {
      bicubic: mkKernel('op_bicubic_main'),
      unshuffle2_scalar: mkKernel('op_unshuffle2_scalar_main'),
      unshuffle2_c4: mkKernel('op_unshuffle2_c4_main'),
      conv3x3_scalar: mkKernel('op_conv3x3_scalar_to_c4_main'),
      conv3x3_c4: mkKernel('op_conv3x3_c4_main'),
      span_combine_c4: mkKernel('op_span_combine_c4_main'),
      shuffle_to_scalar: mkKernel('op_shuffle_to_scalar_main'),
      shuffle2_c4: mkKernel('op_shuffle2_c4_main'),
      seam_lut_lr: mkKernel('op_seam_lut_lr_main'),
      add_seam_trust_clamp: mkKernel('op_add_seam_trust_clamp_main'),
    };

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
      label: 'sr_fast_dispatch_uniforms',
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
    const fast = packFastWeights(baked);
    const neededSize = Math.max(16, fast.weightU32.byteLength);

    if (!this.weightBuf || this.weightBuf.size !== neededSize) {
      this.weightBuf?.destroy();
      this.weightBuf = this.device.createBuffer({
        label: 'csr1_fast_weights',
        size: neededSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.weightBufGeneration++;
    }
    this.device.queue.writeBuffer(
      this.weightBuf,
      0,
      fast.weightU32.buffer,
      fast.weightU32.byteOffset,
      fast.weightU32.byteLength,
    );
    this.activeBaked = baked;
    this.activeFastWeights = fast;

    // If workspace exists for same model geometry, refresh its scalar uniforms (skipAlpha/Beta/Hr/trustDelta)
    if (this.workspace) {
      const sig = this.getModelSignature(baked);
      if (this.workspace.modelSignature === sig) {
        const plan = this.buildExecutionPlan(
          this.weightBuf,
          baked,
          fast,
          this.workspace.padH,
          this.workspace.padW,
          this.workspace.srPadH,
          this.workspace.srPadW,
          this.workspace,
        );
        this.workspace.planSteps = plan.steps;
        this.workspace.planSlotCount = plan.slotCount;
        this.device.queue.writeBuffer(
          this.paramsBuf,
          0,
          this.paramsStaging,
          0,
          plan.slotCount * UNIFORM_STRIDE,
        );
      } else {
        this.workspace.modelSignature = '';
      }
    }

    return {
      ...baked,
      bakeMs: baked.bakeMs + fast.packMs,
    };
  }

  private getModelSignature(baked: BakedSsboResult): string {
    const s = baked.scalars;
    return `${this.weightBufGeneration}:${s.tier}:${s.scale}:${s.numBody}:${s.stemCh}:${s.midCh}:${s.upOutCh}:${s.hasSeamLut}`;
  }

  private writeSlot(
    slot: number,
    p: {
      inC4: number;
      inH: number;
      inW: number;
      outC4: number;
      outH: number;
      outW: number;
      wOff?: number;
      bOff?: number;
      pbOff?: number;
      actOff?: number;
      flags?: number;
      scaleR?: number;
      alpha?: number;
      beta?: number;
      delta?: number;
    },
  ): void {
    const base = (slot * UNIFORM_STRIDE) >>> 2;
    this.paramsI32[base + 0] = p.inC4;
    this.paramsI32[base + 1] = p.inH;
    this.paramsI32[base + 2] = p.inW;
    this.paramsI32[base + 3] = p.outC4;
    this.paramsI32[base + 4] = p.outH;
    this.paramsI32[base + 5] = p.outW;
    this.paramsU32[base + 6] = p.wOff ?? 0;
    this.paramsU32[base + 7] = p.bOff ?? 0;
    this.paramsU32[base + 8] = p.pbOff ?? 0;
    this.paramsU32[base + 9] = p.actOff ?? 0;
    this.paramsU32[base + 10] = p.flags ?? 0;
    this.paramsI32[base + 11] = p.scaleR ?? 1;
    this.paramsF32[base + 12] = p.alpha ?? 0.0;
    this.paramsF32[base + 13] = p.beta ?? 0.0;
    this.paramsF32[base + 14] = p.delta ?? 0.0;
    this.paramsF32[base + 15] = 0.0;
  }

  /**
   * Pre-builds the entire SR compute execution plan (uniform slots + cached GPUBindGroups).
   * Eliminates 100% of per-frame bindGroup creation & uniform encoding overhead!
   */
  private buildExecutionPlan(
    weightBuf: GPUBuffer,
    baked: BakedSsboResult,
    fast: PackedFastWeights,
    lrH: number,
    lrW: number,
    outH: number,
    outW: number,
    bufs: {
      xBuf: GPUBuffer;
      seamLrBuf: GPUBuffer;
      baseBuf: GPUBuffer;
      resBuf: GPUBuffer;
      fHeadBuf: GPUBuffer;
      fMidBuf: GPUBuffer;
      s2Buf: GPUBuffer;
      pingBuf: GPUBuffer;
      pongBuf: GPUBuffer;
    },
  ): { steps: PlanStep[]; slotCount: number } {
    const sc = baked.scalars;
    const uh = lrH >>> 1;
    const uw = lrW >>> 1;
    const rTail = Math.abs(sc.scale - 2.0) < 1e-4 ? 4 : 3;
    const steps: PlanStep[] = [];
    let slot = 0;

    const addDispatch = (
      kernel: FastKernelKey,
      b0: GPUBuffer,
      b1: GPUBuffer,
      b2: GPUBuffer,
      b3: GPUBuffer,
      gx: number,
      gy: number,
      gz: number,
      params: Parameters<CodecSREngine['writeSlot']>[1],
    ): void => {
      if (slot >= MAX_DISPATCH_SLOTS) {
        throw new Error(`Exceeded MAX_DISPATCH_SLOTS (${MAX_DISPATCH_SLOTS})`);
      }
      this.writeSlot(slot, params);
      const bindGroup = this.device.createBindGroup({
        layout: this.srBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: b0 } },
          { binding: 1, resource: { buffer: b1 } },
          { binding: 2, resource: { buffer: b2 } },
          { binding: 3, resource: { buffer: b3 } },
          { binding: 4, resource: { buffer: this.paramsBuf, size: 64 } },
        ],
      });
      steps.push({
        kind: 'compute',
        kernel,
        bindGroup,
        slotOffset: slot * UNIFORM_STRIDE,
        gx,
        gy,
        gz,
      });
      slot++;
    };

    const addConv = (
      prefix: string,
      h: number,
      w: number,
      inBuf: GPUBuffer,
      outBuf: GPUBuffer,
      inIsScalar = false,
    ): void => {
      const m = fast.layers[prefix];
      const flags = (m.hasPb ? 1 : 0) | (m.hasAct ? 2 : 0);
      addDispatch(
        inIsScalar ? 'conv3x3_scalar' : 'conv3x3_c4',
        weightBuf,
        inBuf,
        bufs.xBuf,
        outBuf,
        Math.ceil(w / 16),
        Math.ceil(h / 16),
        Math.ceil(m.outC4 / 2),
        {
          inC4: m.inC4,
          inH: h,
          inW: w,
          outC4: m.outC4,
          outH: h,
          outW: w,
          wOff: m.wOffU32,
          bOff: m.bOffU32,
          pbOff: m.pbOffU32,
          actOff: m.actOffU32,
          flags,
        },
      );
    };

    // 0. Bicubic base (Keys a = -0.75)
    addDispatch(
      'bicubic',
      weightBuf,
      bufs.xBuf,
      bufs.xBuf,
      bufs.baseBuf,
      Math.ceil(outW / 16),
      Math.ceil(outH / 16),
      1,
      { inC4: 1, inH: lrH, inW: lrW, outC4: 1, outH, outW },
    );

    const midIdx = Math.floor(sc.numBody / 2);

    if (sc.isHourglass) {
      const stemC4 = Math.ceil(sc.stemCh / 4);
      const midC4 = Math.ceil(sc.midCh / 4);

      addConv('stem1', lrH, lrW, bufs.xBuf, bufs.pongBuf, true);
      addConv('stem2', lrH, lrW, bufs.pongBuf, bufs.s2Buf, false);

      addDispatch(
        'unshuffle2_c4',
        weightBuf,
        bufs.s2Buf,
        bufs.xBuf,
        bufs.pingBuf,
        Math.ceil(uw / 16),
        Math.ceil(uh / 16),
        stemC4,
        { inC4: stemC4, inH: lrH, inW: lrW, outC4: stemC4 * 4, outH: uh, outW: uw },
      );

      addConv('down', uh, uw, bufs.pingBuf, bufs.fHeadBuf, false);

      let curIn = bufs.fHeadBuf;
      let curOut = bufs.pingBuf;
      for (let idx = 0; idx < sc.numBody; idx++) {
        addConv(`body.${idx}`, uh, uw, curIn, curOut, false);
        if (idx === midIdx) {
          steps.push({ kind: 'copy', src: curOut, dst: bufs.fMidBuf, bytes: midC4 * 4 * uh * uw * 4 });
        }
        curIn = curOut;
        curOut = curOut === bufs.pingBuf ? bufs.pongBuf : bufs.pingBuf;
      }

      addDispatch(
        'span_combine_c4',
        weightBuf,
        bufs.fHeadBuf,
        bufs.fMidBuf,
        curIn,
        Math.ceil(uw / 16),
        Math.ceil(uh / 16),
        midC4,
        {
          inC4: midC4,
          inH: uh,
          inW: uw,
          outC4: midC4,
          outH: uh,
          outW: uw,
          alpha: sc.skipAlpha,
          beta: sc.skipBeta,
        },
      );

      if (sc.hgHasUp) {
        addConv('up', uh, uw, curIn, curOut, false);
        addDispatch(
          'shuffle2_c4',
          weightBuf,
          curOut,
          bufs.s2Buf,
          bufs.fHeadBuf,
          Math.ceil(lrW / 16),
          Math.ceil(lrH / 16),
          stemC4,
          {
            inC4: stemC4 * 4,
            inH: uh,
            inW: uw,
            outC4: stemC4,
            outH: lrH,
            outW: lrW,
            scaleR: 2,
            flags: 8, // add_b
          },
        );
        addConv('refine', lrH, lrW, bufs.fHeadBuf, bufs.pingBuf, false);
        addConv('tail', lrH, lrW, bufs.pingBuf, bufs.pongBuf, false);
        addDispatch(
          'shuffle_to_scalar',
          weightBuf,
          bufs.pongBuf,
          bufs.xBuf,
          bufs.resBuf,
          Math.ceil(outW / 16),
          Math.ceil(outH / 16),
          1,
          { inC4: 1, inH: lrH, inW: lrW, outC4: 1, outH, outW, scaleR: 2 },
        );
      } else {
        if (sc.hgHasHrSkip) {
          addDispatch(
            'unshuffle2_c4',
            weightBuf,
            bufs.s2Buf,
            bufs.xBuf,
            curOut,
            Math.ceil(uw / 16),
            Math.ceil(uh / 16),
            stemC4,
            { inC4: stemC4, inH: lrH, inW: lrW, outC4: stemC4 * 4, outH: uh, outW: uw },
          );
          addDispatch(
            'span_combine_c4',
            weightBuf,
            curOut,
            curOut,
            curIn,
            Math.ceil(uw / 16),
            Math.ceil(uh / 16),
            midC4,
            {
              inC4: midC4,
              inH: uh,
              inW: uw,
              outC4: midC4,
              outH: uh,
              outW: uw,
              alpha: sc.skipHr,
              beta: 0.0,
            },
          );
        }
        addConv('tail', uh, uw, curIn, curOut, false);
        addDispatch(
          'shuffle_to_scalar',
          weightBuf,
          curOut,
          bufs.xBuf,
          bufs.resBuf,
          Math.ceil(outW / 16),
          Math.ceil(outH / 16),
          1,
          { inC4: 3, inH: uh, inW: uw, outC4: 1, outH, outW, scaleR: 3 },
        );
      }
    } else {
      const midC4 = Math.ceil(sc.midCh / 4);
      addDispatch(
        'unshuffle2_scalar',
        weightBuf,
        bufs.xBuf,
        bufs.xBuf,
        bufs.pingBuf,
        Math.ceil(uw / 16),
        Math.ceil(uh / 16),
        1,
        { inC4: 1, inH: lrH, inW: lrW, outC4: 1, outH: uh, outW: uw },
      );

      addConv('head', uh, uw, bufs.pingBuf, bufs.fHeadBuf, false);

      let curIn = bufs.fHeadBuf;
      let curOut = bufs.pingBuf;
      for (let idx = 0; idx < sc.numBody; idx++) {
        addConv(`body.${idx}`, uh, uw, curIn, curOut, false);
        if (idx === midIdx) {
          steps.push({ kind: 'copy', src: curOut, dst: bufs.fMidBuf, bytes: midC4 * 4 * uh * uw * 4 });
        }
        curIn = curOut;
        curOut = curOut === bufs.pingBuf ? bufs.pongBuf : bufs.pingBuf;
      }

      addDispatch(
        'span_combine_c4',
        weightBuf,
        bufs.fHeadBuf,
        bufs.fMidBuf,
        curIn,
        Math.ceil(uw / 16),
        Math.ceil(uh / 16),
        midC4,
        {
          inC4: midC4,
          inH: uh,
          inW: uw,
          outC4: midC4,
          outH: uh,
          outW: uw,
          alpha: sc.skipAlpha,
          beta: sc.skipBeta,
        },
      );

      addConv('tail', uh, uw, curIn, curOut, false);
      addDispatch(
        'shuffle_to_scalar',
        weightBuf,
        curOut,
        bufs.xBuf,
        bufs.resBuf,
        Math.ceil(outW / 16),
        Math.ceil(outH / 16),
        1,
        {
          inC4: Math.ceil((rTail * rTail) / 4),
          inH: uh,
          inW: uw,
          outC4: 1,
          outH,
          outW,
          scaleR: rTail,
        },
      );
    }

    // Optional Seam3DLUT computed ONCE at LR resolution
    const hasSeam = sc.hasSeamLut && baked.halfOffsets['seam_lut.table'] !== undefined;
    if (hasSeam) {
      addDispatch(
        'seam_lut_lr',
        weightBuf,
        bufs.xBuf,
        bufs.xBuf,
        bufs.seamLrBuf,
        Math.ceil(lrW / 16),
        Math.ceil(lrH / 16),
        1,
        {
          inC4: 1,
          inH: lrH,
          inW: lrW,
          outC4: 1,
          outH: lrH,
          outW: lrW,
          wOff: fast.seamTableOffU32,
          bOff: fast.seamMaskOffU32,
        },
      );
    }

    // Final fused pass: Residual + Bilinear(SeamLUT_LR) + Bicubic Base + 3x3 TrustClamp
    addDispatch(
      'add_seam_trust_clamp',
      bufs.seamLrBuf,
      bufs.baseBuf,
      bufs.xBuf,
      bufs.resBuf,
      Math.ceil(outW / 16),
      Math.ceil(outH / 16),
      1,
      {
        inC4: 1,
        inH: lrH,
        inW: lrW,
        outC4: 1,
        outH,
        outW,
        flags: hasSeam ? 16 : 0,
        delta: sc.trustDelta,
      },
    );

    return { steps, slotCount: slot };
  }

  private encodePlanSteps(encoder: GPUCommandEncoder, steps: PlanStep[]): void {
    for (const step of steps) {
      if (step.kind === 'copy') {
        encoder.copyBufferToBuffer(step.src, 0, step.dst, 0, step.bytes);
      } else {
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.kernels[step.kernel]);
        pass.setBindGroup(0, step.bindGroup, [step.slotOffset]);
        pass.dispatchWorkgroups(step.gx, step.gy, step.gz);
        pass.end();
      }
    }
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
    const fast = packFastWeights(baked);
    const outH = Math.round(lrH * baked.scalars.scale);
    const outW = Math.round(lrW * baked.scalars.scale);
    const scratchBytes = computeRequiredScratchBytes(lrH, lrW, outH, outW, baked);

    const wBuf = this.device.createBuffer({
      size: Math.max(16, fast.weightU32.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      wBuf,
      0,
      fast.weightU32.buffer,
      fast.weightU32.byteOffset,
      fast.weightU32.byteLength,
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

    const makeBuf = (size: number) =>
      this.device.createBuffer({
        size: Math.max(16, size),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

    const seamLrBuf = makeBuf(lrH * lrW * 4);
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
      const plan = this.buildExecutionPlan(wBuf, baked, fast, lrH, lrW, outH, outW, {
        xBuf,
        seamLrBuf,
        baseBuf,
        resBuf,
        fHeadBuf,
        fMidBuf,
        s2Buf,
        pingBuf,
        pongBuf,
      });

      this.device.queue.writeBuffer(
        this.paramsBuf,
        0,
        this.paramsStaging,
        0,
        plan.slotCount * UNIFORM_STRIDE,
      );

      const encoder = this.device.createCommandEncoder();
      this.encodePlanSteps(encoder, plan.steps);
      encoder.copyBufferToBuffer(resBuf, 0, readbackBuf, 0, outH * outW * 4);
      this.device.queue.submit([encoder.finish()]);

      await readbackBuf.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(readbackBuf.getMappedRange().slice(0));
      readbackBuf.unmap();
      return copy;
    } finally {
      // Restore active workspace uniform buffer if one exists
      if (this.workspace && this.activeBaked && this.activeFastWeights && this.weightBuf) {
        const restored = this.buildExecutionPlan(
          this.weightBuf,
          this.activeBaked,
          this.activeFastWeights,
          this.workspace.padH,
          this.workspace.padW,
          this.workspace.srPadH,
          this.workspace.srPadW,
          this.workspace,
        );
        this.workspace.planSteps = restored.steps;
        this.workspace.planSlotCount = restored.slotCount;
        this.device.queue.writeBuffer(
          this.paramsBuf,
          0,
          this.paramsStaging,
          0,
          restored.slotCount * UNIFORM_STRIDE,
        );
      }

      wBuf.destroy();
      xBuf.destroy();
      seamLrBuf.destroy();
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

  private ensureWorkspace(
    srcW: number,
    srcH: number,
    workW: number,
    workH: number,
    baked: BakedSsboResult,
    fast: PackedFastWeights,
  ): FrameWorkspace {
    const padW = workW + (workW & 1);
    const padH = workH + (workH & 1);
    const srPadW = Math.round(padW * baked.scalars.scale);
    const srPadH = Math.round(padH * baked.scalars.scale);
    const outW = Math.round(workW * baked.scalars.scale);
    const outH = Math.round(workH * baked.scalars.scale);
    const scratchBytes = computeRequiredScratchBytes(padH, padW, srPadH, srPadW, baked);
    const sig = this.getModelSignature(baked);

    if (
      this.workspace &&
      this.workspace.srcW === srcW &&
      this.workspace.srcH === srcH &&
      this.workspace.workW === workW &&
      this.workspace.workH === workH &&
      this.workspace.outW === outW &&
      this.workspace.outH === outH &&
      this.workspace.scratchBytes >= scratchBytes
    ) {
      if (this.workspace.modelSignature !== sig && this.weightBuf) {
        const plan = this.buildExecutionPlan(
          this.weightBuf,
          baked,
          fast,
          padH,
          padW,
          srPadH,
          srPadW,
          this.workspace,
        );
        this.workspace.planSteps = plan.steps;
        this.workspace.planSlotCount = plan.slotCount;
        this.workspace.modelSignature = sig;
        this.device.queue.writeBuffer(
          this.paramsBuf,
          0,
          this.paramsStaging,
          0,
          plan.slotCount * UNIFORM_STRIDE,
        );
      }
      return this.workspace;
    }

    if (scratchBytes > this.device.limits.maxStorageBufferBindingSize) {
      throw new Error(
        `当前工作分辨率 (${workW}×${workH}) 在此档位下所需显存缓冲 (${Math.round(scratchBytes / 1048576)} MB) 超出设备单 Buffer 上限 (${Math.round(this.device.limits.maxStorageBufferBindingSize / 1048576)} MB)，请切换「流畅 720p/540p」模式或使用 Client-L/S 档位。`,
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
        size: Math.max(16, size),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

    const xBuf = mkBuf('x_luma', padW * padH * 4);
    const seamLrBuf = mkBuf('seam_lr', padW * padH * 4);
    const baseBuf = mkBuf('base_bicubic_luma', srPadW * srPadH * 4);
    const resBuf = mkBuf('res_sr_luma', srPadW * srPadH * 4);
    const fHeadBuf = mkBuf('f_head', scratchBytes);
    const fMidBuf = mkBuf('f_mid', scratchBytes);
    const s2Buf = mkBuf('s2', scratchBytes);
    const pingBuf = mkBuf('ping', scratchBytes);
    const pongBuf = mkBuf('pong', scratchBytes);

    const rgb2yUniform = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      rgb2yUniform,
      0,
      new Uint32Array([srcW, srcH, workW, workH, padW, padH, 0, 0]),
    );

    const y2rgbUniform = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      y2rgbUniform,
      0,
      new Uint32Array([outW, outH, srPadW, srcW, srcH, 0, 0, 0]),
    );

    const rgb2yBindGroup = this.device.createBindGroup({
      layout: this.rgb2yPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: { buffer: xBuf } },
        { binding: 3, resource: { buffer: rgb2yUniform } },
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

    const displayLinearBindGroup = this.device.createBindGroup({
      layout: this.displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srTex.createView() },
        { binding: 1, resource: bicubicTex.createView() },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: { buffer: this.displayUniformBuf } },
      ],
    });

    const displayNearestBindGroup = this.device.createBindGroup({
      layout: this.displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srTex.createView() },
        { binding: 1, resource: bicubicTex.createView() },
        { binding: 2, resource: this.nearestSampler },
        { binding: 3, resource: { buffer: this.displayUniformBuf } },
      ],
    });

    const plan = this.buildExecutionPlan(this.weightBuf!, baked, fast, padH, padW, srPadH, srPadW, {
      xBuf,
      seamLrBuf,
      baseBuf,
      resBuf,
      fHeadBuf,
      fMidBuf,
      s2Buf,
      pingBuf,
      pongBuf,
    });
    this.device.queue.writeBuffer(
      this.paramsBuf,
      0,
      this.paramsStaging,
      0,
      plan.slotCount * UNIFORM_STRIDE,
    );

    this.workspace = {
      srcW,
      srcH,
      workW,
      workH,
      padW,
      padH,
      srPadW,
      srPadH,
      outW,
      outH,
      scratchBytes,
      modelSignature: sig,
      srcTex,
      srTex,
      bicubicTex,
      xBuf,
      seamLrBuf,
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
      displayLinearBindGroup,
      displayNearestBindGroup,
      planSteps: plan.steps,
      planSlotCount: plan.slotCount,
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
   * Uploads one video frame, runs the tiled C4 WebGPU SR pipeline with ZERO per-frame
   * bindGroup allocations, and optionally renders directly to `renderTargetCtx` in the SAME submit!
   */
  processFrame(
    source: GPUCopyExternalImageSource,
    srcW: number,
    srcH: number,
    includeBicubic = true,
    renderTarget?: { ctx: GPUCanvasContext; opts: DisplayOptions },
    targetWorkSize?: { workW: number; workH: number },
  ): { outW: number; outH: number } {
    if (!this.activeBaked || !this.activeFastWeights || !this.weightBuf) {
      throw new Error('模型尚未初始化');
    }
    const workW = targetWorkSize?.workW ?? srcW;
    const workH = targetWorkSize?.workH ?? srcH;
    const ws = this.ensureWorkspace(srcW, srcH, workW, workH, this.activeBaked, this.activeFastWeights);

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

    // 2. Cached high-performance CodecSR V3.5 execution plan
    this.encodePlanSteps(encoder, ws.planSteps);

    // 3. Y_sr + upsampled CbCr -> RGBA
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.y2rgbPipeline);
      pass.setBindGroup(0, ws.y2rgbSrBindGroup);
      pass.dispatchWorkgroups(Math.ceil(ws.outW / 16), Math.ceil(ws.outH / 16), 1);
      pass.end();
    }

    // 4. Optional Y_bicubic + upsampled CbCr -> RGBA
    if (includeBicubic) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.y2rgbPipeline);
      pass.setBindGroup(0, ws.y2rgbBiBindGroup);
      pass.dispatchWorkgroups(Math.ceil(ws.outW / 16), Math.ceil(ws.outH / 16), 1);
      pass.end();
    }

    // 5. Optional fused display render pass in the SAME command buffer (zero extra submit!)
    if (renderTarget) {
      this.writeDisplayUniforms(renderTarget.opts);
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: renderTarget.ctx.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.035, g: 0.043, b: 0.063, a: 1.0 },
          },
        ],
      });
      pass.setPipeline(this.displayPipeline);
      pass.setBindGroup(
        0,
        renderTarget.opts.zoom >= 2.0 ? ws.displayNearestBindGroup : ws.displayLinearBindGroup,
      );
      pass.draw(3, 1, 0, 0);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
    return { outW: ws.outW, outH: ws.outH };
  }

  private writeDisplayUniforms(opts: DisplayOptions): void {
    const f32 = new Float32Array(this.displayUniformStaging);
    const u32 = new Uint32Array(this.displayUniformStaging);
    f32[0] = opts.splitX;
    u32[1] = opts.viewMode;
    f32[2] = opts.zoom;
    f32[3] = 0;
    f32[4] = opts.centerX;
    f32[5] = opts.centerY;
    f32[6] = 0;
    f32[7] = 0;
    this.device.queue.writeBuffer(this.displayUniformBuf, 0, this.displayUniformStaging);
  }

  /**
   * Renders the most recently processed SR & Bicubic textures onto a WebGPU canvas context.
   * Zero re-inference and zero bindGroup allocation when dragging the split slider or zooming!
   */
  renderToCanvas(ctx: GPUCanvasContext, opts: DisplayOptions): void {
    const ws = this.workspace;
    if (!ws) return;

    this.writeDisplayUniforms(opts);
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
    pass.setBindGroup(0, opts.zoom >= 2.0 ? ws.displayNearestBindGroup : ws.displayLinearBindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

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
    ws.seamLrBuf.destroy();
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
