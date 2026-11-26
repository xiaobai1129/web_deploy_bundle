import * as MP4Box from 'mp4box';
import { ArrayBufferTarget, Muxer } from 'mp4-muxer';
import type { CodecSREngine } from '../engine/webgpuRunner';
import type { VideoSourceMeta } from './videoProbe';

export interface ExportProgress {
  phase: 'demux' | 'processing' | 'flushing' | 'done';
  doneFrames: number;
  totalFrames: number;
  fps: number;
  etaSec: number;
  message: string;
}

export interface ExportResult {
  blob: Blob;
  url: string;
  fileName: string;
  width: number;
  height: number;
  frameCount: number;
  elapsedSec: number;
  avgFps: number;
  hasAudio: boolean;
  pipelineUsed: 'webcodecs-mp4box' | 'video-element-fallback';
}

export interface ExportOptions {
  engine: CodecSREngine;
  source: VideoSourceMeta;
  bitrateMbps: number;
  tierLabel: string;
  scale: number;
  signal: AbortSignal;
  onProgress: (p: ExportProgress) => void;
}

const AVC_CANDIDATE_CODECS = [
  'avc1.640034', // High 5.2 (supports 1080p@120fps & 4K@60fps)
  'avc1.640033', // High 5.1
  'avc1.640032', // High 5.0
  'avc1.64002a', // High 4.2 (1080p@60fps)
  'avc1.640028', // High 4.0 (1080p@30fps)
  'avc1.4d0033', // Main 5.1
  'avc1.4d002a', // Main 4.2
  'avc1.420028', // Baseline 4.0
  'avc1.42001f', // Baseline 3.1
];

async function pickH264EncoderConfig(
  width: number,
  height: number,
  fps: number,
  bitrateBps: number,
): Promise<VideoEncoderConfig> {
  if (typeof VideoEncoder === 'undefined') {
    throw new Error('当前浏览器不支持 WebCodecs VideoEncoder，请使用最新 Chrome / Edge 浏览器导出 MP4。');
  }
  const safeFps = Math.max(1, Math.min(120, Math.round(fps)));
  for (const hw of ['prefer-hardware', 'no-preference'] as const) {
    for (const codec of AVC_CANDIDATE_CODECS) {
      const cfg: VideoEncoderConfig = {
        codec,
        width,
        height,
        bitrate: bitrateBps,
        framerate: safeFps,
        hardwareAcceleration: hw,
        avc: { format: 'avc' },
      };
      try {
        const res = await VideoEncoder.isConfigSupported(cfg);
        if (res.supported) {
          return res.config ?? cfg;
        }
      } catch {
        // try next candidate
      }
    }
  }
  throw new Error(`浏览器硬件/软件 H.264 编码器不支持当前输出规格 (${width}×${height} @ ${safeFps}fps)。`);
}

function extractCodecDescription(mp4: MP4Box.MP4File, trackId: number): Uint8Array | undefined {
  const trak = mp4.getTrackById(trackId);
  const entry = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0];
  if (!entry) return undefined;
  const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
  if (!box) return undefined;
  const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
  box.write(stream);
  return new Uint8Array(stream.buffer, 8);
}

interface DemuxedMp4 {
  videoTrack: MP4Box.MP4Track;
  videoDescription?: Uint8Array;
  videoSamples: MP4Box.MP4Sample[];
  audioTrack?: MP4Box.MP4Track;
  audioSamples: MP4Box.MP4Sample[];
}

function demuxMp4Buffer(arrayBuffer: ArrayBuffer): Promise<DemuxedMp4> {
  return new Promise((resolve, reject) => {
    const mp4 = MP4Box.createFile();
    let videoTrack: MP4Box.MP4Track | null = null;
    let audioTrack: MP4Box.MP4Track | undefined;
    const videoSamples: MP4Box.MP4Sample[] = [];
    const audioSamples: MP4Box.MP4Sample[] = [];
    let settled = false;

    const fail = (msg: string) => {
      if (settled) return;
      settled = true;
      reject(new Error(msg));
    };

    mp4.onError = (e) => fail(`MP4 解封装失败: ${e}`);

    mp4.onReady = (info) => {
      if (!info.videoTracks?.length) {
        fail('MP4 文件中未找到视频轨');
        return;
      }
      videoTrack = info.videoTracks[0];
      mp4.setExtractionOptions(videoTrack.id, 'video', { nbSamples: Math.max(1000, videoTrack.nb_samples) });

      const aTrack = info.audioTracks?.find((t) => t.codec?.toLowerCase().startsWith('mp4a'));
      if (aTrack && aTrack.audio) {
        audioTrack = aTrack;
        mp4.setExtractionOptions(aTrack.id, 'audio', { nbSamples: Math.max(2000, aTrack.nb_samples) });
      }
      mp4.start();
    };

    mp4.onSamples = (_trackId, user, samples) => {
      if (user === 'video') {
        for (const s of samples) videoSamples.push(s);
      } else if (user === 'audio') {
        for (const s of samples) audioSamples.push(s);
      }
    };

    const mp4Buf = arrayBuffer.slice(0) as MP4Box.MP4ArrayBuffer;
    mp4Buf.fileStart = 0;
    mp4.appendBuffer(mp4Buf);
    mp4.flush();

    // mp4box processes synchronously for complete buffers; finalize on next microtask
    queueMicrotask(() => {
      if (settled) return;
      if (!videoTrack || videoSamples.length === 0) {
        fail('MP4 中未提取到有效视频帧');
        return;
      }
      settled = true;
      resolve({
        videoTrack,
        videoDescription: extractCodecDescription(mp4, videoTrack.id),
        videoSamples,
        audioTrack,
        audioSamples,
      });
    });
  });
}

function makeOutputFileName(sourceName: string, tierLabel: string, scale: number): string {
  const base = sourceName.replace(/\.[^.]+$/, '');
  const scaleTag = Math.abs(scale - 1.5) < 1e-3 ? 'x1_5' : 'x2';
  return `${base}_CodecSR_${tierLabel}_${scaleTag}.mp4`;
}

function createExportCanvas(device: GPUDevice, format: GPUTextureFormat, width: number, height: number): {
  canvas: HTMLCanvasElement;
  ctx: GPUCanvasContext;
} {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('webgpu');
  if (!ctx) {
    throw new Error('无法为导出画布创建 WebGPU 上下文');
  }
  ctx.configure({
    device,
    format,
    alphaMode: 'opaque',
  });
  return { canvas, ctx };
}

async function exportViaMp4BoxWebCodecs(
  opts: ExportOptions,
  rawBuffer: ArrayBuffer,
  encW: number,
  encH: number,
): Promise<ExportResult> {
  const { engine, source, bitrateMbps, tierLabel, scale, signal, onProgress } = opts;
  const startWall = performance.now();

  onProgress({
    phase: 'demux',
    doneFrames: 0,
    totalFrames: source.frameCount,
    fps: 0,
    etaSec: 0,
    message: '正在解封装 MP4 码流...',
  });

  const demuxed = await demuxMp4Buffer(rawBuffer);
  if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');

  const decoderCfg: VideoDecoderConfig = {
    codec: demuxed.videoTrack.codec,
    codedWidth: demuxed.videoTrack.video?.width ?? source.width,
    codedHeight: demuxed.videoTrack.video?.height ?? source.height,
    description: demuxed.videoDescription,
  };
  const decSupport = await VideoDecoder.isConfigSupported(decoderCfg);
  if (!decSupport.supported) {
    throw new Error(`VideoDecoder 不支持 ${decoderCfg.codec}`);
  }

  const fps =
    demuxed.videoTrack.duration && demuxed.videoTrack.timescale
      ? demuxed.videoSamples.length / (demuxed.videoTrack.duration / demuxed.videoTrack.timescale)
      : source.fps;
  const encoderCfg = await pickH264EncoderConfig(encW, encH, fps, Math.round(bitrateMbps * 1_000_000));

  const target = new ArrayBufferTarget();
  const hasAudio = Boolean(demuxed.audioTrack?.audio && demuxed.audioSamples.length > 0);
  const muxer = new Muxer({
    target,
    video: {
      codec: 'avc',
      width: encW,
      height: encH,
    },
    audio:
      hasAudio && demuxed.audioTrack?.audio
        ? {
            codec: 'aac',
            numberOfChannels: demuxed.audioTrack.audio.channel_count,
            sampleRate: demuxed.audioTrack.audio.sample_rate,
          }
        : undefined,
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  const { canvas: exportCanvas, ctx: exportCtx } = createExportCanvas(
    engine.device,
    engine.canvasFormat,
    encW,
    encH,
  );

  let encoderError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      muxer.addVideoChunk(chunk, meta);
    },
    error: (e) => {
      encoderError = e instanceof Error ? e : new Error(String(e));
    },
  });
  encoder.configure(encoderCfg);

  const decodedQueue: VideoFrame[] = [];
  let notifyFrameAvailable: (() => void) | null = null;
  let decoderDone = false;
  let decoderError: Error | null = null;

  const decoder = new VideoDecoder({
    output: (frame) => {
      decodedQueue.push(frame);
      if (notifyFrameAvailable) {
        const fn = notifyFrameAvailable;
        notifyFrameAvailable = null;
        fn();
      }
    },
    error: (e) => {
      decoderError = e instanceof Error ? e : new Error(String(e));
      if (notifyFrameAvailable) {
        const fn = notifyFrameAvailable;
        notifyFrameAvailable = null;
        fn();
      }
    },
  });
  decoder.configure(decoderCfg);

  const totalFrames = demuxed.videoSamples.length;
  const keyframeInterval = Math.max(15, Math.min(120, Math.round(fps * 2)));
  const defaultDurUs = Math.round(1_000_000 / Math.max(1, fps));
  let processedFrames = 0;

  // Consumer: runs GPU super-resolution & H.264 encode sequentially as frames arrive
  const consumerPromise = (async () => {
    while (true) {
      if (signal.aborted) break;
      if (decoderError) throw decoderError;
      if (encoderError) throw encoderError;

      const frame = decodedQueue.shift();
      if (!frame) {
        if (decoderDone) break;
        await new Promise<void>((r) => {
          notifyFrameAvailable = r;
        });
        continue;
      }

      try {
        const w = frame.displayWidth || frame.codedWidth;
        const h = frame.displayHeight || frame.codedHeight;
        engine.processFrame(frame, w, h, false, {
          ctx: exportCtx,
          opts: {
            splitX: 0,
            viewMode: 1,
            zoom: 1,
            centerX: 0.5,
            centerY: 0.5,
          },
        });
        await engine.waitGpuIdle();

        while (encoder.encodeQueueSize > 8 && !signal.aborted && !encoderError) {
          await new Promise((r) => setTimeout(r, 4));
        }
        if (signal.aborted) break;
        if (encoderError) throw encoderError;

        const outFrame = new VideoFrame(exportCanvas, {
          timestamp: frame.timestamp,
          duration: frame.duration ?? defaultDurUs,
        });
        encoder.encode(outFrame, { keyFrame: processedFrames % keyframeInterval === 0 });
        outFrame.close();
        processedFrames++;

        const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
        const curFps = processedFrames / elapsedSec;
        const etaSec = curFps > 0 ? Math.max(0, (totalFrames - processedFrames) / curFps) : 0;
        onProgress({
          phase: 'processing',
          doneFrames: processedFrames,
          totalFrames,
          fps: curFps,
          etaSec,
          message: `本地 WebGPU 超分渲染中 (${processedFrames}/${totalFrames} 帧)`,
        });
      } finally {
        frame.close();
      }
    }
  })();

  // Producer: feeds MP4 demuxed chunks with gentle backpressure
  try {
    for (let i = 0; i < demuxed.videoSamples.length; i++) {
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (decoderError) throw decoderError;
      if (encoderError) throw encoderError;

      while ((decoder.decodeQueueSize > 8 || decodedQueue.length > 4) && !signal.aborted && !decoderError) {
        await new Promise((r) => setTimeout(r, 4));
      }
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');

      const s = demuxed.videoSamples[i];
      const chunk = new EncodedVideoChunk({
        type: s.is_sync ? 'key' : 'delta',
        timestamp: Math.round((s.cts * 1_000_000) / s.timescale),
        duration: Math.round((s.duration * 1_000_000) / s.timescale),
        data: s.data,
      });
      decoder.decode(chunk);
    }

    await decoder.flush();
    decoderDone = true;
    if (notifyFrameAvailable) {
      const fn = notifyFrameAvailable as () => void;
      notifyFrameAvailable = null;
      fn();
    }

    await consumerPromise;
    if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');

    onProgress({
      phase: 'flushing',
      doneFrames: processedFrames,
      totalFrames,
      fps: processedFrames / Math.max(0.01, (performance.now() - startWall) / 1000),
      etaSec: 0,
      message: '正在封装最终 MP4 文件...',
    });

    await encoder.flush();

    if (hasAudio && demuxed.audioSamples.length > 0) {
      for (const a of demuxed.audioSamples) {
        const tsUs = Math.round((a.cts * 1_000_000) / a.timescale);
        const durUs = Math.round((a.duration * 1_000_000) / a.timescale);
        muxer.addAudioChunkRaw(a.data, 'key', tsUs, durUs);
      }
    }

    muxer.finalize();
  } finally {
    for (const leftover of decodedQueue) leftover.close();
    decodedQueue.length = 0;
    try {
      if (decoder.state !== 'closed') decoder.close();
    } catch {
      // ignore
    }
    try {
      if (encoder.state !== 'closed') encoder.close();
    } catch {
      // ignore
    }
  }

  const blob = new Blob([target.buffer], { type: 'video/mp4' });
  const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
  const res: ExportResult = {
    blob,
    url: URL.createObjectURL(blob),
    fileName: makeOutputFileName(source.name, tierLabel, scale),
    width: encW,
    height: encH,
    frameCount: processedFrames,
    elapsedSec,
    avgFps: processedFrames / elapsedSec,
    hasAudio,
    pipelineUsed: 'webcodecs-mp4box',
  };

  onProgress({
    phase: 'done',
    doneFrames: processedFrames,
    totalFrames: processedFrames,
    fps: res.avgFps,
    etaSec: 0,
    message: '超分 MP4 导出完成',
  });
  return res;
}

async function exportViaVideoElementFallback(
  opts: ExportOptions,
  encW: number,
  encH: number,
): Promise<ExportResult> {
  const { engine, source, bitrateMbps, tierLabel, scale, signal, onProgress } = opts;
  const startWall = performance.now();

  const fps = Math.max(1, Math.min(60, Math.round(source.fps || 30)));
  const totalFrames = Math.max(1, Math.floor(source.durationSec * fps));
  const encoderCfg = await pickH264EncoderConfig(encW, encH, fps, Math.round(bitrateMbps * 1_000_000));

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: 'avc', width: encW, height: encH },
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  const { canvas: exportCanvas, ctx: exportCtx } = createExportCanvas(
    engine.device,
    engine.canvasFormat,
    encW,
    encH,
  );

  let encoderError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => {
      encoderError = e instanceof Error ? e : new Error(String(e));
    },
  });
  encoder.configure(encoderCfg);

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = source.url;

  await new Promise<void>((resolve, reject) => {
    video.onloadeddata = () => resolve();
    video.onerror = () => reject(new Error('无法加载视频进行逐帧导出'));
  });

  const stepSec = 1 / fps;
  const durUs = Math.round(1_000_000 / fps);
  const keyframeInterval = Math.max(15, fps * 2);
  let processed = 0;

  try {
    for (let i = 0; i < totalFrames; i++) {
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (encoderError) throw encoderError;

      const tSec = Math.min(source.durationSec - 0.001, i * stepSec);
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = Math.max(0, tSec);
      });

      engine.processFrame(video, video.videoWidth, video.videoHeight, false, {
        ctx: exportCtx,
        opts: {
          splitX: 0,
          viewMode: 1,
          zoom: 1,
          centerX: 0.5,
          centerY: 0.5,
        },
      });
      await engine.waitGpuIdle();

      while (encoder.encodeQueueSize > 8 && !signal.aborted && !encoderError) {
        await new Promise((r) => setTimeout(r, 4));
      }
      if (encoderError) throw encoderError;

      const outFrame = new VideoFrame(exportCanvas, {
        timestamp: i * durUs,
        duration: durUs,
      });
      encoder.encode(outFrame, { keyFrame: i % keyframeInterval === 0 });
      outFrame.close();
      processed++;

      const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
      const curFps = processed / elapsedSec;
      const etaSec = curFps > 0 ? Math.max(0, (totalFrames - processed) / curFps) : 0;
      onProgress({
        phase: 'processing',
        doneFrames: processed,
        totalFrames,
        fps: curFps,
        etaSec,
        message: `兼容模式逐帧超分中 (${processed}/${totalFrames} 帧)`,
      });
    }

    onProgress({
      phase: 'flushing',
      doneFrames: processed,
      totalFrames,
      fps: processed / Math.max(0.01, (performance.now() - startWall) / 1000),
      etaSec: 0,
      message: '正在封装最终 MP4 文件...',
    });

    await encoder.flush();
    muxer.finalize();
  } finally {
    video.src = '';
    try {
      if (encoder.state !== 'closed') encoder.close();
    } catch {
      // ignore
    }
  }

  const blob = new Blob([target.buffer], { type: 'video/mp4' });
  const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
  const res: ExportResult = {
    blob,
    url: URL.createObjectURL(blob),
    fileName: makeOutputFileName(source.name, tierLabel, scale),
    width: encW,
    height: encH,
    frameCount: processed,
    elapsedSec,
    avgFps: processed / elapsedSec,
    hasAudio: false,
    pipelineUsed: 'video-element-fallback',
  };
  onProgress({
    phase: 'done',
    doneFrames: processed,
    totalFrames: processed,
    fps: res.avgFps,
    etaSec: 0,
    message: '超分 MP4 导出完成',
  });
  return res;
}

/**
 * Exports the entire video super-resolved with local WebGPU into a downloadable H.264 MP4.
 * Uses zero-skip MP4Box + WebCodecs VideoDecoder when possible, with automatic fallback for non-MP4 formats.
 */
export async function exportSuperResolvedMp4(opts: ExportOptions): Promise<ExportResult> {
  const rawOutW = Math.round(opts.source.width * opts.scale);
  const rawOutH = Math.round(opts.source.height * opts.scale);
  // H.264 requires even dimensions
  const encW = Math.max(16, rawOutW & ~1);
  const encH = Math.max(16, rawOutH & ~1);

  let rawBuffer: ArrayBuffer | null = null;
  try {
    if (opts.source.file) {
      rawBuffer = await opts.source.file.arrayBuffer();
    } else {
      const resp = await fetch(opts.source.url);
      if (resp.ok) rawBuffer = await resp.arrayBuffer();
    }
  } catch {
    rawBuffer = null;
  }

  if (rawBuffer && typeof VideoDecoder !== 'undefined') {
    try {
      return await exportViaMp4BoxWebCodecs(opts, rawBuffer, encW, encH);
    } catch (err) {
      if (opts.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
        throw err;
      }
      // Fall back to universal <video> stepping for WebM/MOV or tricky MP4 profiles
      console.warn('[CodecSR Export] MP4Box/VideoDecoder path failed, falling back to <video> stepping:', err);
    }
  }

  return await exportViaVideoElementFallback(opts, encW, encH);
}
