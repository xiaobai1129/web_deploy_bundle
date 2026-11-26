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
  clippedToSec: number | null;
  pipelineUsed: 'webcodecs-mp4box' | 'video-element-fallback';
}

export interface ExportOptions {
  engine: CodecSREngine;
  source: VideoSourceMeta;
  workW: number;
  workH: number;
  bitrateMbps: number;
  tierLabel: string;
  scale: number;
  maxDurationSec?: number; // e.g. 10 for "first 10 seconds" clip mode
  signal: AbortSignal;
  onProgress: (p: ExportProgress) => void;
}

const AVC_CANDIDATE_CODECS = [
  'avc1.640034', // High 5.2 (supports 1080p@120fps & 4K@60fps)
  'avc1.640033', // High 5.1
  'avc1.640032', // High 5.0
  'avc1.64002a', // High 4.2 (1080p@60fps)
  'avc1.640028', // High 4.0 (1080p@30fps)
  'avc1.4d0034', // Main 5.2
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
  throw new Error(
    `浏览器 H.264 编码器不支持当前输出分辨率 (${width}×${height} @ ${safeFps}fps)。建议在右侧开启「720p/540p 流畅模式」降低输出尺寸后再导出。`,
  );
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

async function demuxMp4Source(source: VideoSourceMeta): Promise<DemuxedMp4> {
  const mp4 = MP4Box.createFile();
  let videoTrack: MP4Box.MP4Track | null = null;
  let audioTrack: MP4Box.MP4Track | undefined;
  const videoSamples: MP4Box.MP4Sample[] = [];
  const audioSamples: MP4Box.MP4Sample[] = [];
  let parseErr: string | null = null;

  mp4.onError = (e) => {
    parseErr = `MP4 解封装失败: ${e}`;
  };

  mp4.onReady = (info) => {
    if (!info.videoTracks?.length) {
      parseErr = 'MP4 文件中未找到视频轨';
      return;
    }
    videoTrack = info.videoTracks[0];
    mp4.setExtractionOptions(videoTrack.id, 'video', {
      nbSamples: Math.max(1000, videoTrack.nb_samples),
    });

    const aTrack = info.audioTracks?.find((t) => t.codec?.toLowerCase().startsWith('mp4a'));
    if (aTrack && aTrack.audio) {
      audioTrack = aTrack;
      mp4.setExtractionOptions(aTrack.id, 'audio', {
        nbSamples: Math.max(2000, aTrack.nb_samples),
      });
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

  if (source.file) {
    // Stream in 8MB slices to avoid doubling memory on 150MB+ phone recordings
    const CHUNK = 8 * 1024 * 1024;
    for (let off = 0; off < source.file.size; off += CHUNK) {
      const ab = await source.file.slice(off, Math.min(source.file.size, off + CHUNK)).arrayBuffer();
      const mp4Buf = ab as MP4Box.MP4ArrayBuffer;
      mp4Buf.fileStart = off;
      mp4.appendBuffer(mp4Buf);
      if (parseErr) throw new Error(parseErr);
    }
  } else {
    const resp = await fetch(source.url);
    if (!resp.ok) throw new Error(`读取演示视频失败 (${resp.status})`);
    const ab = await resp.arrayBuffer();
    const mp4Buf = ab as MP4Box.MP4ArrayBuffer;
    mp4Buf.fileStart = 0;
    mp4.appendBuffer(mp4Buf);
  }

  mp4.flush();
  await new Promise((r) => setTimeout(r, 0));

  if (parseErr) throw new Error(parseErr);
  const vt = videoTrack as MP4Box.MP4Track | null;
  if (!vt || videoSamples.length === 0) {
    throw new Error('MP4 中未提取到有效视频帧');
  }
  return {
    videoTrack: vt,
    videoDescription: extractCodecDescription(mp4, vt.id),
    videoSamples,
    audioTrack,
    audioSamples,
  };
}

function makeOutputFileName(
  sourceName: string,
  tierLabel: string,
  scale: number,
  clippedSec: number | null,
): string {
  const base = sourceName.replace(/\.[^.]+$/, '');
  const scaleTag = Math.abs(scale - 1.5) < 1e-3 ? 'x1_5' : 'x2';
  const clipTag = clippedSec !== null ? `_first${Math.round(clippedSec)}s` : '';
  return `${base}_CodecSR_${tierLabel}_${scaleTag}${clipTag}.mp4`;
}

function createExportCanvas(
  device: GPUDevice,
  format: GPUTextureFormat,
  width: number,
  height: number,
): {
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
  encW: number,
  encH: number,
): Promise<ExportResult> {
  const { engine, source, workW, workH, bitrateMbps, tierLabel, scale, maxDurationSec, signal, onProgress } = opts;
  const startWall = performance.now();
  const clippedSec =
    maxDurationSec && maxDurationSec > 0 && source.durationSec > maxDurationSec + 0.25
      ? maxDurationSec
      : null;

  onProgress({
    phase: 'demux',
    doneFrames: 0,
    totalFrames: source.frameCount,
    fps: 0,
    etaSec: 0,
    message: '正在解封装 MP4 / HEVC 码流...',
  });

  const demuxed = await demuxMp4Source(source);
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

  // Filter samples when exporting a time-bounded clip (e.g. first 10s)
  const firstVideoCtsSec = demuxed.videoSamples[0].cts / demuxed.videoSamples[0].timescale;
  const maxCtsSec = clippedSec !== null ? firstVideoCtsSec + clippedSec : Number.POSITIVE_INFINITY;
  const maxUsCutoff = clippedSec !== null ? Math.round((firstVideoCtsSec + clippedSec) * 1_000_000) : Number.POSITIVE_INFINITY;

  // Keep a small B-frame decode margin (+0.5s) then trim exact presentation timestamps on output
  const feedSamples: MP4Box.MP4Sample[] = [];
  for (const s of demuxed.videoSamples) {
    const dtsSec = s.dts / s.timescale;
    if (clippedSec !== null && dtsSec > maxCtsSec + 0.5 && feedSamples.length > 0) {
      break;
    }
    feedSamples.push(s);
  }

  const audioSamplesToMux: MP4Box.MP4Sample[] = [];
  if (demuxed.audioSamples.length > 0) {
    const firstAudioSec = demuxed.audioSamples[0].cts / demuxed.audioSamples[0].timescale;
    const maxAudioSec = clippedSec !== null ? firstAudioSec + clippedSec : Number.POSITIVE_INFINITY;
    for (const a of demuxed.audioSamples) {
      if (a.cts / a.timescale <= maxAudioSec) {
        audioSamplesToMux.push(a);
      }
    }
  }

  const fps =
    demuxed.videoTrack.duration && demuxed.videoTrack.timescale
      ? demuxed.videoSamples.length / (demuxed.videoTrack.duration / demuxed.videoTrack.timescale)
      : source.fps;
  const estimatedTargetFrames =
    clippedSec !== null
      ? Math.max(1, Math.min(feedSamples.length, Math.round(clippedSec * fps)))
      : feedSamples.length;

  const encoderCfg = await pickH264EncoderConfig(encW, encH, fps, Math.round(bitrateMbps * 1_000_000));

  const target = new ArrayBufferTarget();
  const hasAudio = Boolean(demuxed.audioTrack?.audio && audioSamplesToMux.length > 0);
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

  const keyframeInterval = Math.max(15, Math.min(120, Math.round(fps * 2)));
  const defaultDurUs = Math.round(1_000_000 / Math.max(1, fps));
  let processedFrames = 0;

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
        if (clippedSec !== null && frame.timestamp > maxUsCutoff) {
          continue;
        }
        const w = frame.displayWidth || frame.codedWidth;
        const h = frame.displayHeight || frame.codedHeight;
        engine.processFrame(
          frame,
          w,
          h,
          false,
          {
            ctx: exportCtx,
            opts: {
              splitX: 0,
              viewMode: 1,
              zoom: 1,
              centerX: 0.5,
              centerY: 0.5,
            },
          },
          { workW, workH },
        );
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
        const totalEst = Math.max(processedFrames, estimatedTargetFrames);
        const etaSec = curFps > 0 ? Math.max(0, (totalEst - processedFrames) / curFps) : 0;
        onProgress({
          phase: 'processing',
          doneFrames: processedFrames,
          totalFrames: totalEst,
          fps: curFps,
          etaSec,
          message: `本地 WebGPU 超分渲染中 (${processedFrames}/${totalEst} 帧)`,
        });
      } finally {
        frame.close();
      }
    }
  })();

  try {
    for (let i = 0; i < feedSamples.length; i++) {
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (decoderError) throw decoderError;
      if (encoderError) throw encoderError;

      while ((decoder.decodeQueueSize > 8 || decodedQueue.length > 4) && !signal.aborted && !decoderError) {
        await new Promise((r) => setTimeout(r, 4));
      }
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');

      const s = feedSamples[i];
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
      totalFrames: processedFrames,
      fps: processedFrames / Math.max(0.01, (performance.now() - startWall) / 1000),
      etaSec: 0,
      message: '正在封装最终 MP4 文件...',
    });

    await encoder.flush();

    if (hasAudio && audioSamplesToMux.length > 0) {
      for (const a of audioSamplesToMux) {
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
    fileName: makeOutputFileName(source.name, tierLabel, scale, clippedSec),
    width: encW,
    height: encH,
    frameCount: processedFrames,
    elapsedSec,
    avgFps: processedFrames / elapsedSec,
    hasAudio,
    clippedToSec: clippedSec,
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
  const { engine, source, workW, workH, bitrateMbps, tierLabel, scale, maxDurationSec, signal, onProgress } = opts;
  const startWall = performance.now();
  const clippedSec =
    maxDurationSec && maxDurationSec > 0 && source.durationSec > maxDurationSec + 0.25
      ? maxDurationSec
      : null;
  const targetDurSec = clippedSec ?? source.durationSec;

  const fps = Math.max(1, Math.min(60, Math.round(source.fps || 30)));
  const totalFrames = Math.max(1, Math.floor(targetDurSec * fps));
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

      const tSec = Math.min(Math.max(0, source.durationSec - 0.001), i * stepSec);
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        video.addEventListener('seeked', onSeeked);
        video.currentTime = Math.max(0, tSec);
      });

      engine.processFrame(
        video,
        video.videoWidth,
        video.videoHeight,
        false,
        {
          ctx: exportCtx,
          opts: {
            splitX: 0,
            viewMode: 1,
            zoom: 1,
            centerX: 0.5,
            centerY: 0.5,
          },
        },
        { workW, workH },
      );
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
      totalFrames: processed,
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
    fileName: makeOutputFileName(source.name, tierLabel, scale, clippedSec),
    width: encW,
    height: encH,
    frameCount: processed,
    elapsedSec,
    avgFps: processed / elapsedSec,
    hasAudio: false,
    clippedToSec: clippedSec,
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
 * Exports the video super-resolved with local WebGPU into a downloadable H.264 MP4.
 * - Uses chunked MP4Box + WebCodecs VideoDecoder (H.264 / HEVC / VP9 / AV1) when unrotated & supported.
 * - Automatically falls back to browser-rotated <video> stepping for MOV/WebM/rotated phone clips.
 * - Supports optional `maxDurationSec` (e.g. first 10s fast clip export).
 */
export async function exportSuperResolvedMp4(opts: ExportOptions): Promise<ExportResult> {
  const rawOutW = Math.round(opts.workW * opts.scale);
  const rawOutH = Math.round(opts.workH * opts.scale);
  // H.264 strictly requires even dimensions
  const encW = Math.max(16, rawOutW & ~1);
  const encH = Math.max(16, rawOutH & ~1);

  if (typeof VideoDecoder !== 'undefined' && !opts.source.isRotated) {
    try {
      return await exportViaMp4BoxWebCodecs(opts, encW, encH);
    } catch (err) {
      if (opts.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
        throw err;
      }
      console.warn('[CodecSR Export] MP4Box/VideoDecoder path failed, falling back to <video> stepping:', err);
    }
  }

  return await exportViaVideoElementFallback(opts, encW, encH);
}
