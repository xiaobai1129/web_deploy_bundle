import type { ModelScale, ModelTier } from './csr1';

export interface ModelOptionInfo {
  id: string;
  tier: ModelTier;
  scale: ModelScale;
  usePrior: boolean;
  fileName: string;
  label: string;
  tagline: string;
  sizeKb: number;
  gmacs: number;
  activeParamsK: number;
}

export const MODEL_CATALOG: ModelOptionInfo[] = [
  {
    id: 'L_540p_prior',
    tier: 'L',
    scale: 2.0,
    usePrior: true,
    fileName: 'client_L_540p_prior.bin',
    label: 'Client-L · 极速档',
    tagline: '超低延迟 · 1.57 GMACs · 带 7D 先验 + Seam3DLUT',
    sizeKb: 143,
    gmacs: 1.57,
    activeParamsK: 18.3,
  },
  {
    id: 'L_540p_noprior',
    tier: 'L',
    scale: 2.0,
    usePrior: false,
    fileName: 'client_L_540p_noprior.bin',
    label: 'Client-L · 静态极速',
    tagline: '极小体积 38 KB · 无动态先验',
    sizeKb: 38,
    gmacs: 1.57,
    activeParamsK: 18.3,
  },
  {
    id: 'S_540p_prior',
    tier: 'S',
    scale: 2.0,
    usePrior: true,
    fileName: 'client_S_540p_prior.bin',
    label: 'Client-S · 标准档 (推荐)',
    tagline: '画质与速度黄金平衡 · 5.23 GMACs · Seam3DLUT',
    sizeKb: 289,
    gmacs: 5.23,
    activeParamsK: 47.4,
  },
  {
    id: 'S_540p_noprior',
    tier: 'S',
    scale: 2.0,
    usePrior: false,
    fileName: 'client_S_540p_noprior.bin',
    label: 'Client-S · 静态标准',
    tagline: '95 KB · 无动态先验',
    sizeKb: 95,
    gmacs: 5.23,
    activeParamsK: 47.4,
  },
  {
    id: 'P_540p_prior',
    tier: 'P',
    scale: 2.0,
    usePrior: true,
    fileName: 'client_P_540p_prior.bin',
    label: 'Client-P · 旗舰沙漏',
    tagline: '双尺度沙漏结构 · 极致细节修复 · 23.91 GMACs',
    sizeKb: 407,
    gmacs: 23.91,
    activeParamsK: 164.0,
  },
  {
    id: 'L_720p_prior',
    tier: 'L',
    scale: 1.5,
    usePrior: true,
    fileName: 'client_L_720p_prior.bin',
    label: 'Client-L · 极速档',
    tagline: '×1.5 超分 · 2.56 GMACs · 带 7D 先验 + Seam3DLUT',
    sizeKb: 132,
    gmacs: 2.56,
    activeParamsK: 17.3,
  },
  {
    id: 'L_720p_noprior',
    tier: 'L',
    scale: 1.5,
    usePrior: false,
    fileName: 'client_L_720p_noprior.bin',
    label: 'Client-L · 静态极速',
    tagline: '×1.5 超分 · 36 KB · 无动态先验',
    sizeKb: 36,
    gmacs: 2.56,
    activeParamsK: 17.3,
  },
  {
    id: 'S_720p_prior',
    tier: 'S',
    scale: 1.5,
    usePrior: true,
    fileName: 'client_S_720p_prior.bin',
    label: 'Client-S · 标准档 (推荐)',
    tagline: '×1.5 超分 · 6.62 GMACs · Seam3DLUT',
    sizeKb: 210,
    gmacs: 6.62,
    activeParamsK: 35.5,
  },
  {
    id: 'S_720p_noprior',
    tier: 'S',
    scale: 1.5,
    usePrior: false,
    fileName: 'client_S_720p_noprior.bin',
    label: 'Client-S · 静态标准',
    tagline: '×1.5 超分 · 72 KB · 无动态先验',
    sizeKb: 72,
    gmacs: 6.62,
    activeParamsK: 35.5,
  },
  {
    id: 'P_720p_prior',
    tier: 'P',
    scale: 1.5,
    usePrior: true,
    fileName: 'client_P_720p_prior.bin',
    label: 'Client-P · 旗舰沙漏',
    tagline: '×1.5 双尺度沙漏 · 极致细节 · 37.42 GMACs',
    sizeKb: 396,
    gmacs: 37.42,
    activeParamsK: 159.8,
  },
];

export function resolveModelOption(tier: ModelTier, scale: ModelScale, usePrior: boolean): ModelOptionInfo {
  // Client-P only has prior-enabled models
  const effectivePrior = tier === 'P' ? true : usePrior;
  const match = MODEL_CATALOG.find(
    (m) => m.tier === tier && Math.abs(m.scale - scale) < 1e-4 && m.usePrior === effectivePrior,
  );
  if (!match) {
    throw new Error(`No model found for tier=${tier}, scale=${scale}, usePrior=${effectivePrior}`);
  }
  return match;
}

export function getWeightUrl(fileName: string): string {
  return `/weights/${fileName}`;
}
