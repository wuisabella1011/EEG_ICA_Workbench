"""
signal_processor.py — EEG 信号处理核心模块
============================================
本模块实现了科研级 EEG 数据预处理流水线的三个核心功能：
  1. 模拟 EEG 数据生成（含多种噪声成分）
  2. 独立成分分析 (ICA) 分解
  3. 功率谱密度 (PSD) 估计

设计原则：
  - 所有函数均为纯函数，便于单元测试和可复现研究
  - 参数可通过前端 Streamlit 界面动态调节
  - 遵循 MNE-Python 和 FieldTrip 社区约定的命名与数据结构

作者: EEG ICA Workbench 项目组
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.decomposition import FastICA
from scipy import signal


# ============================================================================
# 常量定义
# ============================================================================

# 标准 10-20 系统 EEG 通道标签（19 通道 + 参考）
DEFAULT_CHANNELS: list = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

DEFAULT_SFREQ: float = 250.0        # 采样率 (Hz)
DEFAULT_DURATION: float = 10.0       # 数据时长 (秒)
DEFAULT_N_COMPONENTS: int = 10       # 默认 ICA 分量数


# ============================================================================
# 1. 模拟 EEG 数据生成
# ============================================================================

def generate_mock_eeg(
    n_channels: int = 19,
    n_samples: Optional[int] = None,
    sfreq: float = DEFAULT_SFREQ,
    duration: float = DEFAULT_DURATION,
    noise_config: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成具有真实噪声特征的模拟 EEG 数据。

    模拟策略：
      数据 = 背景振荡 + 瞬态伪迹（眨眼/肌电） + 工频噪声 + 高斯白噪声

    背景振荡：
      - Alpha 波段 (8–13 Hz)：枕叶区 (O1, O2) 主导，模拟闭眼静息态
      - Beta 波段  (13–30 Hz)：中央区 (C3, Cz, C4) 主导，模拟运动皮层活动
      - Theta 波段 (4–8 Hz)：额叶区 (F3, Fz, F4) 主导，模拟认知负荷
      - Delta 波段 (0.5–4 Hz)：广泛分布，模拟慢波活动

    瞬态伪迹：
      - 眼电伪迹 (EOG)：前额通道 (Fp1, Fp2) 上的大振幅低频偏移
      - 肌电伪迹 (EMG)：颞叶通道 (T3, T4, T5, T6) 上的高频突发

    工频噪声：
      - 50 Hz 线噪声及其谐波 (100 Hz)，主要在周边通道

    参数:
        n_channels: 通道数，默认 19（标准 10-20 系统）
        n_samples: 样本点数；若为 None 则由 sfreq * duration 自动计算
        sfreq: 采样率 (Hz)
        duration: 数据时长 (秒)
        noise_config: 可选噪声参数覆盖字典，键:
            - alpha_amplitude, beta_amplitude, theta_amplitude, delta_amplitude
            - eog_amplitude, emg_amplitude, line_noise_amplitude
            - white_noise_amplitude
        random_state: 随机种子

    返回:
        (data, time): data 形状为 (n_channels, n_samples) 的 EEG 数据数组，
                      time 为时间轴 (秒)
    """
    rng = np.random.RandomState(random_state)

    # --- 解析参数 ---
    if n_samples is None:
        n_samples = int(sfreq * duration)

    cfg = {
        "alpha_amplitude": 30.0,
        "beta_amplitude": 15.0,
        "theta_amplitude": 20.0,
        "delta_amplitude": 25.0,
        "eog_amplitude": 150.0,
        "emg_amplitude": 80.0,
        "line_noise_amplitude": 10.0,
        "white_noise_amplitude": 5.0,
    }
    if noise_config is not None:
        cfg.update(noise_config)

    time = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))

    # --- 通道索引 ---
    # 按 10-20 系统大致位置分配：前额 (0-1) → 额叶 (2-7) → 中央 (8-11) →
    #   颞叶 (12-15) → 顶叶 (16) → 枕叶 (17-18)
    frontal_idx  = [0, 1, 2, 3, 4, 5, 6, 7]       # Fp1..F8
    central_idx  = [8, 9, 10, 11]                   # T3, C3, Cz, C4
    temporal_idx = [12, 13, 14, 15]                 # T4, T5, P3, Pz → 修正为颞叶
    occipital_idx = [16, 17, 18]                    # P4, T6, O1, O2 → 枕叶取后部通道

    # 更精确的解剖映射（基于 DEFAULT_CHANNELS 顺序）
    # Fp1(0),Fp2(1),F7(2),F3(3),Fz(4),F4(5),F8(6),T3(7),C3(8),Cz(9),
    # C4(10),T4(11),T5(12),P3(13),Pz(14),P4(15),T6(16),O1(17),O2(18)
    frontal   = [0, 1, 2, 3, 4, 5, 6]              # Fp1..F8
    temporal  = [7, 11, 12, 16]                     # T3, T4, T5, T6
    central   = [8, 9, 10]                          # C3, Cz, C4
    parietal  = [13, 14, 15]                        # P3, Pz, P4
    occipital = [17, 18]                            # O1, O2

    # --- (A) 背景神经振荡 ---
    for ch in range(n_channels):
        # 为每个通道生成带限噪声振荡
        # Alpha: 8–13 Hz, 枕叶主导, 其他区域衰减
        alpha_amp = cfg["alpha_amplitude"] * (1.0 if ch in occipital else 0.3)
        data[ch] += _band_limited_oscillation(
            n_samples, sfreq, low=8.0, high=13.0, amplitude=alpha_amp, rng=rng
        )

        # Beta: 13–30 Hz, 中央区主导
        beta_amp = cfg["beta_amplitude"] * (1.0 if ch in central else 0.2)
        data[ch] += _band_limited_oscillation(
            n_samples, sfreq, low=13.0, high=30.0, amplitude=beta_amp, rng=rng
        )

        # Theta: 4–8 Hz, 额叶主导
        theta_amp = cfg["theta_amplitude"] * (1.0 if ch in frontal else 0.2)
        data[ch] += _band_limited_oscillation(
            n_samples, sfreq, low=4.0, high=8.0, amplitude=theta_amp, rng=rng
        )

        # Delta: 0.5–4 Hz, 广泛分布
        data[ch] += _band_limited_oscillation(
            n_samples, sfreq, low=0.5, high=4.0,
            amplitude=cfg["delta_amplitude"] * 0.5, rng=rng
        )

    # --- (B) 眼电伪迹 (EOG) ---
    # 眼动/眨眼信号：低频 (<5 Hz) 大振幅偏移，前额通道 (Fp1, Fp2) 最显著
    eog_signal = _generate_eog_artifact(
        n_samples, sfreq, duration,
        amplitude=cfg["eog_amplitude"], rng=rng
    )
    # 前额通道加权最大，其他通道按距离衰减
    for ch in frontal:
        weight = 2.0 if ch in (0, 1) else 0.8  # Fp1, Fp2 最重
        data[ch] += eog_signal * weight

    # --- (C) 肌电伪迹 (EMG) ---
    # 高频率 (>20 Hz) 突发噪声，颞叶通道 (T3, T4, T5, T6) 最显著
    for ch in temporal:
        emg_burst = _generate_emg_burst(
            n_samples, sfreq, duration,
            amplitude=cfg["emg_amplitude"], rng=rng
        )
        data[ch] += emg_burst

    # --- (D) 工频噪声 ---
    # 50 Hz 线噪声 + 100 Hz 谐波
    line_noise = np.zeros(n_samples)
    line_noise += cfg["line_noise_amplitude"] * np.sin(2 * np.pi * 50.0 * time)
    line_noise += cfg["line_noise_amplitude"] * 0.3 * np.sin(2 * np.pi * 100.0 * time)
    # 加入小幅相位和幅度随机波动以模拟真实工频干扰
    line_noise += cfg["line_noise_amplitude"] * 0.15 * rng.randn(n_samples)
    for ch in range(n_channels):
        data[ch] += line_noise * rng.uniform(0.5, 1.2)

    # --- (E) 高斯白噪声（传感器底噪） ---
    data += cfg["white_noise_amplitude"] * rng.randn(n_channels, n_samples)

    return data, time


# ============================================================================
# 1a. 辅助：带限振荡生成器
# ============================================================================

def _band_limited_oscillation(
    n_samples: int,
    sfreq: float,
    low: float,
    high: float,
    amplitude: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    生成指定频段的带限振荡信号。
    方法：使用 FIR 带通滤波器对白噪声进行滤波，产生自然的窄带振荡。

    参数:
        n_samples: 样本点数
        sfreq: 采样率
        low, high: 带通上下截止频率 (Hz)
        amplitude: 缩放振幅
        rng: NumPy 随机数生成器

    返回:
        形状为 (n_samples,) 的带限信号
    """
    nyquist = sfreq / 2.0
    # 设计 FIR 带通滤波器（Kaiser 窗，高阻带衰减）
    taps = signal.firwin(
        numtaps=min(501, n_samples // 4),
        cutoff=[low / nyquist, high / nyquist],
        pass_zero=False,
        window="blackman",
    )
    # 白噪声驱动
    white = rng.randn(n_samples + len(taps) - 1)
    filtered = signal.lfilter(taps, [1.0], white)
    # 截断为原始长度，并归一化
    filtered = filtered[len(taps) - 1:]
    if np.std(filtered) > 0:
        filtered = filtered / np.std(filtered)
    return amplitude * filtered


# ============================================================================
# 1b. 辅助：EOG 伪迹生成
# ============================================================================

def _generate_eog_artifact(
    n_samples: int,
    sfreq: float,
    duration: float,
    amplitude: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    生成模拟眼电伪迹（如眨眼）。
    特征：每隔约 2-3 秒出现一次大振幅单相偏移，宽度约 200–400 ms。

    参数:
        n_samples, sfreq, duration: 数据参数
        amplitude: 峰值幅度 (µV)
        rng: 随机数生成器

    返回:
        形状为 (n_samples,) 的伪迹信号
    """
    eog = np.zeros(n_samples)
    time = np.arange(n_samples) / sfreq

    # 随机生成眨眼事件，平均每 2.5 秒一次
    blink_interval_mean = 2.5
    n_blinks_expected = int(duration / blink_interval_mean)
    # 随机化事件时间
    blink_times = np.cumsum(rng.exponential(blink_interval_mean, size=n_blinks_expected + 2))
    blink_times = blink_times[blink_times < duration * 0.95]

    for t_blink in blink_times:
        idx_start = int(t_blink * sfreq)
        # 眨眼宽度：200–400 ms 的高斯脉冲
        width = int(sfreq * rng.uniform(0.20, 0.40))
        idx_end = min(idx_start + width, n_samples)
        if idx_end <= idx_start:
            continue
        indices = np.arange(idx_start, idx_end)
        # 生成不对称脉冲：快速上升 → 缓慢下降（模拟真实眨眼形状）
        t_local = (indices - idx_start) / sfreq
        # Rice 脉冲形状
        pulse = (t_local / 0.05) * np.exp(-t_local / 0.08)
        eog[indices] += amplitude * rng.uniform(0.7, 1.3) * pulse

    return eog


# ============================================================================
# 1c. 辅助：EMG 突发伪迹生成
# ============================================================================

def _generate_emg_burst(
    n_samples: int,
    sfreq: float,
    duration: float,
    amplitude: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    生成模拟肌电伪迹。
    特征：高频 (30–80 Hz) 短时突发，持续 50–150 ms。

    参数:
        n_samples, sfreq, duration: 数据参数
        amplitude: 峰值幅度 (µV)
        rng: 随机数生成器

    返回:
        形状为 (n_samples,) 的伪迹信号
    """
    emg = np.zeros(n_samples)

    # 随机突发事件（平均每秒 1-2 次）
    burst_interval_mean = 0.8
    n_bursts = int(duration / burst_interval_mean)
    burst_times = np.cumsum(rng.exponential(burst_interval_mean, size=n_bursts + 3))
    burst_times = burst_times[burst_times < duration * 0.95]

    for t_burst in burst_times:
        idx_start = int(t_burst * sfreq)
        # 突发宽度：50–150 ms
        width = int(sfreq * rng.uniform(0.05, 0.15))
        idx_end = min(idx_start + width, n_samples)
        if idx_end <= idx_start:
            continue
        # 高频噪声突发
        burst = rng.randn(idx_end - idx_start)
        # 使用汉宁窗平滑包络
        window = np.hanning(len(burst))
        emg[idx_start:idx_end] += amplitude * rng.uniform(0.5, 1.0) * burst * window

    return emg


# ============================================================================
# 2. 独立成分分析 (ICA)
# ============================================================================

def run_ica(
    data: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
    random_state: int = 42,
    max_iter: int = 2000,
    tol: float = 1e-4,
    algorithm: str = "parallel",
    fun: str = "logcosh",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FastICA]:
    """
    使用 FastICA 算法对 EEG 数据进行独立成分分解。

    算法原理：

      FastICA 基于互信息极小化准则，通过固定点迭代寻找线性变换 W，
      使得输出 S = W * X 的各分量之间统计独立性最大化。

      在 EEG 去噪场景中，ICA 能够将眨眼、肌电、工频等伪迹分离为
      独立的成分 (IC)，研究者可据此识别并剔除噪声分量，从而保留
      真实的神经活动信号。这是脑连接分析中至关重要的预处理步骤。

    参数:
        data: 形状为 (n_channels, n_samples) 的 EEG 数据矩阵
        n_components: 要提取的独立分量数
        random_state: 随机种子（保证可复现性）
        max_iter: 最大迭代次数
        tol: 收敛容差
        algorithm: FastICA 算法变体 ("parallel" | "deflation")
        fun: 非线性函数 ("logcosh" | "exp" | "cube")

    返回:
        (S, A, W, ica_obj):
          S        — 形状 (n_components, n_samples) 的独立分量时间序列
          A        — 形状 (n_channels, n_components) 的混合矩阵
          W        — 形状 (n_components, n_channels) 的解混矩阵
          ica_obj  — 训练好的 FastICA 对象（供后续 transform/inverse_transform 使用）

    异常:
        ValueError: 当 n_components > n_channels 时抛出
    """
    n_channels, n_samples = data.shape

    if n_components > n_channels:
        raise ValueError(
            f"ICA 分量数 ({n_components}) 不能超过通道数 ({n_channels})。"
            f"请减少 n_components 或增加数据通道数。"
        )

    # 数据转置为 (n_samples, n_channels) — sklearn 要求 (samples, features)
    X = data.T.copy()

    # 初始化 FastICA
    ica = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        whiten="unit-variance",  # 白化预处理：各分量方差归一化
    )

    # 拟合并分解
    S = ica.fit_transform(X)       # 独立分量, shape: (n_samples, n_components)

    # 获取混合矩阵 A 和解混矩阵 W
    A = ica.mixing_                # (n_channels, n_components)
    W = ica.components_            # (n_components, n_channels)

    # 统一输出形状
    S = S.T                        # (n_components, n_samples)

    return S, A, W, ica


# ============================================================================
# 3. 功率谱密度 (PSD) 估计
# ============================================================================

def calculate_psd(
    data: np.ndarray,
    sfreq: float = DEFAULT_SFREQ,
    method: str = "welch",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    fmax: Optional[float] = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Welch 方法估计各通道/分量的功率谱密度。

    科研意义：
      PSD 是评估信号质量的定量指标。在 ICA 去噪前后对比 PSD，可以：
        1. 量化 α (8–13 Hz) 波段信噪比的改善程度
        2. 验证伪迹去除后高频成分 (>30 Hz) 的衰减是否合理
        3. 检测是否存在残留的工频干扰 (50 Hz) 或谐波
      这些指标直接关系到后续脑连接分析（如 coherence、PLV、格兰杰因果）
      的可靠性——"Garbage in, garbage out" 原则在脑网络分析中尤为严峻。

    参数:
        data: 形状为 (n_signals, n_samples) 的输入数据
        sfreq: 采样率 (Hz)
        method: PSD 估计方法 ("welch" 使用 Welch 平均周期图法)
        nperseg: 每段样本数 (None → 自动选择)
        noverlap: 段重叠样本数 (None → nperseg // 2)
        fmax: 最大频率 (Hz)，用于截断显示；None 表示全部保留

    返回:
        (freqs, psd, psd_mean):
          freqs    — 频率轴 (Hz), 形状 (n_freqs,)
          psd      — PSD 矩阵, 形状 (n_signals, n_freqs)，单位 µV²/Hz
          psd_mean — 平均 PSD (所有信号取平均)，形状 (n_freqs,)

    参考文献:
        Welch, P.D. (1967). "The use of Fast Fourier Transform for the
        estimation of power spectra." IEEE Trans. Audio Electroacoust.
    """
    n_signals = data.shape[0]

    # 自动选择窗长：约 2 秒的 Welch 窗
    if nperseg is None:
        nperseg = min(int(sfreq * 2.0), data.shape[1] // 2)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs = None
    psds = []

    for i in range(n_signals):
        f, pxx = signal.welch(
            data[i],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",      # 去除直流分量
            scaling="density",       # 功率谱密度 (µV²/Hz)
        )
        if freqs is None:
            freqs = f
        psds.append(pxx)

    psd = np.array(psds)            # (n_signals, n_freqs)
    psd_mean = psd.mean(axis=0)     # (n_freqs,)

    # 截断频率
    if fmax is not None:
        freq_mask = freqs <= fmax
        freqs = freqs[freq_mask]
        psd = psd[:, freq_mask]
        psd_mean = psd_mean[freq_mask]

    return freqs, psd, psd_mean


# ============================================================================
# 4. 频带功率提取
# ============================================================================

def extract_band_powers(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """
    从 PSD 中提取各频带的平均功率。

    标准 EEG 频带（与临床脑电图报告一致）：
      Delta: 0.5–4 Hz   — 深度睡眠/病理慢波
      Theta: 4–8 Hz     — 瞌睡/认知负荷
      Alpha: 8–13 Hz    — 闭眼静息/皮层空闲节律
      Beta:  13–30 Hz   — 活跃思维/运动准备
      Gamma: 30–80 Hz   — 高级认知/感知绑定

    参数:
        freqs: 频率轴 (Hz)
        psd: PSD 矩阵 (n_signals, n_freqs)
        bands: 自定义频带字典，None 则使用默认

    返回:
        字典，键为频带名称，值为 (n_signals,) 数组（各信号在该频带的平均功率）
    """
    if bands is None:
        bands = {
            "Delta (0.5–4 Hz)":  (0.5, 4.0),
            "Theta (4–8 Hz)":    (4.0, 8.0),
            "Alpha (8–13 Hz)":   (8.0, 13.0),
            "Beta (13–30 Hz)":   (13.0, 30.0),
            "Gamma (30–80 Hz)":  (30.0, 80.0),
        }

    band_powers: Dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_powers[name] = np.trapz(psd[:, mask], freqs[mask], axis=1)

    return band_powers


# ============================================================================
# 5. 信号信噪比 (SNR) 估计
# ============================================================================

def estimate_snr(
    data: np.ndarray,
    sfreq: float = DEFAULT_SFREQ,
    signal_band: Tuple[float, float] = (1.0, 45.0),
    noise_band: Tuple[float, float] = (45.0, 80.0),
) -> np.ndarray:
    """
    估计每个通道的信噪比 (SNR)。

    方法：
      SNR = 10 * log10(P_signal / P_noise)
      其中 P_signal 为信号频带的总功率，P_noise 为高频噪声频带的功率。

    参数:
        data: EEG 数据矩阵 (n_channels, n_samples)
        sfreq: 采样率
        signal_band: 信号频带 (low, high) Hz
        noise_band: 噪声频带 (low, high) Hz

    返回:
        snr: (n_channels,) 数组，每通道 SNR (dB)
    """
    freqs, psd, _ = calculate_psd(data, sfreq=sfreq, fmax=noise_band[1])

    p_signal = _band_integral(freqs, psd, signal_band[0], signal_band[1])
    p_noise = _band_integral(freqs, psd, noise_band[0], noise_band[1])

    with np.errstate(divide="ignore"):
        snr = 10.0 * np.log10(p_signal / p_noise)
    # 将 inf 替换为大型有限值
    snr = np.nan_to_num(snr, nan=0.0, posinf=60.0, neginf=-60.0)

    return snr


def _band_integral(
    freqs: np.ndarray, psd: np.ndarray, low: float, high: float
) -> np.ndarray:
    """辅助函数：计算指定频带内的积分功率"""
    mask = (freqs >= low) & (freqs <= high)
    if np.sum(mask) == 0:
        return np.zeros(psd.shape[0])
    return np.trapz(psd[:, mask], freqs[mask], axis=1)
