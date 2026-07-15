"""
pages/1_📈_Signal_Processing.py — EEG 信号处理与 ICA 去噪界面
================================================================
本页面是 EEG ICA Workbench 的核心交互界面，提供：

  左侧面板：
    - 原始 EEG 数据波形图（各通道时间序列）
    - 原始数据的功率谱密度 (PSD) 图
    - 频带功率柱状图（Delta / Theta / Alpha / Beta / Gamma）

  右侧面板：
    - ICA 分解后的独立分量 (IC) 时间序列
    - 各 IC 的功率谱密度
    - 混合矩阵热力图（展示各 IC 在各通道上的空间权重分布）

  侧边栏控件：
    - ICA 分量数调节
    - ICA 算法选择（parallel / deflation）
    - 非线性函数选择 (logcosh / exp / cube)
    - 工频噪声幅值调节

设计原则：
  - 所有图表使用 Plotly 实现，支持交互式缩放和拖拽
  - 配色采用科学可视化友好方案
  - 每个图表配有科研意义说明

作者: EEG ICA Workbench 项目组
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 确保 src 目录在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signal_processor import (
    generate_mock_eeg,
    run_ica,
    calculate_psd,
    extract_band_powers,
    DEFAULT_CHANNELS,
    DEFAULT_SFREQ,
    DEFAULT_DURATION,
)

# ============================================================================
# 页面配置
# ============================================================================

st.title("📈 EEG 信号处理与 ICA 去噪分析")
st.markdown(
    """
    本页面展示完整的 **ICA 去噪流水线**。左侧为原始信号分析，右侧为 ICA
    分解结果。通过对比两边的 PSD 曲线，可以定量评估去噪效果。
    """
)

# ============================================================================
# 侧边栏 — 参数控制
# ============================================================================

st.sidebar.header("🎛️ 实验参数设置")

# --- 数据生成参数 ---
st.sidebar.subheader("数据生成")
sfreq = st.sidebar.slider(
    "采样率 (Hz)", min_value=100, max_value=1000,
    value=int(DEFAULT_SFREQ), step=50,
    help="模拟 EEG 的采样频率。临床常用 250 Hz 或 500 Hz。"
)
duration = st.sidebar.slider(
    "数据时长 (秒)", min_value=2, max_value=30,
    value=int(DEFAULT_DURATION), step=1,
    help="模拟数据的长度。较长时间有助于更好地估计 PSD。"
)
line_noise_amplitude = st.sidebar.slider(
    "工频噪声幅度 (µV)", min_value=0.0, max_value=50.0,
    value=10.0, step=1.0,
    help="50 Hz 工频干扰的强度。设置较高值可观察 ICA 如何分离线噪声。"
)
eog_amplitude = st.sidebar.slider(
    "眼电伪迹幅度 (µV)", min_value=0.0, max_value=300.0,
    value=150.0, step=10.0,
    help="眨眼/眼动伪迹的强度。这是 EEG 中最常见的伪迹来源。"
)
emg_amplitude = st.sidebar.slider(
    "肌电伪迹幅度 (µV)", min_value=0.0, max_value=200.0,
    value=80.0, step=10.0,
    help="肌肉活动伪迹的强度，主要影响颞叶通道。"
)
random_seed = st.sidebar.number_input(
    "随机种子", value=42, min_value=0, max_value=9999,
    help="固定随机种子以保证结果可复现。更改种子可生成不同的模拟数据。"
)

st.sidebar.markdown("---")

# --- ICA 参数 ---
st.sidebar.subheader("ICA 参数")
n_components = st.sidebar.slider(
    "ICA 分量数",
    min_value=5, max_value=19, value=10, step=1,
    help=(
        "要提取的独立分量数量。理论上不应超过通道数。\n\n"
        "较少的分量数可能合并多个信号源，较多的分量数可能导致过拟合。"
        "建议从 10 开始调整。"
    ),
)
ica_algorithm = st.sidebar.selectbox(
    "ICA 算法",
    options=["parallel", "deflation"],
    index=0,
    help=(
        "**parallel**: 同时估计所有分量（默认，通常更快）。\n"
        "**deflation**: 逐个估计分量，可能更稳定但速度较慢。"
    ),
)
ica_fun = st.sidebar.selectbox(
    "非线性函数",
    options=["logcosh", "exp", "cube"],
    index=0,
    help=(
        "FastICA 使用的对比函数：\n"
        "**logcosh**: 通用稳健选择（默认）\n"
        "**exp**: 适用于超高斯源（如 EEG 瞬态伪迹）\n"
        "**cube**: 适用于亚高斯源"
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 📖 使用说明

    1. 调整左侧参数以改变模拟数据特征
    2. 观察左侧原始信号的 **时域波形** 和 **PSD**
    3. 对照右侧 ICA 分解后的 **独立分量** 及其 **PSD**
    4. 注意 50 Hz 处的工频噪声峰值在各分量中的分布

    **科研提示**: 理想情况下，ICA 应能将眼电伪迹集中到 1–2 个分量中，
    工频噪声集中到另外 1–2 个分量。通过剔除这些伪迹分量并重构信号，
    即可获得干净的 EEG 数据用于后续脑连接分析。
    """
)

# ============================================================================
# 数据生成（利用缓存避免重复计算）
# ============================================================================

@st.cache_data(ttl=10)
def load_data(
    sfreq: float,
    duration: float,
    line_noise_amp: float,
    eog_amp: float,
    emg_amp: float,
    seed: int,
):
    """生成模拟 EEG 数据（带缓存）"""
    noise_config = {
        "line_noise_amplitude": line_noise_amp,
        "eog_amplitude": eog_amp,
        "emg_amplitude": emg_amp,
    }
    data, time = generate_mock_eeg(
        n_channels=19,
        sfreq=sfreq,
        duration=duration,
        noise_config=noise_config,
        random_state=seed,
    )
    return data, time


@st.cache_data(ttl=10)
def compute_ica(data: np.ndarray, n_comp: int, algorithm: str, fun: str, seed: int):
    """执行 ICA 分解（带缓存）"""
    S, A, W, ica_obj = run_ica(
        data,
        n_components=n_comp,
        algorithm=algorithm,
        fun=fun,
        random_state=seed,
    )
    return S, A, W, ica_obj


def compute_psds(data: np.ndarray, sfreq: float):
    """计算原始数据和 ICA 源的 PSD"""
    f_raw, psd_raw, psd_raw_mean = calculate_psd(data, sfreq=sfreq)
    return f_raw, psd_raw, psd_raw_mean


# 加载数据
data, time_vec = load_data(
    sfreq, duration, line_noise_amplitude, eog_amplitude, emg_amplitude, random_seed
)

# ICA 分解
S_ica, A_mix, W_unmix, ica_obj = compute_ica(
    data, n_components, ica_algorithm, ica_fun, random_seed
)

# PSD 计算
f_raw, psd_raw, psd_raw_mean = compute_psds(data, sfreq)
f_ica, psd_ica, psd_ica_mean = calculate_psd(S_ica, sfreq=sfreq)

# ============================================================================
# 可视化
# ============================================================================

# --- Plotly 配色方案 ---
COLOR_RAW = "#1f77b4"       # 原始数据蓝色
COLOR_ICA = "#d62728"       # ICA 分量红色
COLOR_PSD_FILL = "rgba(31, 119, 180, 0.1)"   # PSD 填充色
COLOR_ALPHA_BAND = "rgba(255, 127, 14, 0.15)" # Alpha 频带高亮
COLOR_LINE_NOISE = "rgba(0, 0, 0, 0.15)"      # 50 Hz 标记

# ============================================================================
# 主布局：左栏 (原始数据) vs 右栏 (ICA 结果)
# ============================================================================

col_left, col_right = st.columns(2)

# ============================================================================
# 左栏: 原始 EEG 数据
# ============================================================================

with col_left:
    st.markdown("### 🔴 原始 EEG 数据")

    # --- 1a. 原始 EEG 波形图 ---
    st.markdown(
        """
        **时域波形 (Time Series)**
        <br><small>展示各通道的原始 EEG 信号。注意前额通道 (Fp1, Fp2) 中的
        大振幅眼电伪迹，以及颞叶通道 (T3–T6) 中的肌电高频突发。</small>
        """,
        unsafe_allow_html=True,
    )

    n_display_ch = min(10, data.shape[0])  # 显示前 10 通道以避免过度拥挤
    fig_raw_ts = make_subplots(
        rows=n_display_ch, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    for i in range(n_display_ch):
        fig_raw_ts.add_trace(
            go.Scattergl(
                x=time_vec,
                y=data[i],
                mode="lines",
                line=dict(color=COLOR_RAW, width=0.6),
                name=DEFAULT_CHANNELS[i],
                showlegend=False,
            ),
            row=i + 1, col=1,
        )
        # 通道标签
        fig_raw_ts.update_yaxes(
            title_text=DEFAULT_CHANNELS[i],
            title_standoff=0,
            row=i + 1, col=1,
        )

    fig_raw_ts.update_xaxes(title_text="时间 (秒)", row=n_display_ch, col=1)
    fig_raw_ts.update_layout(
        height=400,
        margin=dict(l=50, r=20, t=20, b=40),
        title="原始 EEG 信号（10 通道）",
        hovermode="x unified",
    )
    st.plotly_chart(fig_raw_ts, use_container_width=True)

    # --- 1b. PSD 图 ---
    st.markdown(
        """
        **功率谱密度 (Power Spectral Density)**
        <br><small>Welch 法估计的 PSD。灰色半透明区域为各通道 PSD 范围，
        蓝色实线为平均 PSD。橙色高亮为 Alpha 波段 (8–13 Hz)。注意 50 Hz
        处的工频噪声峰值。</small>
        """,
        unsafe_allow_html=True,
    )

    fig_raw_psd = go.Figure()

    # PSD 范围填充
    psd_raw_min = np.min(psd_raw, axis=0)
    psd_raw_max = np.max(psd_raw, axis=0)
    # 转换为 dB 以便显示
    psd_db = 10 * np.log10(psd_raw + 1e-12)
    psd_db_mean = np.mean(psd_db, axis=0)
    psd_db_min = np.min(psd_db, axis=0)
    psd_db_max = np.max(psd_db, axis=0)

    fig_raw_psd.add_trace(go.Scatter(
        x=f_raw, y=psd_db_max,
        mode="lines", line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_raw_psd.add_trace(go.Scatter(
        x=f_raw, y=psd_db_min,
        mode="lines",
        fill="tonexty",
        fillcolor=COLOR_PSD_FILL,
        line=dict(width=0),
        name="各通道 PSD 范围",
        hoverinfo="skip",
    ))

    # 平均 PSD
    fig_raw_psd.add_trace(go.Scatter(
        x=f_raw, y=psd_db_mean,
        mode="lines",
        line=dict(color=COLOR_RAW, width=2),
        name="平均 PSD",
    ))

    # Alpha 波段高亮
    fig_raw_psd.add_vrect(
        x0=8, x1=13,
        fillcolor=COLOR_ALPHA_BAND,
        layer="below", line_width=0,
        annotation_text="Alpha (8–13 Hz)",
        annotation_position="top left",
    )

    # 50 Hz 线标记
    if line_noise_amplitude > 0.5:
        fig_raw_psd.add_vline(
            x=50, line_dash="dash", line_color="gray",
            annotation_text="50 Hz 工频",
            annotation_position="top",
            opacity=0.7,
        )
        fig_raw_psd.add_vline(
            x=100, line_dash="dot", line_color="lightgray",
            annotation_text="100 Hz 谐波",
            annotation_position="top",
            opacity=0.4,
        )

    fig_raw_psd.update_layout(
        xaxis_title="频率 (Hz)",
        yaxis_title="功率谱密度 (dB/Hz)",
        height=380,
        margin=dict(l=50, r=20, t=20, b=40),
        title="原始数据功率谱密度 (Welch 法)",
        hovermode="x",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig_raw_psd, use_container_width=True)

    # --- 1c. 频带功率柱状图 ---
    st.markdown(
        """
        **频带功率分布**
        <br><small>各经典 EEG 频带的平均功率。此图有助于快速评估哪些频带
        受到噪声污染最为严重。</small>
        """,
        unsafe_allow_html=True,
    )

    band_powers_raw = extract_band_powers(f_raw, psd_raw)
    band_names = list(band_powers_raw.keys())
    band_vals = [np.mean(v) for v in band_powers_raw.values()]
    band_stds = [np.std(v) for v in band_powers_raw.values()]

    fig_band = go.Figure()
    fig_band.add_trace(go.Bar(
        x=band_names,
        y=band_vals,
        error_y=dict(type="data", array=band_stds, visible=True),
        marker=dict(
            color=band_vals,
            colorscale="Viridis",
            showscale=False,
        ),
        text=[f"{v:.1f}" for v in band_vals],
        textposition="outside",
        name="平均频带功率",
    ))
    fig_band.update_layout(
        xaxis_title="频带",
        yaxis_title="功率 (µV²)",
        height=300,
        margin=dict(l=50, r=20, t=20, b=60),
        title="原始数据 — 各频带功率",
        showlegend=False,
    )
    st.plotly_chart(fig_band, use_container_width=True)

# ============================================================================
# 右栏: ICA 分解结果
# ============================================================================

with col_right:
    st.markdown("### 🔵 ICA 独立分量分析")

    # --- 2a. ICA 分量波形图 ---
    st.markdown(
        """
        **独立分量时域波形 (Independent Components)**
        <br><small>FastICA 分解得到的独立分量。理想情况下，伪迹分量（如
        眼电、肌电）应集中在少数几个 IC 中，呈现与神经信号明显不同的
        时域特征（如大振幅尖峰 vs. 持续振荡）。</small>
        """,
        unsafe_allow_html=True,
    )

    fig_ica_ts = make_subplots(
        rows=min(n_components, 10), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    for i in range(min(n_components, 10)):
        fig_ica_ts.add_trace(
            go.Scattergl(
                x=time_vec,
                y=S_ica[i],
                mode="lines",
                line=dict(color=COLOR_ICA, width=0.6),
                name=f"IC {i + 1}",
                showlegend=False,
            ),
            row=i + 1, col=1,
        )
        fig_ica_ts.update_yaxes(
            title_text=f"IC{i + 1}", title_standoff=0, row=i + 1, col=1,
        )

    fig_ica_ts.update_xaxes(
        title_text="时间 (秒)", row=min(n_components, 10), col=1
    )
    fig_ica_ts.update_layout(
        height=400,
        margin=dict(l=50, r=20, t=20, b=40),
        title=f"ICA 独立分量 ({n_components} 个分量, 显示前 10)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_ica_ts, use_container_width=True)

    # --- 2b. ICA 分量 PSD ---
    st.markdown(
        """
        **各独立分量的功率谱密度**
        <br><small>对比各 IC 的 PSD。注意：
        (1) 神经信号分量通常在 Alpha 波段 (8–13 Hz) 有明显峰值；
        (2) 工频噪声分量在 50 Hz 处有极端窄带峰值；
        (3) 肌电分量在高频 (>30 Hz) 有抬升的宽带功率。</small>
        """,
        unsafe_allow_html=True,
    )

    fig_ica_psd = go.Figure()

    # 选取部分 IC 显示（避免过度拥挤，最多展示 8 条）
    n_ic_show = min(n_components, 8)
    for i in range(n_ic_show):
        psd_ic_db = 10 * np.log10(psd_ica[i] + 1e-12)
        fig_ica_psd.add_trace(go.Scatter(
            x=f_ica, y=psd_ic_db,
            mode="lines",
            line=dict(width=1.5),
            name=f"IC {i + 1}",
        ))

    # 频带标记
    fig_ica_psd.add_vrect(
        x0=8, x1=13,
        fillcolor=COLOR_ALPHA_BAND,
        layer="below", line_width=0,
        annotation_text="Alpha",
        annotation_position="top left",
    )
    if line_noise_amplitude > 0.5:
        fig_ica_psd.add_vline(
            x=50, line_dash="dash", line_color="gray",
            annotation_text="50 Hz", annotation_position="top",
            opacity=0.7,
        )

    fig_ica_psd.update_layout(
        xaxis_title="频率 (Hz)",
        yaxis_title="功率谱密度 (dB/Hz)",
        height=380,
        margin=dict(l=50, r=20, t=20, b=40),
        title=f"ICA 分量 PSD（前 {n_ic_show} 个分量）",
        hovermode="x",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig_ica_psd, use_container_width=True)

    # --- 2c. 混合矩阵热力图 ---
    st.markdown(
        """
        **混合矩阵 (Mixing Matrix) 热力图**
        <br><small>混合矩阵 A 的每一列描述一个独立分量在各 EEG 通道上的
        空间投影权重（即头皮地形图的向量化表示）。识别伪迹分量的关键线索：
        眼电分量在前额通道 (Fp1, Fp2) 权重极高；肌电分量集中在颞叶通道
        (T3–T6)；工频分量通常在周边通道均匀分布。</small>
        """,
        unsafe_allow_html=True,
    )

    fig_mix = go.Figure(data=go.Heatmap(
        z=A_mix,
        x=[f"IC{i + 1}" for i in range(n_components)],
        y=DEFAULT_CHANNELS,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(A_mix, 2),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="权重"),
        hovertemplate=(
            "通道: %{y}<br>分量: %{x}<br>权重: %{z:.3f}<extra></extra>"
        ),
    ))

    fig_mix.update_layout(
        xaxis_title="独立分量 (IC)",
        yaxis_title="EEG 通道",
        height=380,
        margin=dict(l=50, r=20, t=20, b=60),
        title="ICA 混合矩阵 (Mixing Matrix A)",
    )
    st.plotly_chart(fig_mix, use_container_width=True)

# ============================================================================
# 底部：分析摘要
# ============================================================================

st.markdown("---")
st.markdown("### 📋 分析摘要")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    # 原始数据 SNR
    from src.signal_processor import estimate_snr
    snr_vals = estimate_snr(data, sfreq=sfreq)
    st.metric(
        "平均 SNR (原始数据)",
        f"{np.mean(snr_vals):.1f} dB",
        help="信噪比 = 10 log₁₀(P₁–₄₅ Hz / P₄₅–₈₀ Hz)。值越高表示信号质量越好。"
    )

with col_s2:
    # Alpha 波段功率
    alpha_power_raw = band_powers_raw.get("Alpha (8–13 Hz)", np.array([0]))
    st.metric(
        "Alpha 波段平均功率",
        f"{np.mean(alpha_power_raw):.1f} µV²",
        help="8–13 Hz 频带的总功率。Alpha 是静息态 EEG 的主要节律。"
    )

with col_s3:
    # 50 Hz 峰值检测
    freq_50_idx = np.argmin(np.abs(f_raw - 50.0))
    psd_50hz_mean = psd_db_mean[freq_50_idx]
    # 50 Hz 附近（45-55 Hz）的平均功率作为基线
    mask_baseline = (f_raw >= 30) & (f_raw <= 70)
    psd_baseline = np.mean(psd_db_mean[mask_baseline])
    peak_ratio = psd_50hz_mean - psd_baseline
    st.metric(
        "50 Hz 工频峰高",
        f"{max(0, peak_ratio):.1f} dB",
        help="50 Hz 处相对于基线 (30-70 Hz) 的功率升高。较高的值表示工频污染严重。"
    )

with col_s4:
    # IC 数量
    st.metric(
        "ICA 分离分量数",
        f"{n_components}",
        help="当前 ICA 提取的独立分量数量。"
    )

st.markdown(
    """
    ---

    ### 🧪 科研意义总结

    **ICA 在 EEG 预处理中的核心价值（向陈志明教授汇报要点）：**

    1. **盲源分离**：ICA 无需任何先验模板即能将伪迹（眼电、肌电、工频）
       与神经信号分离，是真正的数据驱动方法。

    2. **定量评估**：通过对比去噪前后的 PSD，可以精确量化：
       - Alpha 波段信噪比提升
       - 50 Hz 工频抑制程度
       - 高频 (>30 Hz) 宽带噪声残留水平

    3. **脑连接分析保障**：功能连接指标（如相干性、PLV、wPLI）对伪迹
       高度敏感。尤其需要注意：
       - 未去除的眼电伪迹 → 额叶虚假 Delta/Theta 超连接
       - 未去除的肌电伪迹 → 宽带 Beta/Gamma 虚假连接
       - 残留工频噪声 → 50 Hz 窄带相干假象

    4. **可复现性**：本工具通过固定随机种子、参数化噪声模型，确保实验
       结果可被其他研究者完全复现，符合开放科学 (Open Science) 标准。
    """
)
