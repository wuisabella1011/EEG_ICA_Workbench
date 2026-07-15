<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-≥3.10-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/status-research--grade-purple.svg" alt="Status">
</p>

<h1 align="center">🧠 EEG ICA Workbench</h1>

<h3 align="center">科研级脑电独立成分分析与去噪工作台</h3>

<p align="center">
  <em>为脑连接分析提供高质量、可复现、可交互的数据预处理流水线</em>
</p>

---

## 📋 项目概述

**EEG ICA Workbench** 是一个面向神经科学研究的交互式 EEG 预处理工具，基于 **独立成分分析（Independent Component Analysis, ICA）** 实现盲源分离，将眼电、肌电、工频噪声等伪迹从真实神经活动中分离出来。

本工作台专为 **脑连接分析（Brain Connectivity Analysis）** 的上游数据质量保障而设计——未经 ICA 去噪的 EEG 数据会导致虚假的功能连接，严重污染相干性（Coherence）、相位锁值（PLV）、加权相位滞后指数（wPLI）和格兰杰因果（Granger Causality）等指标。

### 🎯 核心功能

| 模块 | 功能 | 科研价值 |
|------|------|----------|
| 📡 **信号模拟** | 生成含多类噪声的 19 通道 EEG 数据 | 可控实验条件，验证算法有效性 |
| 🔬 **ICA 分解** | sklearn FastICA 盲源分离 | 将伪迹与神经信号分离为独立分量 |
| 📊 **PSD 分析** | Welch 平均周期图法功率谱估计 | 量化去噪前后的频谱变化 |
| 📈 **频带分析** | Delta/Theta/Alpha/Beta/Gamma 功率提取 | 评估各频带信噪比改善 |

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.10
- pip（或 conda）

### 安装

```bash
git clone https://github.com/wuisabella1011/EEG_ICA_Workbench.git
cd EEG_ICA_Workbench
pip install -r requirements.txt
```

### 启动

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，左侧边栏选择 **📈 Signal Processing** 进入交互分析界面。

### 依赖项

| 包 | 版本 | 用途 |
|---|---|---|
| `streamlit` | ≥ 1.28.0 | Web 应用框架 |
| `numpy` | ≥ 1.24.0 | 数值计算 |
| `scipy` | ≥ 1.10.0 | 信号处理（FIR 滤波、PSD） |
| `scikit-learn` | ≥ 1.3.0 | FastICA 实现 |
| `plotly` | ≥ 5.15.0 | 交互式可视化 |

---

## 📁 项目结构

```
EEG_ICA_Workbench/
├── app.py                              # Streamlit 主入口
├── requirements.txt                    # Python 依赖
├── README.md                           # 项目文档
├── src/
│   ├── __init__.py                     # 包初始化
│   └── signal_processor.py             # 核心算法模块
│       ├── generate_mock_eeg()         #   模拟 EEG 数据生成
│       ├── run_ica()                   #   FastICA 分解
│       ├── calculate_psd()             #   Welch PSD 估计
│       ├── extract_band_powers()       #   频带功率提取
│       └── estimate_snr()              #   信噪比估计
├── pages/
│   └── 1_📈_Signal_Processing.py       # 信号处理交互界面
└── data/                               # 样例数据目录
```

---

## 🔬 科学原理

### ICA 在 EEG 去噪中的应用

EEG 头皮记录的信号是多种源信号经过容积传导（volume conduction）后的线性混合：

$$\mathbf{x}(t) = \mathbf{A} \cdot \mathbf{s}(t)$$

其中 $\mathbf{x}(t)$ 为头皮观测信号，$\mathbf{A}$ 为混合矩阵，$\mathbf{s}(t)$ 为未知的源信号。

FastICA 通过最大化非高斯性（等价于最小化互信息），找到一个解混矩阵 $\mathbf{W}$ 使得：

$$\mathbf{s}(t) \approx \mathbf{W} \cdot \mathbf{x}(t), \quad \mathbf{W} \approx \mathbf{A}^{-1}$$

从而将伪迹成分（眨眼、肌电、工频噪声）与神经振荡分离到不同的独立分量中。

### 模拟噪声模型

本工具生成的模拟 EEG 包含 **五层** 噪声成分：

$$
\begin{aligned}
\text{EEG}(t) &= \underbrace{\sum_{b \in \{\delta,\theta,\alpha,\beta\}} \text{Osc}_b(t)}_{\text{背景神经振荡}} \\
&+ \underbrace{\text{EOG}(t)}_{\text{眼电伪迹}} + \underbrace{\text{EMG}(t)}_{\text{肌电伪迹}} \\
&+ \underbrace{A_{\text{line}} \cdot \sin(2\pi \cdot 50t)}_{\text{工频噪声}} + \underbrace{\mathcal{N}(0, \sigma^2)}_{\text{传感器底噪}}
\end{aligned}
$$

其中：
- **眼电伪迹** 使用 Rice 脉冲（快速上升 → 缓慢衰减）模拟眨眼形态
- **肌电伪迹** 使用高频白噪声经汉宁窗包络产生
- **背景振荡** 按 10-20 系统中的解剖区域分布（Alpha 枕叶主导，Beta 中央区主导，Theta 额叶主导）

### 伪迹对功能连接的影响

| 伪迹类型 | 频带污染 | 虚假连接模式 |
|----------|----------|-------------|
| 眼电（EOG） | Delta/Theta (0.5–8 Hz) | 额叶区域超连接 |
| 肌电（EMG） | Beta/Gamma (20–80 Hz) | 宽频高频假阳性 |
| 工频噪声 | 50/60 Hz 窄带 | 窄带相干假象 |

---

## 🎛️ 交互功能

通过侧边栏可动态调节以下参数并实时观察结果变化：

### 数据生成参数
- 采样率（100–1000 Hz）
- 数据时长（2–30 秒）
- 工频噪声幅度
- 眼电伪迹幅度
- 肌电伪迹幅度
- 随机种子（保证可复现性）

### ICA 参数
- 分量数（5–19）
- 算法类型（parallel / deflation）
- 非线性函数（logcosh / exp / cube）

### 可视化输出
- **左侧面板**：原始 EEG 时域波形 × 10 通道 + 平均 PSD + 频带功率柱状图
- **右侧面板**：ICA 独立分量波形 + 各 IC 的 PSD + 混合矩阵热力图
- **底部摘要**：平均 SNR、Alpha 波段功率、50 Hz 峰高、ICA 分离分量数

所有图表使用 **Plotly** 渲染，支持缩放、拖拽、悬停提示和局部放大。

---

## 📊 算法参考

| 方法 | 实现 | 文献 |
|------|------|------|
| FastICA | `sklearn.decomposition.FastICA` | Hyvärinen, A. & Oja, E. (2000). Independent component analysis: algorithms and applications. *Neural Networks*, 13(4-5), 411–430. |
| Welch PSD | `scipy.signal.welch` | Welch, P.D. (1967). The use of Fast Fourier Transform for the estimation of power spectra. *IEEE Trans. Audio Electroacoustics*, 15(2), 70–73. |
| FIR 滤波 | `scipy.signal.firwin` | Blackman 窗 FIR 带通滤波器 |

---

## 🧪 设计原则

1. **科研级可复现性** — 所有随机过程固定种子，参数显式暴露
2. **模块化架构** — 核心算法与前端展示解耦，纯函数设计便于单元测试
3. **交互式可视化** — 基于 Plotly 的实时参数探索
4. **透明化流水线** — 每一步都提供定量指标和可视化验证

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ for the neuroscience community · EEG ICA Workbench v1.0.0</sub>
</p>
