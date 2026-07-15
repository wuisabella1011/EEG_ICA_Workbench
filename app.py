"""
app.py — EEG ICA Workbench 主入口
===================================
项目目标：
  为脑连接分析（Brain Connectivity Analysis）提供高质量的数据预处理流水线。

设计理念：
  - 科研级可复现性：所有随机过程固定种子
  - 交互式可视化：基于 Plotly 的缩放/拖拽/Pan 支持
  - 模块化架构：核心算法与前端展示解耦，便于单元测试和论文复现

适用场景：
  - EEG 伪迹去除（眼电、肌电、工频噪声）
  - ICA 分量特征分析与筛选
  - 功率谱密度 (PSD) 的定量对比（去噪前后）

向陈志明教授汇报：
  本工具聚焦于脑连接分析中至关重要却常被忽视的数据预处理环节。
  高质量的 ICA 去噪能够显著降低基于相干性和相位的连接指标的假阳性率，
  从而提升格兰杰因果和 DCM 等高级分析的可信度。

启动方式：
  streamlit run app.py

作者: EEG ICA Workbench 项目组
版本: 1.0.0
"""

import streamlit as st

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="EEG ICA Workbench — 脑电独立成分分析工作台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 主页面内容
# ============================================================================

def main():
    # --- 标题区域 ---
    st.title("🧠 EEG ICA Workbench")
    st.subheader("脑电独立成分分析与去噪工作台")

    st.markdown("---")

    # --- 简介 ---
    col_intro_left, col_intro_right = st.columns([2, 1])

    with col_intro_left:
        st.markdown(
            """
            ### 项目设计目标

            本工作台专为 **脑连接分析 (Brain Connectivity Analysis)** 的
            **数据预处理** 而设计，提供从原始 EEG 信号到干净神经活动信号的
            完整流水线。

            **核心功能模块：**

            | 模块 | 功能 | 科研价值 |
            |------|------|----------|
            | 📡 **信号模拟** | 生成含多类噪声的 EEG 数据 | 可控的实验条件，验证算法有效性 |
            | 🔬 **ICA 分解** | FastICA 盲源分离 | 将伪迹与神经信号分离为独立分量 |
            | 📊 **PSD 分析** | Welch 法功率谱估计 | 量化去噪前后的频谱变化 |
            | 📈 **频带分析** | Delta/Theta/Alpha/Beta/Gamma | 评估各频带信噪比改善 |

            **为什么 ICA 去噪对脑网络分析至关重要：**

            基于 EEG 的功能连接指标（相干性、PLV、PLI、格兰杰因果等）极易受到
            伪迹污染。研究表明，未经 ICA 去噪的 EEG 数据中：
            - 眼电伪迹会导致额叶区域虚假的 **Delta/Theta 频段超连接**
            - 肌电伪迹会在 **Beta/Gamma 频段** 引入宽频噪声
            - 工频噪声 (50/60 Hz) 会产生高频窄带相干假象

            本工具通过透明、可交互的 ICA 流水线，使研究者能够在去噪的每一步
            进行可视化验证，确保只有真正的神经活动信号进入后续的连接分析。
            """
        )

    with col_intro_right:
        st.info(
            """
            ### 🚀 快速开始

            1. 在侧边栏选择 **📈 Signal Processing**
            2. 调整 ICA 参数（分量数、算法类型）
            3. 观察原始数据与去噪后数据的对比
            4. 检查 PSD 图以评估噪声去除效果

            ---

            ### 📁 项目结构

            ```
            EEG_ICA_Workbench/
            ├── app.py                 ← 主入口
            ├── src/
            │   ├── __init__.py
            │   └── signal_processor.py  ← 核心算法
            ├── pages/
            │   └── 1_📈_Signal_Processing.py ← 信号处理界面
            └── data/                  ← 样例数据
            ```
            """
        )

    st.markdown("---")

    # --- 技术说明 ---
    st.markdown(
        """
        ### 🔧 技术栈与学术引用

        | 技术 | 版本/方法 | 学术引用 |
        |------|-----------|----------|
        | FastICA | sklearn.decomposition | Hyvärinen & Oja (2000). *Independent component analysis: algorithms and applications*. Neural Networks. |
        | PSD (Welch) | scipy.signal.welch | Welch, P.D. (1967). *The use of Fast Fourier Transform for the estimation of power spectra*. IEEE Trans. Audio Electroacoust. |
        | 可视化 | Plotly | 交互式数据探索平台 |

        ---

        <div style="text-align: center; color: #888; font-size: 0.85rem;">
        EEG ICA Workbench v1.0.0 · 科研级脑电预处理平台 <br>
        为陈志明教授课题组 — 脑连接分析数据质量保障工具
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
