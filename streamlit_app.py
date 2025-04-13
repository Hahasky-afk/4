#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit部署配置文件
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from fredapi import Fred
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設置頁面配置
st.set_page_config(
    page_title="美股宏觀儀表盤",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主要指標列表
MAIN_INDICATORS = ["GDP增長率", "失業率", "通脹率", "利率", "消費者信心指數"]
# 次要指標列表
SECONDARY_INDICATORS = ["工業生產指數", "零售銷售", "住房開工數", "貿易差額", "企業信心指數"]
# 所有指標列表
ALL_INDICATORS = MAIN_INDICATORS + SECONDARY_INDICATORS

# 指標顏色映射
INDICATOR_COLORS = {
    "GDP增長率": "#1f77b4",
    "失業率": "#ff7f0e",
    "通脹率": "#2ca02c",
    "利率": "#d62728",
    "消費者信心指數": "#9467bd",
    "工業生產指數": "#8c564b",
    "零售銷售": "#e377c2",
    "住房開工數": "#7f7f7f",
    "貿易差額": "#bcbd22",
    "企業信心指數": "#17becf"
}

# 指標說明
INDICATOR_DESCRIPTIONS = {
    "GDP增長率": "國內生產總值增長率，反映經濟整體增長情況，季度數據（年化）",
    "失業率": "失業人口佔勞動力的百分比，反映就業市場健康狀況",
    "通脹率": "消費者物價指數(CPI)同比變化，反映物價水平變化",
    "利率": "聯邦基金利率，美聯儲的主要政策工具，影響借貸成本",
    "消費者信心指數": "密歇根大學消費者信心指數，反映消費者對經濟的信心",
    "工業生產指數": "衡量製造業、採礦業和公用事業的產出水平",
    "零售銷售": "零售商品銷售總額，反映消費者支出情況",
    "住房開工數": "新建住房開工數量，反映房地產市場活動",
    "貿易差額": "出口減進口的差額，負值表示貿易逆差",
    "企業信心指數": "反映企業對經濟前景的信心水平"
}

# 經濟指標代碼
INDICATORS = {
    "GDP增長率": "A191RL1Q225SBEA",  # 實質GDP季度增長率（年化）
    "失業率": "UNRATE",              # 失業率
    "通脹率": "CPIAUCSL",            # 消費者物價指數（通脹率需要計算）
    "利率": "FEDFUNDS",              # 聯邦基金利率
    "消費者信心指數": "UMCSENT",      # 密歇根大學消費者信心指數
    "工業生產指數": "INDPRO",         # 工業生產指數
    "零售銷售": "RSAFS",              # 零售銷售
    "住房開工數": "HOUST",            # 住房開工數
    "貿易差額": "BOPGSTB",            # 貿易差額
    "企業信心指數": "BSCICP03USM665S"  # 企業信心指數
}

# 在Streamlit Cloud中使用環境變量獲取API密鑰
def get_api_key():
    """從環境變量或Streamlit Secrets獲取FRED API密鑰"""
    try:
        # 嘗試從Streamlit Secrets獲取
        return st.secrets["FRED_API_KEY"]
    except:
        # 如果不在Secrets中，嘗試從環境變量獲取
        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            return api_key
        else:
            st.error("未找到FRED API密鑰。請在Streamlit Cloud中設置FRED_API_KEY密鑰，或在本地運行時設置環境變量。")
            return None

def fetch_fred_data(api_key):
    """從FRED獲取數據並處理"""
    if not api_key:
        return None
    
    try:
        fred = Fred(api_key=api_key)
        
        # 計算開始日期（當前日期減去5年）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        st.info(f"正在從FRED獲取數據，這可能需要一些時間...")
        
        # 獲取所有指標數據
        all_data = {}
        progress_bar = st.progress(0)
        
        for i, (indicator_name, series_id) in enumerate(INDICATORS.items()):
            with st.spinner(f"獲取 {indicator_name} 數據..."):
                # 獲取原始數據
                if indicator_name == "通脹率":
                    # 對於通脹率，我們需要獲取CPI數據然後計算
                    cpi_data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                    if not cpi_data.empty:
                        # 計算同比百分比變化（年度通脹率）
                        inflation_data = cpi_data.pct_change(periods=12) * 100
                        all_data[indicator_name] = pd.DataFrame(inflation_data, columns=[indicator_name])
                else:
                    # 對於其他指標，直接獲取數據
                    kwargs = {}
                    if indicator_name == "GDP增長率":
                        kwargs['frequency'] = 'q'
                    else:
                        kwargs['frequency'] = 'm'
                        
                    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date, **kwargs)
                    if len(data) > 0:
                        all_data[indicator_name] = pd.DataFrame(data, columns=[indicator_name])
            
            # 更新進度條
            progress_bar.progress((i + 1) / len(INDICATORS))
        
        # 合併所有數據
        if all_data:
            # 找出所有數據框的索引
            all_indices = sorted(set().union(*[df.index for df in all_data.values()]))
            
            # 創建一個空的數據框，包含所有日期
            merged_df = pd.DataFrame(index=all_indices)
            
            # 將每個指標的數據合併到主數據框中
            for indicator_name, df in all_data.items():
                merged_df = merged_df.join(df)
            
            # 清理和處理數據
            return process_data(merged_df)
        else:
            st.error("無法獲取任何數據，請檢查API密鑰和網絡連接。")
            return None
    except Exception as e:
        st.error(f"獲取FRED數據時出錯: {str(e)}")
        return None

def process_data(df):
    """處理和清理數據"""
    if df.empty:
        return df
    
    # 複製數據以避免修改原始數據
    cleaned_df = df.copy()
    
    # 處理GDP增長率的缺失值（季度數據）
    if 'GDP增長率' in cleaned_df.columns:
        cleaned_df['GDP增長率'] = cleaned_df['GDP增長率'].ffill()
    
    # 對其他列使用適當的填充方法
    for column in cleaned_df.columns:
        if column != 'GDP增長率':
            # 對於月度數據，使用線性插值填充缺失值
            cleaned_df[column] = cleaned_df[column].interpolate(method='linear')
    
    # 處理剩餘的缺失值（如果有）
    cleaned_df = cleaned_df.bfill()
    
    # 標準化數據
    normalized_df = cleaned_df.copy()
    normalized_values = pd.DataFrame(index=df.index)
    
    for column in normalized_df.columns:
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()
        
        # 避免除以零
        if max_val > min_val:
            normalized_values[f"{column}_normalized"] = (normalized_df[column] - min_val) / (max_val - min_val)
        else:
            normalized_values[f"{column}_normalized"] = 0
    
    # 將原始數據和標準化數據合併
    result_df = pd.concat([normalized_df, normalized_values], axis=1)
    
    # 計算相關性矩陣
    original_columns = [col for col in result_df.columns if not col.endswith('_normalized')]
    correlation_matrix = result_df[original_columns].corr()
    
    return result_df, correlation_matrix

def filter_data_by_date_range(df, start_date, end_date):
    """按日期範圍過濾數據"""
    if df.empty:
        return df
    
    # 確保索引是日期時間類型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 過濾數據
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    return filtered_df

def plot_indicator_trends(df, indicators, start_date, end_date, use_normalized=False):
    """繪製指標趨勢圖"""
    if df.empty or not indicators:
        return go.Figure()
    
    # 過濾數據
    filtered_df = filter_data_by_date_range(df, start_date, end_date)
    if filtered_df.empty:
        return go.Figure()
    
    # 創建子圖
    fig = make_subplots(rows=len(indicators), cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=[f"{ind} ({INDICATOR_DESCRIPTIONS[ind]})" for ind in indicators])
    
    # 為每個指標添加折線圖
    for i, indicator in enumerate(indicators):
        col_name = f"{indicator}_normalized" if use_normalized else indicator
        
        if col_name in filtered_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df[col_name],
                    mode='lines',
                    name=indicator,
                    line=dict(color=INDICATOR_COLORS[indicator], width=2),
                    showlegend=False
                ),
                row=i+1, col=1
            )
    
    # 更新布局
    fig.update_layout(
        height=250 * len(indicators),
        margin=dict(l=50, r=50, t=50, b=50),
        title_text="美國宏觀經濟指標趨勢",
        title_font=dict(size=24),
        template="plotly_white"
    )
    
    # 更新Y軸標題
    for i, indicator in enumerate(indicators):
        y_title = "標準化值 (0-1)" if use_normalized else "原始值"
        fig.update_yaxes(title_text=y_title, row=i+1, col=1)
    
    # 更新X軸標題
    fig.update_xaxes(title_text="日期", row=len(indicators), col=1)
    
    return fig

def plot_correlation_heatmap(corr_df):
    """繪製相關性熱力圖"""
    if corr_df.empty:
        return go.Figure()
    
    # 創建熱力圖
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    # 更新布局
    fig.update_layout(
        title_text="指標相關性熱力圖",
        title_font=dict(size=20),
        height=600,
        width=800,
        margin=dict(l=50, r=50, t=80, b=50),
        coloraxis_colorbar=dict(
            title="相關係數",
            titleside="right",
            ticks="outside"
        )
    )
    
    return fig

def plot_comparison_chart(df, indicators, start_date, end_date):
    """繪製指標對比圖（使用標準化數據）"""
    if df.empty or not indicators:
        return go.Figure()
    
    # 過濾數據
    filtered_df = filter_data_by_date_range(df, start_date, end_date)
    if filtered_df.empty:
        return go.Figure()
    
    # 創建圖表
    fig = go.Figure()
    
    # 為每個指標添加折線
    for indicator in indicators:
        norm_col = f"{indicator}_normalized"
        if norm_col in filtered_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df[norm_col],
                    mode='lines',
                    name=indicator,
                    line=dict(color=INDICATOR_COLORS[indicator], width=2)
                )
            )
    
    # 更新布局
    fig.update_layout(
        title_text="指標對比圖 (標準化值)",
        title_font=dict(size=20),
        xaxis_title="日期",
        yaxis_title="標準化值 (0-1)",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    return fig

def main():
    """主函數"""
    # 頁面標題
    st.title("美股宏觀儀表盤")
    st.markdown("---")
    
    # 獲取API密鑰
    api_key = get_api_key()
    
    # 如果在Streamlit Cloud上運行，直接從FRED獲取數據
    if 'data' not in st.session_state:
        with st.spinner("正在獲取和處理數據..."):
            result = fetch_fred_data(api_key)
            if result:
                st.session_state.data, st.session_state.corr_matrix = result
                st.success("數據獲取和處理完成！")
            else:
                st.error("無法獲取或處理數據。")
                return
    
    df = st.session_state.data
    corr_df = st.session_state.corr_matrix
    
    if df.empty:
        st.error("無法載入數據")
        return
    
    # 側邊欄 - 時間範圍選擇
    st.sidebar.header("時間範圍選擇")
    
    # 預設時間範圍選項
    date_ranges = {
        "最近一年": (datetime.now() - timedelta(days=365), datetime.now()),
        "最近三年": (datetime.now() - timedelta(days=3*365), datetime.now()),
        "最近五年": (datetime.now() - timedelta(days=5*365), datetime.now()),
        "自定義": None
    }
    
    # 時間範圍選擇
    selected_range = st.sidebar.selectbox(
        "選擇時間範圍",
        options=list(date_ranges.keys()),
        index=2  # 預設選擇"最近五年"
    )
    
    # 如果選擇自定義，顯示日期選擇器
    if selected_range == "自定義":
        min_date = df.index.min().to_pydatetime()
        max_date = df.index.max().to_pydatetime()
        
        start_date = st.sidebar.date_input(
            "開始日期",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "結束日期",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # 轉換為datetime
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    else:
        # 使用預設範圍
        start_date, end_date = date_ranges[selected_range]
    
    # 側邊欄 - 數據顯示選項
    st.sidebar.header("數據顯示選項")
    
    # 選擇是否使用標準化數據
    use_normalized = st.sidebar.checkbox("使用標準化數據", value=False)
    
    # 選擇要顯示的指標
    st.sidebar.subheader("選擇要顯示的指標")
    
    # 主要指標選擇
    selected_main_indicators = []
    for indicator in MAIN_INDICATORS:
        if st.sidebar.checkbox(indicator, value=True, key=f"main_{indicator}"):
            selected_main_indicators.append(indicator)
    
    # 次要指標選擇
    show_secondary = st.sidebar.checkbox("顯示次要指標", value=False)
    selected_secondary_indicators = []
    
    if show_secondary:
        for indicator in SECONDARY_INDICATORS:
            if st.sidebar.checkbox(indicator, value=False, key=f"secondary_{indicator}"):
                selected_secondary_indicators.append(indicator)
    
    # 合併選中的指標
    selected_indicators = selected_main_indicators + selected_secondary_indicators
    
    # 主要內容區域
    st.header(f"經濟指標趨勢 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})")
    
    # 繪製趨勢圖
    if selected_indicators:
        trend_fig = plot_indicator_trends(df, selected_indicators, start_date, end_date, use_normalized)
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.warning("請至少選擇一個指標以顯示趨勢圖")
    
    # 指標對比分析
    st.header("指標對比分析")
    
    # 選擇要對比的指標
    st.subheader("選擇要對比的指標")
    cols = st.columns(5)
    compare_indicators = []
    
    for i, indicator in enumerate(ALL_INDICATORS):
        col_idx = i % 5
        if cols[col_idx].checkbox(indicator, value=indicator in MAIN_INDICATORS[:3], key=f"compare_{indicator}"):
            compare_indicators.append(indicator)
    
    # 繪製對比圖
    if compare_indicators:
        comparison_fig = plot_comparison_chart(df, compare_indicators, start_date, end_date)
        st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        st.warning("請至少選擇一個指標以顯示對比圖")
    
    # 相關性分析
    st.header("指標相關性分析")
    
    if not corr_df.empty:
        heatmap_fig = plot_correlation_heatmap(corr_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # 相關性解釋
        st.subheader("相關性解釋")
        st.markdown("""
        - **正相關 (0到1)**: 兩個指標同向變動，數值越接近1表示相關性越強
        - **負相關 (-1到0)**: 兩個指標反向變動，數值越接近-1表示相關性越強
        - **無相關 (接近0)**: 兩個指標變動之間幾乎沒有關係
        """)
    else:
        st.warning("無法載入相關性數據")
    
    # 頁腳
    st.markdown("---")
    st.markdown("**美股宏觀儀表盤** | 數據來源: FRED (Federal Reserve Economic Data)")

if __name__ == "__main__":
    main()
