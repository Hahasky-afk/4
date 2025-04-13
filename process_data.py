#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
處理和分析從FRED獲取的美國宏觀經濟數據
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 設置數據路徑
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_data():
    """
    載入所有指標數據
    
    返回:
        pandas.DataFrame: 包含所有指標數據的DataFrame
    """
    try:
        data_path = os.path.join(DATA_DIR, "all_indicators.csv")
        df = pd.read_csv(data_path, index_col='date', parse_dates=True)
        logger.info(f"成功載入數據，共 {len(df)} 行")
        return df
    except Exception as e:
        logger.error(f"載入數據時出錯: {str(e)}")
        return pd.DataFrame()

def clean_data(df):
    """
    清理數據，處理缺失值和異常值
    
    參數:
        df (pandas.DataFrame): 原始數據
        
    返回:
        pandas.DataFrame: 清理後的數據
    """
    if df.empty:
        logger.warning("輸入數據為空，無法進行清理")
        return df
    
    # 記錄原始數據的缺失值情況
    missing_values = df.isnull().sum()
    logger.info(f"原始數據缺失值情況:\n{missing_values}")
    
    # 複製數據以避免修改原始數據
    cleaned_df = df.copy()
    
    # 處理GDP增長率的缺失值（季度數據）
    if 'GDP增長率' in cleaned_df.columns:
        # 對於GDP增長率，使用前向填充，因為它是季度數據
        cleaned_df['GDP增長率'] = cleaned_df['GDP增長率'].fillna(method='ffill')
        logger.info("使用前向填充處理GDP增長率的缺失值")
    
    # 處理通脹率的缺失值
    if '通脹率' in cleaned_df.columns:
        # 檢查通脹率列是否全為空
        if cleaned_df['通脹率'].isnull().all():
            logger.warning("通脹率列全為空，需要重新計算")
            # 嘗試從CPI數據重新計算通脹率
            try:
                cpi_path = os.path.join(DATA_DIR, "通脹率.csv")
                cpi_df = pd.read_csv(cpi_path, index_col='date', parse_dates=True)
                # 確保數據不為空
                if not cpi_df.empty:
                    # 將通脹率數據合併到主數據框中
                    cleaned_df['通脹率'] = cpi_df.iloc[:, 0]
                    logger.info("成功從單獨的CSV文件中獲取通脹率數據")
            except Exception as e:
                logger.error(f"重新計算通脹率時出錯: {str(e)}")
    
    # 對其他列使用適當的填充方法
    for column in cleaned_df.columns:
        if column not in ['GDP增長率', '通脹率']:
            # 對於月度數據，使用線性插值填充缺失值
            cleaned_df[column] = cleaned_df[column].interpolate(method='linear')
            logger.info(f"使用線性插值處理 {column} 的缺失值")
    
    # 處理剩餘的缺失值（如果有）
    cleaned_df = cleaned_df.fillna(method='bfill')
    
    # 記錄清理後的缺失值情況
    missing_values_after = cleaned_df.isnull().sum()
    logger.info(f"清理後數據缺失值情況:\n{missing_values_after}")
    
    return cleaned_df

def normalize_data(df):
    """
    標準化數據，使不同指標可比較
    
    參數:
        df (pandas.DataFrame): 清理後的數據
        
    返回:
        pandas.DataFrame: 標準化後的數據
    """
    if df.empty:
        logger.warning("輸入數據為空，無法進行標準化")
        return df
    
    # 複製數據以避免修改原始數據
    normalized_df = df.copy()
    
    # 創建一個新的DataFrame來存儲標準化後的數據
    normalized_values = pd.DataFrame(index=df.index)
    
    # 對每個指標進行Min-Max標準化
    for column in normalized_df.columns:
        min_val = normalized_df[column].min()
        max_val = normalized_df[column].max()
        
        # 避免除以零
        if max_val > min_val:
            normalized_values[f"{column}_normalized"] = (normalized_df[column] - min_val) / (max_val - min_val)
            logger.info(f"成功標準化 {column} 數據")
        else:
            normalized_values[f"{column}_normalized"] = 0
            logger.warning(f"{column} 數據無法標準化，因為最大值等於最小值")
    
    # 將原始數據和標準化數據合併
    result_df = pd.concat([normalized_df, normalized_values], axis=1)
    
    return result_df

def calculate_correlations(df):
    """
    計算指標之間的相關性
    
    參數:
        df (pandas.DataFrame): 清理後的數據
        
    返回:
        pandas.DataFrame: 相關性矩陣
    """
    if df.empty:
        logger.warning("輸入數據為空，無法計算相關性")
        return pd.DataFrame()
    
    # 只使用原始指標列（非標準化列）計算相關性
    original_columns = [col for col in df.columns if not col.endswith('_normalized')]
    correlation_matrix = df[original_columns].corr()
    
    logger.info("成功計算指標之間的相關性")
    return correlation_matrix

def resample_data(df, freq='M'):
    """
    重採樣數據到指定頻率
    
    參數:
        df (pandas.DataFrame): 清理後的數據
        freq (str): 目標頻率，如'M'表示月度，'Q'表示季度
        
    返回:
        pandas.DataFrame: 重採樣後的數據
    """
    if df.empty:
        logger.warning("輸入數據為空，無法進行重採樣")
        return df
    
    # 確保索引是日期時間類型
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("數據索引不是日期時間類型，嘗試轉換")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"轉換索引為日期時間類型時出錯: {str(e)}")
            return df
    
    # 重採樣數據
    # 對於上採樣（如從季度到月度），使用前向填充
    # 對於下採樣（如從月度到季度），使用平均值
    resampled_df = df.resample(freq).mean()
    
    logger.info(f"成功將數據重採樣為 {freq} 頻率")
    return resampled_df

def prepare_data_for_visualization():
    """
    準備用於可視化的數據
    
    返回:
        pandas.DataFrame: 處理後的數據，適合用於可視化
    """
    # 載入數據
    df = load_data()
    if df.empty:
        logger.error("無法載入數據，退出準備過程")
        return pd.DataFrame()
    
    # 清理數據
    cleaned_df = clean_data(df)
    
    # 標準化數據
    normalized_df = normalize_data(cleaned_df)
    
    # 計算相關性
    correlation_matrix = calculate_correlations(cleaned_df)
    
    # 保存處理後的數據
    processed_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
    normalized_df.to_csv(processed_data_path)
    logger.info(f"已將處理後的數據保存到 {processed_data_path}")
    
    # 保存相關性矩陣
    correlation_path = os.path.join(PROCESSED_DATA_DIR, "correlation_matrix.csv")
    correlation_matrix.to_csv(correlation_path)
    logger.info(f"已將相關性矩陣保存到 {correlation_path}")
    
    # 為不同時間範圍準備數據
    # 最近一年的數據
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
    one_year_data = normalized_df[normalized_df.index >= one_year_ago]
    one_year_path = os.path.join(PROCESSED_DATA_DIR, "one_year_data.csv")
    one_year_data.to_csv(one_year_path)
    logger.info(f"已將最近一年的數據保存到 {one_year_path}")
    
    # 最近三年的數據
    three_years_ago = pd.Timestamp.now() - pd.DateOffset(years=3)
    three_years_data = normalized_df[normalized_df.index >= three_years_ago]
    three_years_path = os.path.join(PROCESSED_DATA_DIR, "three_years_data.csv")
    three_years_data.to_csv(three_years_path)
    logger.info(f"已將最近三年的數據保存到 {three_years_path}")
    
    return normalized_df

def main():
    """主函數"""
    logger.info("開始處理和分析經濟數據")
    
    # 準備用於可視化的數據
    processed_data = prepare_data_for_visualization()
    
    if not processed_data.empty:
        logger.info("數據處理和分析完成")
    else:
        logger.error("數據處理和分析失敗")

if __name__ == "__main__":
    main()
