#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
從FRED獲取美國宏觀經濟數據
"""

import os
import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FRED API密鑰
FRED_API_KEY = "96472d8ae6dea88245d1c85a29e4e2cd"

# 設置數據保存路徑
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

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

def get_fred_data(api_key, series_id, start_date, end_date=None, frequency=None, transform=None):
    """
    從FRED獲取特定系列的數據
    
    參數:
        api_key (str): FRED API密鑰
        series_id (str): 數據系列ID
        start_date (str): 開始日期，格式為'YYYY-MM-DD'
        end_date (str, optional): 結束日期，格式為'YYYY-MM-DD'，默認為當前日期
        frequency (str, optional): 數據頻率，如'm'表示月度，'q'表示季度
        transform (str, optional): 數據轉換方式，如'pc1'表示百分比變化
        
    返回:
        pandas.DataFrame: 包含時間序列數據的DataFrame
    """
    try:
        fred = Fred(api_key=api_key)
        # 根據API文檔正確調用get_series方法
        kwargs = {}
        if frequency:
            kwargs['frequency'] = frequency
        if transform:
            kwargs['units'] = transform
            
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date, **kwargs)
        
        # 將Series轉換為DataFrame
        df = pd.DataFrame(data, columns=[series_id])
        df.index.name = 'date'
        
        return df
    except Exception as e:
        logger.error(f"獲取FRED數據時出錯 (series_id: {series_id}): {str(e)}")
        return pd.DataFrame()

def calculate_inflation_rate(cpi_data):
    """
    根據CPI數據計算年度通脹率
    
    參數:
        cpi_data (pandas.DataFrame): 包含CPI數據的DataFrame
        
    返回:
        pandas.DataFrame: 包含通脹率的DataFrame
    """
    # 確保數據按日期排序
    cpi_data = cpi_data.sort_index()
    
    # 計算同比百分比變化（年度通脹率）
    inflation_data = cpi_data.pct_change(periods=12) * 100
    inflation_data.columns = ['通脹率']
    
    return inflation_data

def fetch_all_indicators(api_key, indicators, years=5):
    """
    獲取所有指定的經濟指標數據
    
    參數:
        api_key (str): FRED API密鑰
        indicators (dict): 指標名稱和系列ID的字典
        years (int): 要獲取的年數，默認為5年
        
    返回:
        dict: 包含所有指標數據的字典
    """
    # 計算開始日期（當前日期減去指定年數）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    logger.info(f"獲取從 {start_date} 到 {end_date} 的經濟指標數據")
    
    all_data = {}
    
    for indicator_name, series_id in indicators.items():
        logger.info(f"獲取 {indicator_name} (系列ID: {series_id}) 的數據")
        
        # 獲取原始數據
        if indicator_name == "通脹率":
            # 對於通脹率，我們需要獲取CPI數據然後計算
            cpi_data = get_fred_data(api_key, series_id, start_date, end_date, frequency='m')
            if not cpi_data.empty:
                all_data[indicator_name] = calculate_inflation_rate(cpi_data)
                logger.info(f"成功計算 {indicator_name} 數據")
            else:
                logger.warning(f"無法獲取 {indicator_name} 的CPI數據")
        else:
            # 對於其他指標，直接獲取數據
            frequency = 'q' if indicator_name == "GDP增長率" else 'm'
            df = get_fred_data(api_key, series_id, start_date, end_date, frequency=frequency)
            
            if not df.empty:
                df.columns = [indicator_name]
                all_data[indicator_name] = df
                logger.info(f"成功獲取 {indicator_name} 數據")
            else:
                logger.warning(f"無法獲取 {indicator_name} 數據")
    
    return all_data

def save_data_to_csv(data_dict, output_dir):
    """
    將數據保存為CSV文件
    
    參數:
        data_dict (dict): 包含數據的字典
        output_dir (str): 輸出目錄路徑
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每個指標的數據
    for indicator_name, df in data_dict.items():
        if not df.empty:
            file_path = os.path.join(output_dir, f"{indicator_name}.csv")
            df.to_csv(file_path)
            logger.info(f"已將 {indicator_name} 數據保存到 {file_path}")
    
    # 合併所有數據到一個文件
    try:
        all_data = pd.concat([df for df in data_dict.values() if not df.empty], axis=1)
        all_data_path = os.path.join(output_dir, "all_indicators.csv")
        all_data.to_csv(all_data_path)
        logger.info(f"已將所有指標數據合併並保存到 {all_data_path}")
    except Exception as e:
        logger.error(f"合併數據時出錯: {str(e)}")

def main():
    """主函數"""
    logger.info("開始從FRED獲取經濟數據")
    
    # 獲取所有指標數據
    all_data = fetch_all_indicators(FRED_API_KEY, INDICATORS)
    
    # 保存數據
    save_data_to_csv(all_data, DATA_DIR)
    
    logger.info("數據獲取和保存完成")

if __name__ == "__main__":
    main()
