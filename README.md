# 美股宏觀儀表盤

這是一個使用FRED (Federal Reserve Economic Data) API獲取美國宏觀經濟數據並通過Streamlit創建互動式儀表盤的專案。

## 功能特點

- 從FRED獲取多種宏觀經濟指標數據
- 數據清理、處理和標準化
- 互動式折線圖可視化
- 時間範圍選擇功能
- 指標對比分析
- 相關性熱力圖分析

## 包含的經濟指標

主要指標：
- GDP增長率
- 失業率
- 通脹率
- 利率
- 消費者信心指數

次要指標：
- 工業生產指數
- 零售銷售
- 住房開工數
- 貿易差額
- 企業信心指數

## 安裝說明

### 前提條件

- Python 3.8+
- FRED API密鑰 (可從 [FRED網站](https://fred.stlouisfed.org/docs/api/api_key.html) 獲取)

### 安裝步驟

1. 克隆此倉庫到本地：

```bash
git clone https://github.com/你的用戶名/us_macro_dashboard.git
cd us_macro_dashboard
```

2. 安裝所需的Python庫：

```bash
pip install pandas numpy matplotlib streamlit fredapi plotly
```

## 使用說明

### 1. 獲取經濟數據

首先，運行數據獲取腳本來從FRED獲取經濟數據：

```bash
python src/fetch_fred_data.py
```

注意：請確保在運行腳本前已在`src/fetch_fred_data.py`文件中更新您的FRED API密鑰。

### 2. 處理數據

接下來，運行數據處理腳本來清理和分析獲取的數據：

```bash
python src/process_data.py
```

### 3. 啟動儀表盤

最後，啟動Streamlit應用程序：

```bash
streamlit run src/app.py
```

應用程序將在瀏覽器中打開，默認地址為 http://localhost:8501

## 部署到Streamlit Cloud

要將此應用程序部署到Streamlit Cloud，請按照以下步驟操作：

1. 將代碼推送到您的GitHub倉庫：

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. 訪問 [Streamlit Cloud](https://streamlit.io/cloud) 並登錄您的帳戶

3. 點擊 "New app" 按鈕

4. 選擇您的GitHub倉庫、分支和主文件路徑（src/app.py）

5. 在高級設置中，添加以下密鑰：
   - FRED_API_KEY: 您的FRED API密鑰

6. 點擊 "Deploy" 按鈕

7. 等待部署完成，您的應用程序將在幾分鐘內上線

## 項目結構

```
us_macro_dashboard/
├── data/                  # 數據目錄
│   ├── processed/         # 處理後的數據
│   └── ...                # 原始數據文件
├── src/                   # 源代碼
│   ├── fetch_fred_data.py # 數據獲取腳本
│   ├── process_data.py    # 數據處理腳本
│   └── app.py             # Streamlit應用程序
├── README.md              # 項目說明
└── todo.md                # 開發任務清單
```

## 自定義

您可以通過修改以下文件來自定義儀表盤：

- `src/fetch_fred_data.py`: 更改要獲取的經濟指標
- `src/app.py`: 修改儀表盤布局和功能

## 注意事項

- 數據每次運行腳本時都會從FRED獲取最新數據
- 默認獲取最近5年的數據
- 儀表盤提供了多種時間範圍選擇選項

## 故障排除

如果遇到問題：

1. 確保您的FRED API密鑰有效且正確設置
2. 檢查是否已安裝所有必需的Python庫
3. 確保已按順序運行腳本：先獲取數據，再處理數據，最後啟動儀表盤

## 貢獻

歡迎提交問題和改進建議！

## 許可

此項目採用MIT許可證。
