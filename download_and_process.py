import yfinance as yf # 导入yfinance库，用于下载股票数据
import pandas as pd # 导入pandas库，用于数据处理和分析
import ta # 导入ta（Technical Analysis）库，用于计算技术指标
from sklearn.preprocessing import StandardScaler # 导入StandardScaler，用于数据标准化
import joblib # 导入joblib库，用于保存和加载Python对象（这里用于保存Scaler）
import os # 导入os模块，用于文件路径操作

# 获取当前脚本的目录，用于构建文件的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 函数：下载股票数据并计算初步的技术指标特征
# ticker: 股票代码，默认为"AAPL" (苹果公司)
# period: 数据周期，默认为"3y" (3年)
# interval: 数据时间间隔，默认为"1d" (1天)
# 返回值: 包含股票数据和技术指标的DataFrame，如果下载失败则返回None
def fetch_and_calculate_features(ticker="AAPL", period="3y", interval="1d"):
    """下载股票数据并计算初步的技术指标特征。"""
    print(f"--- Step 1: Downloading {ticker} data ---")
    # 使用yfinance下载股票数据
    # progress=False: 不显示下载进度条
    # auto_adjust=True: 自动调整历史价格，考虑拆股和分红
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    # 检查下载的DataFrame是否为空
    if df.empty:
        print(f"Warning: Downloaded dataframe for {ticker} is empty.")
        return None

    # --- 核心修正：检查并压平多级索引列名 ---
    # yfinance有时会返回多级索引的列名，例如 ('Close', 'AAPL')
    # 我们需要将其转换为单级索引，例如 'Close'，以便后续处理
    if isinstance(df.columns, pd.MultiIndex):
        # get_level_values(0) 会获取多级索引的第0层，也就是'Open', 'Close'等
        df.columns = df.columns.get_level_values(0)

    print("--- Step 2: Calculating technical indicators ---")
    
    # 提取收盘价、最高价和最低价序列，用于技术指标计算
    close_prices = df["Close"].squeeze()
    high_prices = df["High"].squeeze()
    low_prices = df["Low"].squeeze()

    # 计算常用技术指标并添加到DataFrame中
    df["MA20"] = ta.trend.sma_indicator(close_prices, window=20) # 20日简单移动平均线
    df["MA50"] = ta.trend.sma_indicator(close_prices, window=50) # 50日简单移动平均线
    df["RSI14"] = ta.momentum.rsi(close_prices, window=14) # 14日相对强弱指数
    macd = ta.trend.MACD(close_prices) # 移动平均聚散离合器
    df["MACD"] = macd.macd() # MACD线
    df["MACD_signal"] = macd.macd_signal() # MACD信号线
    df["MACD_diff"] = macd.macd_diff() # MACD柱（MACD线 - 信号线）
    bollinger = ta.volatility.BollingerBands(close_prices) # 布林带
    df["Bollinger_High"] = bollinger.bollinger_hband() # 布林带上轨
    df["Bollinger_Low"] = bollinger.bollinger_lband() # 布林带下轨
    df["ATR"] = ta.volatility.average_true_range(high_prices, low_prices, close_prices, window=14) # 14日平均真实波幅
    
    # 删除包含NaN值的行（通常是由于技术指标计算需要历史数据，导致开头部分为NaN）
    df.dropna(inplace=True)
    # 重置DataFrame索引
    df.reset_index(inplace=True)

    # 构建保存原始特征数据CSV文件的路径
    features_csv_path = os.path.join(SCRIPT_DIR, f"{ticker}_3y_features.csv")
    # 将包含原始特征的DataFrame保存到CSV文件
    df.to_csv(features_csv_path, index=False)
    print(f"Saved raw features data to {features_csv_path}")
    return df

# 函数：对特征进行归一化，并构造成适用于RL环境的最终格式
# df: 包含股票数据和技术指标的DataFrame
# ticker: 股票代码
# 返回值: 经过归一化和格式化后的DataFrame
def normalize_and_format_for_env(df, ticker="AAPL"):
    """对特征进行归一化，并构造成适用于RL环境的最终格式。"""
    # 如果输入的DataFrame为空，则跳过归一化
    if df is None:
        print("Input dataframe is None. Skipping normalization.")
        return

    print("--- Step 3: Normalizing features ---")
    # 定义需要进行归一化的特征列
    feature_cols = ["MA20", "MA50", "RSI14", "MACD", "MACD_signal", "MACD_diff", "Bollinger_High", "Bollinger_Low", "ATR"]
    
    # 检查所有必要的特征列是否存在于DataFrame中
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
        
    # 初始化StandardScaler，用于将特征缩放到标准正态分布（均值为0，方差为1）
    scaler = StandardScaler()
    # 对选定的特征列进行拟合和转换
    df_features_scaled = scaler.fit_transform(df[feature_cols])
    # 将归一化后的特征转换回DataFrame，并添加'_scaled'后缀
    df_scaled = pd.DataFrame(df_features_scaled, columns=[f"{col}_scaled" for col in feature_cols], index=df.index)
    
    # 构建Scaler对象的保存路径
    scaler_path = os.path.join(SCRIPT_DIR, "scaler.joblib")
    # 使用joblib保存Scaler对象，以便在模型推理时使用相同的Scaler进行数据预处理
    joblib.dump(scaler, scaler_path)
    print(f"Scaler object saved to {scaler_path}")
    
    # 将原始的OHLCV（开高低收量）数据与归一化后的特征数据合并
    df_final = pd.concat([df[['Open', 'High', 'Low', 'Close', 'Volume']], df_scaled], axis=1)
    # 添加'date'列，用于时间序列排序
    df_final['date'] = df['Date']
    # 添加'tic'列，表示股票代码
    df_final['tic'] = ticker
    
    # 按照日期和股票代码进行排序，确保时间序列的正确性
    df_final.sort_values(by=['date', 'tic'], inplace=True)
    # 重置索引
    df_final.reset_index(drop=True, inplace=True)

    # 构建完整数据的CSV文件路径
    final_csv_path = os.path.join(SCRIPT_DIR, f"{ticker}_3y_finrl.csv")
    # 将完整数据保存到CSV文件
    df_final.to_csv(final_csv_path, index=False)
    print(f"Final data for environment saved to: {final_csv_path}")
    print("\n=== Final Data Head ===")
    print(df_final.head()) # 打印数据头部，方便查看

    # --- Step 4: Splitting data into training and validation sets ---
    print("\n--- Step 4: Splitting data into training and validation sets ---")
    # 定义训练集和验证集的划分比例，例如80%用于训练，20%用于验证
    split_ratio = 0.8 
    # 计算划分索引，确保是整数
    split_idx = int(len(df_final) * split_ratio)

    # 按照时间顺序划分训练集和验证集
    train_df = df_final.iloc[:split_idx] # 从开头到split_idx-1的行作为训练集
    val_df = df_final.iloc[split_idx:] # 从split_idx到末尾的行作为验证集

    # 构建训练集和验证集CSV文件的保存路径
    train_csv_path = os.path.join(SCRIPT_DIR, f"{ticker}_3y_finrl_train.csv")
    val_csv_path = os.path.join(SCRIPT_DIR, f"{ticker}_3y_finrl_val.csv")

    # 保存训练集和验证集到各自的CSV文件
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print(f"Training data saved to: {train_csv_path} ({len(train_df)} rows)")
    print(f"Validation data saved to: {val_csv_path} ({len(val_df)} rows)")

# 当脚本作为主程序运行时，执行以下代码块
if __name__ == "__main__":
    # 调用函数下载数据并计算特征
    raw_features_df = fetch_and_calculate_features(ticker="AAPL")
    # 调用函数对特征进行归一化并格式化数据，同时进行训练集和验证集划分
    normalize_and_format_for_env(raw_features_df, ticker="AAPL")
    print("\nData processing complete.")