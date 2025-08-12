import matplotlib.pyplot as plt # 导入matplotlib库，用于绘制图表
import pandas as pd # 导入pandas库，用于数据处理
import numpy as np # 导入numpy库，用于数值计算
from sb3_contrib import RecurrentPPO # 导入RecurrentPPO模型，用于加载和预测
from custom_trading_env import SimpleTradingEnv # 导入自定义的交易环境
import os # 导入os模块，用于文件路径操作
from delanguage import parse_risk_preference, CLARIFY_MESSAGE # 导入NLP模块，用于解析风险偏好

# 函数：计算回测指标
# portfolio_values: 投资组合价值随时间变化的列表
# trading_days: 年化计算所依据的交易天数，默认为252（一年大约的交易天数）
# 返回值: 包含累计回报率、年化回报率、年化波动率、夏普比率和最大回撤的字典
def calculate_metrics(portfolio_values, trading_days=252):
    # 计算每日收益率，并去除NaN值
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # 计算累计回报率
    cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1
    # 计算年化回报率
    annualized_return = (1 + cumulative_return) ** (trading_days / len(portfolio_values)) - 1
    
    # 计算年化波动率（风险）
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    
    # 计算夏普比率（风险调整后的回报），假设无风险利率为0
    # 如果年化波动率为0，则夏普比率为NaN，避免除以零错误
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    
    # 计算最大回撤
    # rolling_max: 记录历史最高投资组合价值
    rolling_max = pd.Series(portfolio_values).cummax()
    # drawdown: 计算每个时间点的回撤百分比
    drawdown = pd.Series(portfolio_values) / rolling_max - 1
    # max_drawdown: 找到最大的回撤（最负的值）
    max_drawdown = drawdown.min()
    
    # 返回所有计算出的指标
    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

# 主验证函数
def validate():
    # --- 调试信息：打印当前工作目录 ---
    # 帮助用户确认脚本运行时的当前目录，以便排查文件路径问题
    print("当前工作目录:", os.getcwd())

    # --- 数据加载：使用预先划分的验证集 ---
    # 验证集数据已由 download_and_process.py 脚本生成并保存为单独的文件
    # 这里使用 os.path.join 和 os.path.dirname(__file__) 来构建绝对路径，确保文件能被找到
    data_file_path = os.path.join(os.path.dirname(__file__), "AAPL_3y_finrl_val.csv")
    # 读取验证集数据
    validation_data = pd.read_csv(data_file_path)

    # 打印验证数据的使用情况，告知用户正在使用哪个文件进行验证
    print(f"使用 {len(validation_data)} 条数据进行验证 (来自 {os.path.basename(data_file_path)})")

    # --- 获取用户风险偏好 ---
    user_risk_pref = None
    # 提示用户输入风险偏好描述
    print("\n请输入您的风险偏好描述（例如：'我希望稳健一点' 或 '我能接受高风险'）：")
    # 循环等待用户输入，直到解析出有效的风险偏好值
    while user_risk_pref is None:
        user_input = input("用户: ").strip() # 获取用户输入并去除首尾空格
        parsed_value = parse_risk_preference(user_input) # 调用NLP模块解析风险偏好
        # 如果解析结果是浮点数，表示成功解析
        if isinstance(parsed_value, float):
            user_risk_pref = parsed_value # 存储解析出的风险偏好值
            print(f"已解析到风险偏好值: {user_risk_pref:.2f}") # 打印解析结果
        else:
            # 如果解析结果不是浮点数（而是CLARIFY_MESSAGE），则打印提示信息，并继续循环等待用户输入
            print(parsed_value) 

    # --- 环境初始化 ---
    # 实例化自定义的交易环境，并传入验证数据文件的路径
    env = SimpleTradingEnv(data_path=data_file_path) # 直接使用验证文件路径
    
    # --- 模型加载 ---
    # 加载预训练的RecurrentPPO模型
    # 模型文件路径也是通过 os.path.join 和 os.path.dirname(__file__) 构建的绝对路径
    model = RecurrentPPO.load(os.path.join(os.path.dirname(__file__), "ppo_recurrent_agent")) 
    
    # 重置环境到初始状态，并传入用户指定的风险偏好
    obs, info = env.reset(risk_pref=user_risk_pref) 
    # 初始化投资组合价值列表，用于记录回测过程中的价值变化
    portfolio_values = [env.portfolio_value]
    
    # --- LSTM 状态初始化 ---
    # RecurrentPPO模型需要管理其内部的LSTM状态，以便在时间序列上保持记忆
    lstm_states = None # 初始LSTM状态为None
    num_envs = 1 # 对于单个环境，向量化环境的数量通常是1
    
    done = False # 标记回合是否结束
    # 回测循环：直到回合结束
    while not done:
        # 使用 model.predict 进行预测，并传递/更新 lstm_states
        # state: 传入当前的LSTM状态
        # episode_start: 布尔数组，指示每个环境的episode是否开始（对于单个环境，通常是[done]）
        # deterministic=True: 确保预测结果是确定性的，不引入随机性
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=np.array([done]), deterministic=True)
        
        # 在环境中执行预测的动作，并获取下一个观察状态、奖励、是否结束等信息
        obs, reward, terminated, truncated, info = env.step(action)
        # 更新回合结束标志
        done = terminated or truncated
        # 记录当前投资组合价值
        portfolio_values.append(info["portfolio_value"])
    
    # --- 清理临时文件 ---
    # 由于现在直接使用预先划分的验证集文件，不再需要清理临时文件

    # 计算回测指标
    metrics = calculate_metrics(portfolio_values)
    # 打印回测结果
    print("\n📊 回测结果:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2%}")
    
    # 绘制资金曲线图
    plt.figure(figsize=(10, 5)) # 设置图表大小
    plt.plot(portfolio_values, label="Portfolio Value") # 绘制投资组合价值曲线
    plt.title("Portfolio Value Over Time") # 设置图表标题
    plt.xlabel("Step") # 设置X轴标签
    plt.ylabel("Portfolio Value ($)") # 设置Y轴标签
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格
    
    # 保存图表到文件，而不是显示在屏幕上
    plot_save_path = os.path.join(os.path.dirname(__file__), "portfolio_value.png")
    plt.savefig(plot_save_path) # 保存图表为PNG文件
    print(f"\n图表已保存至: {plot_save_path}") # 打印保存路径
    plt.close() # 关闭图表，释放内存资源，确保程序可以自动退出

# 当脚本作为主程序运行时，执行validate函数
if __name__ == "__main__":
    validate()