# 导入必要的库
import gymnasium as gym # 导入Gymnasium库，用于构建强化学习环境
from gymnasium import spaces # 导入Gymnasium的spaces模块，用于定义观察空间和动作空间
import numpy as np # 导入NumPy库，用于数值计算
import pandas as pd # 导入Pandas库，用于数据处理

# 定义一个名为SimpleTradingEnv的自定义交易环境，它继承自gymnasium.Env
class SimpleTradingEnv(gym.Env):
    # 定义环境的元数据，这里指定了渲染模式为"human"，表示可以打印信息到控制台
    metadata = {"render_modes": ["human"]}

    # 环境的初始化函数
    # data_path: 股票数据文件的路径
    # initial_cash: 初始现金量，默认为100000
    # transaction_cost_pct: 交易成本百分比，默认为0.001 (0.1%)
    def __init__(self, data_path, initial_cash=100000, transaction_cost_pct=0.001):
        # 调用父类gym.Env的初始化方法
        super(SimpleTradingEnv, self).__init__()
        
        # 加载股票数据。数据应包含日期、开盘价、最高价、最低价、收盘价、成交量以及归一化后的技术指标
        self.data = pd.read_csv(data_path)

        # 识别特征列：所有列名中包含'_scaled'的列都被认为是特征
        self.feature_cols = [col for col in self.data.columns if '_scaled' in col]
        # 如果没有找到特征列，则抛出错误，因为环境需要这些特征来构建观察空间
        if not self.feature_cols:
            raise ValueError("No scaled feature columns found in data.")

        # 设置初始现金量
        self.initial_cash = initial_cash
        # 设置交易成本百分比
        self.transaction_cost_pct = transaction_cost_pct
        # 定义最小交易仓位百分比（例如，每次交易至少动用10%的现金或持仓）
        self.min_trade_pct = 0.1
        # 定义最大交易仓位百分比（例如，每次交易最多动用40%的现金或持仓）
        self.max_trade_pct = 0.4

        # 定义动作空间：离散空间，包含3个动作
        # 0: 卖出 (Sell) - 卖出当前持仓的一部分
        # 1: 持有 (Hold) - 不进行任何交易
        # 2: 买入 (Buy) - 买入股票
        self.action_space = spaces.Discrete(3)
        
        # 定义观察空间：这是一个连续的Box空间，包含特征、持仓比例、现金比例和风险偏好
        # num_features: 技术指标特征的数量
        num_features = len(self.feature_cols)
        # 观察空间的下限：特征可以是负无穷到正无穷，比例和风险偏好在0到1之间
        obs_low = np.array([-np.inf] * num_features + [0, 0, 0], dtype=np.float32)
        # 观察空间的上限：特征可以是负无穷到正无穷，比例和风险偏好在0到1之间
        obs_high = np.array([np.inf] * num_features + [1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 在环境初始化时调用reset方法，设置初始状态
        self.reset()

    # 辅助函数：获取当前时间步的观察状态
    # 返回值：一个NumPy数组，代表当前环境的观察状态
    def _get_observation(self):
        # 获取当前时间步的特征数据（归一化后的技术指标）
        features = self.data.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        # 定义一个很小的常数，用于避免除以零的错误
        epsilon = 1e-10
        # 获取当前时间步的收盘价
        price = self.data.loc[self.current_step, "Close"]
        # 计算当前持有的股票的市值
        pos_value = self.position * price
        # 计算总资产（现金 + 股票市值）
        total_assets = self.cash + pos_value
        # 计算持仓比例和现金比例，避免总资产为零时出现除以零错误
        if total_assets < epsilon:
            pos_ratio = 0.0 # 持仓比例
            cash_ratio = 1.0 # 现金比例
        else:
            pos_ratio = pos_value / total_assets
            cash_ratio = self.cash / total_assets
        
        # 将特征、持仓比例、现金比例和风险偏好拼接成一个完整的观察状态数组
        return np.concatenate([features, [pos_ratio, cash_ratio, self.risk_pref]]).astype(np.float32)

    # 重置环境到初始状态
    # seed: 随机种子，用于可复现性
    # options: 额外选项，这里用于传递风险偏好
    # risk_pref: 风险偏好值（0到1之间），如果提供则使用，否则随机生成
    # 返回值：
    #   observation: 初始观察状态
    #   info: 包含风险偏好和交易仓位百分比的字典
    def reset(self, seed=None, options=None, risk_pref=None):
        # 调用父类gym.Env的reset方法
        super().reset(seed=seed)
        # 重置当前时间步为0
        self.current_step = 0
        # 重置持仓股票数量为0
        self.position = 0.0
        # 重置现金量为初始现金量
        self.cash = float(self.initial_cash)
        # 重置投资组合总价值为初始现金量
        self.portfolio_value = float(self.initial_cash)
        # 重置上一时间步的投资组合总价值（用于奖励计算）
        self.prev_portfolio_value = float(self.initial_cash)
        # 清空交易日志
        self.trading_log = []

        # 根据传入的risk_pref设置风险偏好，如果未传入则随机生成
        if risk_pref is not None:
            self.risk_pref = risk_pref
        else:
            self.risk_pref = np.random.uniform(0, 1) # 随机生成风险偏好
        
        # 根据风险偏好计算当前交易的仓位百分比
        # 风险偏好越高，交易仓位百分比越大，交易越激进
        self.trade_size_pct = self.min_trade_pct + (self.max_trade_pct - self.min_trade_pct) * self.risk_pref
        
        # 返回初始观察状态和额外信息
        return self._get_observation(), {"risk_pref": self.risk_pref, "trade_size_pct": self.trade_size_pct}

    # 执行一个动作，并推进环境一个时间步
    # action: 代理选择的动作 (0:卖出, 1:持有, 2:买入)
    # 返回值：
    #   observation: 下一个时间步的观察状态
    #   reward: 执行动作后获得的奖励
    #   done: 布尔值，表示回合是否结束
    #   truncated: 布尔值，表示回合是否因截断而结束（例如达到最大步数），这里未使用
    #   info: 包含当前投资组合总价值的字典
    def step(self, action):
        # 记录当前投资组合总价值，用于计算奖励
        self.prev_portfolio_value = self.portfolio_value
        # 获取当前时间步的收盘价
        price = self.data.loc[self.current_step, "Close"]

        # 特殊处理：如果价格极低（接近于零），强制动作为持有，避免除以零或不合理交易
        if price <= 1e-8:
            action = 1 # 强制持有

        # 根据代理选择的动作执行交易逻辑
        if action == 2:  # 买入动作
            # 计算本次买入应使用的现金量（基于交易仓位百分比）
            buy_amount_in_cash = self.cash * self.trade_size_pct
            # 确保买入金额大于0
            if buy_amount_in_cash > 1e-8:
                # 计算买入所需的总成本（包含交易成本）
                cost = buy_amount_in_cash * (1 + self.transaction_cost_pct)
                # 如果现金足够支付成本
                if self.cash >= cost:
                    # 计算可以买入的股票数量
                    num_shares_to_buy = buy_amount_in_cash / price
                    # 更新持仓数量
                    self.position += num_shares_to_buy
                    # 更新现金量
                    self.cash -= cost
        elif action == 0:  # 卖出动作
            # 计算本次卖出应卖出的股票数量（基于交易仓位百分比）
            num_shares_to_sell = self.position * self.trade_size_pct
            # 确保卖出数量大于0
            if num_shares_to_sell > 1e-8:
                # 更新持仓数量
                self.position -= num_shares_to_sell
                # 更新现金量（包含交易成本）
                self.cash += (num_shares_to_sell * price) * (1 - self.transaction_cost_pct)
        # 如果 action == 1 (持有)，则不执行任何交易，持仓和现金不变
        
        # 更新投资组合总价值（现金 + 当前持仓市值）
        self.portfolio_value = self.cash + self.position * price
        # 定义一个很小的常数，用于避免对数计算中出现零或负数
        epsilon = 1e-10
        # 计算奖励：当前投资组合价值与上一时间步投资组合价值的对数收益率
        # 这种奖励函数鼓励代理最大化每一步的收益增长
        reward = np.log((self.portfolio_value + epsilon) / (self.prev_portfolio_value + epsilon))

        # 记录当前时间步的交易日志，用于后续分析
        self.trading_log.append({
            "step": self.current_step, "price": price, "action": action,
            "position": self.position, "cash": self.cash, "portfolio_value": self.portfolio_value
        })

        # 推进时间步
        self.current_step += 1
        # 判断回合是否结束：如果当前时间步达到数据末尾，则回合结束
        done = self.current_step >= len(self.data) - 1

        # 返回下一个观察状态、奖励、回合是否结束、截断标志和额外信息
        return self._get_observation(), reward, done, False, {"portfolio_value": self.portfolio_value}

    # 渲染环境信息（打印到控制台）
    # mode: 渲染模式，这里是"human"
    def render(self, mode='human'):
        # 打印当前时间步、投资组合总价值、持仓数量和现金量
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, Position: {self.position:.4f}, Cash: {self.cash:.2f}")