from sb3_contrib import RecurrentPPO # 从sb3_contrib库导入RecurrentPPO，这是一个支持循环神经网络（RNN）的PPO实现
from stable_baselines3.common.vec_env import DummyVecEnv # 从stable_baselines3导入DummyVecEnv，用于将单个环境包装成向量化环境
from custom_trading_env import SimpleTradingEnv # 导入自定义的交易环境
import torch # 导入PyTorch库，用于检查GPU是否可用
import os # 导入os模块，用于文件路径操作

# 定义文件路径
# SCRIPT_DIR: 当前脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH: 训练数据文件的路径，这里指向经过划分后的训练集数据
DATA_PATH = os.path.join(SCRIPT_DIR, "AAPL_3y_finrl_train.csv")
# MODEL_SAVE_PATH: 训练好的模型保存的路径和文件名（不含.zip后缀）
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "ppo_recurrent_agent") # 新模型命名

# 函数：训练强化学习代理
# total_timesteps: 总的训练时间步数，默认为500000
def train(total_timesteps=500000):
    """
    训练一个基于LSTM的、拥有记忆能力的RecurrentPPO模型。
    这个模型能够处理时间序列数据，并根据历史信息做出决策。
    """
    print("--- Step 1: Initializing environment for RecurrentPPO training ---")
    # 实例化自定义的交易环境，并传入训练数据路径
    env = SimpleTradingEnv(data_path=DATA_PATH)
    # 将单个环境包装成向量化环境。对于单个环境，DummyVecEnv是最简单的选择。
    # 向量化环境可以并行运行多个环境实例，提高数据收集效率，但这里只有一个实例。
    vec_env = DummyVecEnv([lambda: env])

    # 检查是否有可用的GPU (CUDA)，否则使用CPU进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    # --- 核心升级: 使用 RecurrentPPO 和 MlpLstmPolicy ---
    # policy_kwargs: 用于配置策略网络（Policy Network）的参数
    # RecurrentPPO使用MlpLstmPolicy，它结合了多层感知机（MLP）和长短期记忆网络（LSTM）
    policy_kwargs = dict(
        # net_arch: 定义MLP层的结构
        # pi: 策略网络（Policy Network）的MLP层结构，这里是两个128个神经元的隐藏层
        # vf: 价值网络（Value Network）的MLP层结构，这里是两个128个神经元的隐藏层
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        # lstm_hidden_size: LSTM层的隐藏状态大小，决定了LSTM的记忆能力
        lstm_hidden_size=128,
    )
    
    print("--- Step 2: Defining and initializing a RecurrentPPO (LSTM) model ---")
    # 实例化RecurrentPPO模型
    # "MlpLstmPolicy": 指定使用带有LSTM的策略网络
    # vec_env: 传入向量化环境
    # verbose=1: 打印训练过程中的详细信息
    # n_steps: 每个环境在更新策略之前收集的步数。对于RecurrentPPO，这通常是LSTM序列的长度。
    # batch_size: 每次梯度更新使用的样本数量
    # learning_rate: 优化器的学习率
    # policy_kwargs: 传入策略网络的配置参数
    # device: 指定训练设备（CPU或GPU）
    model = RecurrentPPO(
        "MlpLstmPolicy",        # 使用带有LSTM的策略
        vec_env, 
        verbose=1,
        n_steps=4096,           
        batch_size=128,         
        learning_rate=1e-4,
        policy_kwargs=policy_kwargs,
        device=device
    )

    print(f"--- Step 3: Starting training for {total_timesteps} timesteps ---")
    # 开始模型训练
    # total_timesteps: 模型将学习的总时间步数
    # progress_bar=True: 显示训练进度条
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- Training complete ---")

    print(f"--- Step 4: Saving recurrent model to {MODEL_SAVE_PATH}.zip ---")
    # 保存训练好的模型。模型会被保存为一个.zip文件，包含模型参数和训练状态。
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully.")

# 当脚本作为主程序运行时，执行train函数
if __name__ == "__main__":
    train()