# 导入必要的库
import json # 用于处理JSON数据
import re # 用于正则表达式操作
from openai import OpenAI  # 导入OpenAI库，用于调用大语言模型API

# ==== 配置（请替换为你的 DeepSeek key / endpoint） ====
# 初始化OpenAI客户端，用于与DeepSeek API进行通信
# api_key: 你的DeepSeek API密钥
# base_url: DeepSeek API的基础URL
client = OpenAI(
    api_key="",    # 替换成你的 DeepSeek secret
    base_url="https://api.deepseek.com" # 如果需要可保留/修改
)

# 当模型无法解析用户输入时，返回给用户的提示信息
CLARIFY_MESSAGE = "请输入更准确的描述，以帮助您制定更好的投资策略"

# 辅助函数：尝试从任意文本中提取 JSON 对象
# 作用：当大语言模型返回的文本中包含JSON，但可能混杂了其他解释性文字时，尝试从中提取出有效的JSON结构。
# text: 包含可能JSON的字符串
# 返回值：解析出的JSON对象（字典或列表），如果无法解析则返回None
def _find_json_in_text(text: str):
    # 使用正则表达式查找第一个以'{'开头并以'}'结尾的字符串，或者以'['开头并以']'结尾的字符串
    # re.S 标志表示'.'可以匹配包括换行符在内的所有字符
    jmatch = re.search(r'({.*})', text, flags=re.S)
    if jmatch:
        # 提取匹配到的JSON字符串
        js = jmatch.group(1)
        try:
            # 尝试将提取的字符串解析为JSON对象
            return json.loads(js)
        except:
            # 如果解析失败，则忽略错误并继续
            pass
    return None

# 辅助函数：尝试从任意文本中提取 0 到 1 之间的浮点数
# 作用：当大语言模型直接返回一个数字（可能带有一些解释性文字）时，从中提取出有效的风险偏好数值。
# text: 包含可能数字的字符串
# 返回值：提取出的浮点数，如果无法解析或不在0-1范围内则返回None
def _find_number_in_text(text: str):
    # 使用正则表达式查找 0.x 或 1 或 0 的匹配
    # (?<!\d) 和 (?!
# 确保匹配的是独立的数字，而不是数字的一部分
    m = re.search(r'(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)', text)
    if m:
        try:
            # 尝试将匹配到的字符串转换为浮点数
            v = float(m.group(1))
            # 检查数字是否在0到1的有效范围内
            if 0.0 <= v <= 1.0:
                return v
        except:
            # 如果转换失败，则忽略错误并继续
            pass
    return None

# 核心函数：解析用户输入的自然语言风险偏好
# user_input: 用户输入的描述风险偏好的字符串
# 返回值：
#   - 若能解析出有效风险偏好（0~1的浮点数），返回该浮点数；
#   - 若无法解析（不相关或不确定），返回 CLARIFY_MESSAGE 字符串。
def parse_risk_preference(user_input: str):
    """
    将用户输入的自然语言风险偏好描述转换为一个0到1之间的数值。
    如果无法识别或不相关，则返回一个澄清提示信息。
    """
    # 构建系统提示 (System Prompt)：给大语言模型设定角色和行为规则
    system_prompt = (
        "你是一个把自然语言风险偏好映射为数值的助手。" # 设定助手角色
        "当且仅当用户确实在描述投资风险偏好或需要根据风险倾向调整策略时，返回有效的 risk 数值（0 到 1）。" # 明确何时返回数值
        "**必须**以严格的 JSON 格式返回，且不要返回其它非 JSON 文本，JSON 格式如下：\n" # 强制JSON输出格式
        '{"valid": true|false, "risk": <number|null>, "message": "<简短说明>"}\n\n' # 给出JSON格式示例
        "规则：\n" # 详细说明JSON字段的含义和规则
        "- valid 为 true 表示你判定用户输入与风险偏好相关且可以给出数值（risk为0~1的小数）。\n" # valid字段说明
        "- valid 为 false 表示用户输入与风险偏好无关或信息不足（risk 为 null），此时 message 请设置为：" # valid字段说明
        " '请输入更准确的描述，以帮助您指定更好的投资策略'。\n" # message字段说明
        "- 如果用户希望更激进（例如“我希望赚钱，不在乎风险”），risk 取较高值如 0.9 或 1.0；\n" # 激进偏好映射
        "- 如果用户希望稳健/无风险（例如“我不希望有任何风险”），risk 取较低值如 0.1 或 0.2；\n" # 保守偏好映射
        "- 如果用户表达‘平衡’或‘中等’，risk 为 0.5。\n" # 平衡偏好映射
        "只输出 JSON，不要多余文字。\n" # 再次强调只输出JSON
    )

    # 构建用户提示 (User Prompt)：将用户的实际输入嵌入到提示中
    user_prompt = f'用户输入: "{user_input}"\n请根据上面规则输出 JSON。'

    try:
        # 调用 DeepSeek（或兼容 OpenAI 的 client）聊天接口
        # model: 指定使用deepseek-chat模型
        # messages: 包含系统提示和用户提示的对话历史
        # temperature: 控制模型输出的随机性，0.0表示确定性最高
        # max_tokens: 限制模型生成响应的最大token数量
        response = client.chat.completions.create(
            model="deepseek-chat",              # 如果 DeepSeek 模型名称不同请替换
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        # 解析模型返回的文本内容
        text = ""
        if hasattr(response, "choices"):
            # 兼容OpenAI风格的响应对象结构
            text = response.choices[0].message.content.strip()
        elif isinstance(response, dict) and "choices" in response:
            # 兼容字典形式的响应结构
            text = response["choices"][0]["message"]["content"].strip()
        else:
            # 尝试直接从响应中获取内容
            text = str(response).strip()

    except Exception as e:
        # 如果API调用失败（例如网络问题、API Key无效等），返回澄清提示
        # 在生产环境中，这里通常会记录详细的异常日志
        return CLARIFY_MESSAGE

    # 1) 首先尝试将模型返回的文本解析为严格的 JSON
    parsed_json = None
    try:
        parsed_json = json.loads(text)
    except:
        # 如果不是严格的JSON，尝试使用辅助函数从文本中提取JSON子串
        parsed_json = _find_json_in_text(text)

    # 如果成功解析到JSON对象
    if isinstance(parsed_json, dict):
        # 从JSON中获取valid、risk和message字段
        valid = parsed_json.get("valid", None)
        risk = parsed_json.get("risk", None)
        message = parsed_json.get("message", "")

        # 如果valid为True且risk是一个有效的数字
        if valid is True and isinstance(risk, (int, float)):
            # 将risk值限制在0到1的范围内（钳制操作），确保其有效性
            try:
                rv = float(risk)
                rv = max(0.0, min(1.0, rv))
                return rv
            except:
                # 如果钳制过程中出现错误，返回澄清提示
                return CLARIFY_MESSAGE
        else:
            # 如果valid为False或risk不是有效数字，返回澄清提示
            return CLARIFY_MESSAGE

    # 2) 如果未能成功解析JSON，则尝试直接从文本中提取 0 到 1 之间的数字
    num = _find_number_in_text(text)
    if num is not None:
        return float(num)

    # 3) 额外判断：如果用户输入本身内容很少或像问候语，直接返回提示
    low_content = user_input.strip().lower()
    # 检查用户输入是否过短或只包含常见问候语
    if len(low_content) < 6 or re.match(r'^(hi|hello|你好|谢谢|thanks|ps)$', low_content):
        return CLARIFY_MESSAGE

    # 4) 最后兜底：如果以上所有尝试都未能成功解析出有效的风险偏好，则返回澄清提示
    return CLARIFY_MESSAGE

# 主函数：用于在命令行中测试风险偏好解析功能
def main():
    print("请输入你的风险偏好描述，输入 exit 或 quit 退出：")
    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("程序结束")
            break
        out = parse_risk_preference(user_input)
        if isinstance(out, float):
            print(f"解析的风险偏好值: {out:.2f}")
        else:
            # 返回字符串（澄清提示）
            print(out)

# 当脚本作为主程序运行时，执行main函数
if __name__ == "__main__":
    main()
