import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# ==================== 测试数据 ====================
test_data = [
    {
        "question": "2026年世界杯冠军是哪支球队？",
        "answer": "西班牙队",
        "contexts": ["西班牙队通过加时赛以1比0战胜卫冕冠军阿根廷队，队史第二次夺得世界杯冠军。"]
    },
    {
        "question": "2026年世界杯金靴奖得主是谁？进了多少球？",
        "answer": "姆巴佩，10球",
        "contexts": ["法国前锋姆巴佩以10粒进球追平了盖德·穆勒在1970年创造的单届进球纪录，并成为世界杯历史上首位两次获得金靴奖的球员。"]
    },
    {
        "question": "2026年世界杯金球奖得主是谁？",
        "answer": "罗德里",
        "contexts": ["西班牙中场罗德里凭借在攻防两端的稳定表现，荣获金球奖（最佳球员）。"]
    },
    {
        "question": "2026年世界杯决赛的比分是多少？",
        "answer": "西班牙 1-0 阿根廷",
        "contexts": ["西班牙队凭借费兰·托雷斯在加时赛第106分钟的绝杀进球，以1比0战胜10人应战的阿根廷队。"]
    },
    {
        "question": "2026年世界杯季军是哪支球队？",
        "answer": "英格兰队",
        "contexts": ["季军争夺战中，法国队以6比4战胜英格兰队获得季军。"]
    },
    {
        "question": "公司工作日加班费怎么算？",
        "answer": "工作日加班按1.5倍工资计算",
        "contexts": ["加班时间按照劳动法规定计算加班费，工作日加班按1.5倍工资计算。"]
    },
    {
        "question": "员工工作满3年，年休假有几天？",
        "answer": "5天",
        "contexts": ["工作满1年不满10年的，年休假5天"]
    },
    {
        "question": "迟到或早退超过多少分钟算旷工半天？",
        "answer": "超过30分钟",
        "contexts": ["迟到或早退超过30分钟的，视为旷工半天。"]
    },
    {
        "question": "2026年世界杯最佳年轻球员是谁？",
        "answer": "亚马尔",
        "contexts": ["最佳年轻球员：亚马尔（西班牙），18岁"]
    },
    {
        "question": "2026年世界杯在哪几个国家举办？",
        "answer": "美国、加拿大、墨西哥",
        "contexts": ["由美国、加拿大和墨西哥联合主办"]
    }
]
# ==================== 评估函数 ====================
def evaluate_faithfulness(question: str, answer: str, contexts: list) -> float:
    """评估答案是否基于上下文（忠实度），无幻觉则分数接近 1"""
    prompt = f"""你是一个专业的评估模型。请判断以下AI生成的答案是否完全基于提供的上下文信息，没有添加任何外部知识。
上下文：
{contexts[0] if contexts else '无'}
问题：{question}
AI回答：{answer}
请只回答一个数字分数（0到1之间）：
- 1.0：回答完全基于上下文，没有添加任何外部信息
- 0.5：回答大部分基于上下文，但有一些信息不在上下文中
- 0.0：回答包含大量上下文以外的信息
只返回数字："""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return min(1.0, max(0.0, score))
    except:
        return 0.5
def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """评估回答是否直接回答了问题（答案相关性）"""
    prompt = f"""你是一个专业的评估模型。请判断以下AI回答是否直接回答了用户的问题。
用户问题：{question}
AI回答：{answer}
请只回答一个数字分数（0到1之间）：
- 1.0：回答直接、完整地解决了问题
- 0.5：回答部分相关，但没有完全回答问题
- 0.0：回答完全跑题
只返回数字："""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=50
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return min(1.0, max(0.0, score))
    except:
        return 0.5
# ==================== 运行评估 ====================
print(f"📊 开始评估 {len(test_data)} 个测试用例...")
print("=" * 60)
results = []
for i, item in enumerate(test_data):
    print(f"正在评估第 {i+1}/{len(test_data)} 个问题...")
    faithfulness = evaluate_faithfulness(
        item["question"],
        item["answer"],
        item["contexts"]
    )
    answer_relevancy = evaluate_answer_relevancy(
        item["question"],
        item["answer"]
    )
    results.append({
        "question": item["question"][:30] + "...",
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy
    })
    print(f"  忠实度: {faithfulness:.2f}, 答案相关性: {answer_relevancy:.2f}")
    time.sleep(0.5)
# ==================== 输出结果 ====================
print("\n" + "=" * 60)
print("📊 评估结果汇总：")
df = pd.DataFrame(results)
print(df.to_string(index=False))
avg_faithfulness = df["faithfulness"].mean()
avg_relevancy = df["answer_relevancy"].mean()
print("\n" + "=" * 60)
print(f"✅ 平均忠实度 (Faithfulness): {avg_faithfulness:.3f}")
print(f"✅ 平均答案相关性 (Answer Relevancy): {avg_relevancy:.3f}")
df.to_csv("evaluation_manual_report.csv", index=False)
print("\n✅ 详细报告已保存为 evaluation_manual_report.csv")