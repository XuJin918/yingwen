# 导入核心库
import streamlit as st  
import joblib  
import numpy as np  
import pandas as pd  
import shap  
import matplotlib.pyplot as plt  
from lime.lime_tabular import LimeTabularExplainer  
import warnings

# ===================== 基础配置：解决兼容性问题 =====================
warnings.filterwarnings('ignore')
# 设置matplotlib后端，适配Streamlit绘图
plt.switch_backend('Agg')
# 设置中文字体（解决LIME/SHAP图表中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 无中文字体时用这个，有则替换为['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 模型与数据加载（带异常处理） =====================
@st.cache_resource  # 缓存模型，避免重复加载
def load_model_and_data():
    """加载模型和测试数据，带异常处理"""
    try:
        # 加载训练好的随机森林模型（确保RF.pkl与脚本同目录）
        model = joblib.load('RF.pkl')  
        st.success("✅ 模型加载成功！")
    except FileNotFoundError:
        st.error("❌ 未找到RF.pkl文件，请确认模型文件在脚本同目录下！")
        st.stop()
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()

    try:
        # 加载测试数据（用于LIME解释器，确保X_test.csv与脚本同目录）
        X_test = pd.read_csv('X_test.csv')  
        st.success("✅ 测试数据加载成功！")
    except FileNotFoundError:
        st.error("❌ 未找到X_test.csv文件，请确认数据文件在脚本同目录下！")
        st.stop()
    except Exception as e:
        st.error(f"❌ 测试数据加载失败：{str(e)}")
        st.stop()
    
    return model, X_test

# 执行加载
model, X_test = load_model_and_data()

# 定义特征名称（与输入顺序严格一致）
feature_names = [
    "硬的食物", "睡眠时长", "洗手扶手", "多药", "安全警示",
    "是否住院", "经济", "PHQ", "锻炼次数", "ACEzong", "教育程度", "运动场所", "童年健康"
]  

# ===================== 2. Streamlit页面配置 =====================
st.set_page_config(
    page_title="Frailty Risk Predictor for Older Adults in Long-Term Care Facilities", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Frailty Risk Predictor for Older Adults in Long-Term Care Facilities")  
st.markdown("### Please complete the following items and click Predict to obtain frailty risk assessment results")

# ===================== 3. 特征输入组件（按编码规则设计） =====================
# 1. 咀嚼功能（0：完全没问题，1：有问题）
hard_food = st.selectbox(
    "1. Are you able to chew hard foods such as apples?",
    options=[0, 1],
    format_func=lambda x: "Yes, no difficulty" if x == 0 else "Yes with caution / Unable to chew",
    key="hard_food"
)

# 2. 睡眠时长（0：不正常，1：正常）
sleep_hours = st.selectbox(
    "2. How many hours do you usually sleep per night?",
    options=[0, 1],
    format_func=lambda x: "< 7 hours / ≥ 9 hours" if x == 0 else "7–9 hours",
    key="sleep_hours"
)

# 3. 适老化设施不足（0：否，1：是）
handrail = st.selectbox(
    "3. Does your long-term care facility lack age-friendly facilities (e.g., handrails beside wash basins)?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    key="handrail"
)

# 4. 多药（0：否，1：是）
multiple_drugs = st.selectbox(
    "4. Do you currently take five or more prescription medications?",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是",
    key="multiple_drugs"
)

# 5. 安全警示（0：否，1：是）
safety_warning = st.selectbox(
    "5. Are there safety warning signs in the exercise areas of your facility?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    key="safety_warning"
)

# 6. 是否住院（0：否，1：是）
hospitalization = st.selectbox(
    "6. Have you been hospitalized within the past year?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    key="hospitalization"
)

# 7. 经济状况（0：贫困，1：非贫困）
economy = st.selectbox(
    "7. How would you rate your current economic status compared with others?",
    options=[0, 1],
    format_func=lambda x: "Poor / Relatively poor" if x == 0 else "Average / Relatively good / Good",
    key="economy"
)

# 8. PHQ（0：否，1：是）
phq = st.selectbox(
    "8. Depressive symptoms (PHQ-2):\n1. Over the past two weeks, how often have you had little interest or pleasure in doing things?\n2. Over the past two weeks, how often have you felt down, depressed or hopeless?",
    options=[0, 1],
    format_func=lambda x: "Total score < 3" if x == 0 else "Total score ≥ 3",
    key="phq"
)

# 9. 锻炼次数（0：无体育锻炼，1：有体育锻炼）
exercise_times = st.selectbox(
    "9. 您每周是否有体育锻炼？",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是",
    key="exercise_times"
)

# 10. ACEzong（0：否，1：是）
acezong = st.selectbox(
    "10. Adverse Childhood Experiences (ACEs): Please indicate whether you experienced any of the following before the age of 18:\n- Verbal abuse, humiliation or contempt;\n- Physical pushing, hitting or assault;\n- Unwanted physical contact or sexual coercion;\n- Lack of care or emotional support from family;\n- Neglect of basic daily needs such as food and clothing;\n- Parental separation, divorce or abandonment;\n- Frequent exposure to domestic violence;\n- Living with family members with alcohol or drug abuse;\n- Living with family members with mental illness or suicide attempts;\n- Having an incarcerated family member.",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    key="acezong"
)

# 11. 教育程度（0：小学及以下，1：初中及以上）
education = st.selectbox(
    "11. What is your highest educational level?",
    options=[0, 1],
    format_func=lambda x: "Primary school or below" if x == 0 else "Junior high school or above",
    key="education"
)

# 12. 运动场所区（0：无，1：有）
fitness_area = st.selectbox(
    "12. Does your long-term care facility have dedicated exercise venues?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes",
    key="fitness_area"
)

# 13. 童年健康（0：不差，1：差）
childhood_health = st.selectbox(
    "13. Compared with peers, how would you rate your health status before the age of 18?",
    options=[0, 1],
    format_func=lambda x: "Much better / Slightly better / Similar" if x == 0 else "Slightly poor / Very poor",
    key="childhood_health"
)

# ===================== 4. 数据处理与预测 =====================
# 整合用户输入特征（顺序与feature_names严格一致）
feature_values = [
    hard_food, sleep_hours, handrail, multiple_drugs,
    safety_warning, hospitalization, economy, phq, exercise_times,
    acezong, education, fitness_area, childhood_health
]

# 转换为模型输入格式（二维数组，适配sklearn要求）
features = np.array([feature_values])  

# 预测按钮逻辑
if st.button("📌 Start Prediction", type="primary"):
    try:
        # 模型预测
        predicted_class = model.predict(features)[0]  # 0：低风险，1：高风险
        predicted_proba = model.predict_proba(features)[0]  # 概率值

        # 显示预测结果（中文适配）
        st.subheader("📊 Prediction Result")
        risk_label = "High Risk" if predicted_class == 1 else "Low High Risk"
        st.write(f"**风险等级：{predicted_class}（{risk_label}）**")
        st.write(f"**风险概率：** 低风险概率 {predicted_proba[0]:.2%} | 高风险概率 {predicted_proba[1]:.2%}")

        # 生成个性化建议（中文）
        st.subheader("💡 Health Recommendations")
        probability = predicted_proba[predicted_class] * 100
        if predicted_class == 1:
            advice = (
                f"模型预测您的衰弱风险为高风险（概率{probability:.1f}%）。"
                "建议尽快前往医疗机构进行全面的衰弱评估，重点关注营养摄入、睡眠质量、心理健康等方面，"
                "同时可根据自身情况增加适宜的体育锻炼，改善生活环境。"
            )
        else:
            advice = (
                f"模型预测您的衰弱风险为低风险（概率{probability:.1f}%）。"
                "建议保持现有健康生活方式，定期进行健康体检，关注童年健康/经济等潜在影响因素，"
                "持续维持规律锻炼和良好的经济、睡眠状况。"
            )
        st.write(advice)
    # 关键补充：添加except块，闭合try结构（解决语法错误）
    except Exception as e:
        # 捕获异常并友好提示用户
        st.error(f"⚠️ 预测过程出错：{str(e)}")
        # 可选：打印详细错误到控制台，方便调试
        # print(f"调试信息 - 预测错误：{e}")

# ===================== 5. 页脚信息 =====================
st.markdown("---")
st.markdown("⚠️ 提示：本预测结果仅作参考，最终健康评估请以专业医疗机构诊断为准。")