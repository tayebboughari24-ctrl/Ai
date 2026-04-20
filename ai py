import streamlit as st
from transformers import pipeline
import pandas as pd
from PIL import Image

# إعداد الصفحة
st.set_page_config(page_title="AI Sentiment Pro", layout="wide")

@st.cache_resource
def load_ai():
    # أقوى نموذج عربي للنصوص
    text_model = pipeline("sentiment-analysis", model="UBC-NLP/MARBERTv2")
    # نموذج متقدم للصور
    image_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return text_model, image_model

text_ai, img_ai = load_ai()

st.title("📊 محلل المشاعر الذكي")

tab1, tab2, tab3 = st.tabs(["📝 نصوص", "🖼️ صور", "📁 ملفات (أغلبية)"])

with tab1:
    user_input = st.text_area("أدخل النص المراد تحليله:")
    if st.button("تحليل النص"):
        res = text_ai(user_input)[0]
        st.success(f"النتيجة: {res['label']} | الدقة: {res['score']:.2f}")

with tab2:
    img_file = st.file_uploader("ارفع صورة هنا", type=['jpg', 'png', 'jpeg'])
    if img_file:
        img = Image.open(img_file)
        st.image(img, width=300)
        res = img_ai(img)
        st.info(f"المشاعر المكتشفة: {res[0]['label']}")

with tab3:
    data_file = st.file_uploader("ارفع ملف Excel أو CSV", type=['csv', 'xlsx'])
    if data_file:
        df = pd.read_csv(data_file) if data_file.name.endswith('csv') else pd.read_excel(data_file)
        col = st.selectbox("اختر عمود النصوص:", df.columns)
        if st.button("بدء تحليل الأغلبية"):
            with st.spinner("جاري التحليل..."):
                df['Result'] = df[col].apply(lambda x: text_ai(str(x))[0]['label'])
                stats = df['Result'].value_counts(normalize=True) * 100
                st.metric("الشعور الغالب", stats.idxmax())
                st.bar_chart(stats)
