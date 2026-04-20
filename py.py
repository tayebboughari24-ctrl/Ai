import streamlit as st
from transformers import pipeline
import pandas as pd
from PIL import Image

# 1. Page Setup
st.set_page_config(page_title="Emotions Analyst", layout="wide")

@st.cache_resource
def load_ai_models():
    # Text Model: High-accuracy multilingual model
    text_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # Image Model: Specialized in facial and visual emotions
    image_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return text_pipe, image_pipe

text_ai, img_ai = load_ai_models()

# Mapping stars to Sentiment Labels
def get_sentiment_label(result):
    label = result['label']
    stars = int(label.split()[0])
    if stars <= 2: return "Negative ❌"
    if stars == 3: return "Neutral 😐"
    return "Positive ✅"

# Header
st.title("🎭 Emotions Analyst")
st.markdown("---")

# Main Interface Tabs
tab_text, tab_image, tab_file = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "📁 Bulk File Analysis"])

# --- TAB 1: TEXT ANALYSIS ---
with tab_text:
    st.header("Analyze Text Sentiment")
    user_input = st.text_area("Paste your text here (English or Arabic):", height=150)
    if st.button("Analyze Text"):
        if user_input:
            res = text_ai(user_input)[0]
            sentiment = get_sentiment_label(res)
            st.subheader(f"Result: {sentiment}")
            st.write(f"Confidence Score: {res['score']:.2f}")
        else:
            st.warning("Please enter text first.")

# --- TAB 2: IMAGE ANALYSIS ---
with tab_image:
    st.header("Analyze Image Emotions")
    uploaded_image = st.file_uploader("Upload a photo...", type=['jpg', 'png', 'jpeg'])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Target Image", width=400)
        if st.button("Identify Emotion"):
            with st.spinner("Processing image..."):
                res = img_ai(img)
                st.success(f"Detected Emotion: **{res[0]['label']}**")

# --- TAB 3: FILE ANALYSIS (THE MAJORITY) ---
with tab_file:
    st.header("Analyze Large Data Files")
    st.write("Upload a file to find out the **Majority Sentiment**.")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        # Load Data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        column_to_analyze = st.selectbox("Select the column containing the text:", df.columns)
        
        if st.button("Run Bulk Analysis"):
            with st.spinner("AI is analyzing all rows..."):
                # Apply Sentiment to every row
                df['Sentiment'] = df[column_to_analyze].apply(lambda x: get_sentiment_label(text_ai(str(x))[0]))
                
                # Calculate Statistics
                counts = df['Sentiment'].value_counts()
                percentages = df['Sentiment'].value_counts(normalize=True) * 100
                majority_sentiment = counts.idxmax()
                
                st.divider()
                st.subheader(f"🏆 Majority Result: {majority_sentiment}")
                
                # Visuals
                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    st.bar_chart(percentages)
                with col_table:
                    st.write("Distribution (%)")
                    st.table(percentages.map("{:.1f}%".format))
                
                # Download results
                st.download_button("Download Full Report", df.to_csv(index=False), "results.csv")
            
