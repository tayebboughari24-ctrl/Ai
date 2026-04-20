import streamlit as st
from transformers import pipeline
import pandas as pd
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Emotions Analyst", layout="wide")

@st.cache_resource
def load_ai():
    # High-accuracy multilingual model (Supports Arabic & English)
    # Using a 5-star rating model which we map to Neg/Neu/Pos
    text_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # Advanced Image Emotion Model
    image_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return text_model, image_model

text_ai, img_ai = load_ai()

# Sidebar
st.sidebar.title("🤖 Emotions Analyst")
st.sidebar.info("High-precision AI for analyzing Text, Images, and Files.")
mode = st.sidebar.radio("Choose Analysis Type:", ["Text Analysis", "Image Analysis", "Bulk File Analysis"])

st.title("🎭 Emotions Analyst")

# Function to map stars to Sentiment Labels
def map_sentiment(label):
    stars = int(label.split()[0])
    if stars <= 2: return "Negative ❌"
    if stars == 3: return "Neutral 😐"
    return "Positive ✅"

# 1. TEXT ANALYSIS
if mode == "Text Analysis":
    st.header("📝 Live Text Analysis")
    user_text = st.text_area("Enter your text (English or Arabic):", placeholder="Type here...")
    if st.button("Analyze Sentiment"):
        if user_text:
            result = text_ai(user_text)[0]
            sentiment = map_sentiment(result['label'])
            st.subheader(f"Result: {sentiment}")
            st.write(f"Confidence Score: {result['score']:.2f}")
        else:
            st.warning("Please enter some text first.")

# 2. IMAGE ANALYSIS
elif mode == "Image Analysis":
    st.header("🖼️ Image Emotion Recognition")
    img_file = st.file_uploader("Upload a photo...", type=['jpg', 'png', 'jpeg'])
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", width=400)
        with st.spinner("Analyzing Image..."):
            res = img_ai(img)
            st.success(f"Detected Emotion: **{res[0]['label']}**")

# 3. BULK FILE ANALYSIS
elif mode == "Bulk File Analysis":
    st.header("📁 Bulk Data Analysis (Majority Vote)")
    data_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    
    if data_file:
        df = pd.read_csv(data_file) if data_file.name.endswith('csv') else pd.read_excel(data_file)
        column = st.selectbox("Select the column containing text:", df.columns)
        
        if st.button("Start Bulk Analysis"):
            with st.spinner("Processing large dataset..."):
                # Apply AI analysis
                df['Raw_Label'] = df[column].apply(lambda x: text_ai(str(x))[0]['label'])
                df['Sentiment'] = df['Raw_Label'].apply(map_sentiment)
                
                # Statistics
                stats = df['Sentiment'].value_counts(normalize=True) * 100
                majority = stats.idxmax()
                
                # Display Results
                st.divider()
                st.subheader(f"Main Result: The majority is {majority}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Percentage Distribution:")
                    st.table(stats.rename("Percentage %"))
                
                with col2:
                    st.write("### Visual Representation:")
                    st.bar_chart(stats)
                
                # Option to download results
                st.download_button("Download Analyzed Data", df.to_csv().encode('utf-8'), "analyzed_data.csv")
                
