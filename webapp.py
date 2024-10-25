import os
from main.Crawldata import *  
from main.Process import *
from streamlit_option_menu import option_menu
import streamlit as st
from pathlib import Path
import time
import base64
import pandas as pd
import plotly.express as px
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

st.set_page_config(page_title='Twan-final' ,layout="wide")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# img_path = Path("Webapp_files/image.jpg").resolve()
# img = get_img_as_base64(img_path)
# Đảm bảo rằng đường dẫn là từ gốc dự án
img_path = Path(__file__).parent / "Webapp_files/image.jpg"

# Đưa đường dẫn vào hàm
img = get_img_as_base64(img_path)
# Lấy đường dẫn của chromedriver.exe theo file đang chạy
chrome_driver_path = Path(__file__).parent / "chromedriver.exe"

# img = get_img_as_base64(r"D:\Toan\Scrape\google-review-scraper-main\Webapp_files\image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1569428226604-5e24b547bcb4");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

def save_to_csv(df, file_path= Path(__file__).parent / 'Webapp_files/Reviews_crawled.csv'):
    if os.path.isfile(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
    else:
        combined_df = df.drop_duplicates()

    combined_df.to_csv(file_path, index=False)
    print(f"Saved {len(combined_df)} reviews to {file_path}")

@st.cache_resource
def load_sentiment_model():
    tokenizer_sentiment = AutoTokenizer.from_pretrained('mr4/phobert-base-vi-sentiment-analysis')
    model_sentiment = AutoModelForSequenceClassification.from_pretrained("TwanNDT/PhoBert-TelecomReview")
    return tokenizer_sentiment, model_sentiment

tokenizer_sentiment, model_sentiment = load_sentiment_model()

def main():
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    if 'final_df' not in st.session_state:
        st.session_state.final_df = pd.DataFrame()  # Dữ liệu chưa có
    if 'sentiment_counts' not in st.session_state:
        st.session_state.sentiment_counts = None  # Biểu đồ chưa có

    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Menu", "Phân tích đánh giá"],
            icons=["house", "archive", "bar-chart-fill"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Menu":
        st.title("Giới thiệu")
        st.write("Chào mừng bạn đến với ứng dụng phân tích đánh giá của khách hàng về FPT Telecom!")

    elif selected == "Phân tích đánh giá":
        st.title("Trích xuất ý chính của comment")

        # Hiển thị dữ liệu nếu có
        if not st.session_state.final_df.empty:
            st.write('**Bảng Comment thu thập được:**')
            st.dataframe(st.session_state.final_df, use_container_width=True)

            # Hiển thị biểu đồ nếu có
            if st.session_state.sentiment_counts is not None:
                st.write("Biểu đồ phân tích cảm xúc:")
                fig = px.bar(st.session_state.sentiment_counts, 
                             x=st.session_state.sentiment_counts.index, 
                             y=st.session_state.sentiment_counts.values, 
                             labels={'x': 'Loại cảm xúc', 'y': 'Số lượng'},
                             color=st.session_state.sentiment_counts.index,  
                             title="Biểu đồ biểu diễn số lượng của từng loại cảm xúc của dữ liệu")
                st.plotly_chart(fig)

        url_input = st.text_input("Nhập vào URL của chi nhánh cần đánh giá (ngăn cách bởi 'and'):") 

        if st.button("Lấy đánh giá"):
            if url_input: 
                urls = [url.strip() for url in url_input.split('and')]
                driver_path = Path(__file__).parent / "chromedriver.exe"
                service = Service(executable_path=str(driver_path))  # Chuyển đường dẫn thành chuỗi
                chrome_options = Options()
                chrome_options.add_argument("--headless")  
                chrome_options.add_argument("--log-level=3") 
                chrome_options.add_argument("--start-maximized")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument('--disable-software-rasterizer')
                
                driver = webdriver.Chrome(service=service, options=chrome_options)

                final_df = pd.DataFrame()  # Biến tạm để lưu trữ dữ liệu trong lần này
                with st.spinner('Đang lấy đánh giá...'): 
                    for url in urls:
                        try:
                            df = collect_reviews_from_url(driver, url)
                            final_df = pd.concat([final_df, df], ignore_index=True)
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý URL {url}: {e}")

                save_to_csv(final_df)
                driver.quit()

                # Lưu vào session_state
                st.session_state.final_df = final_df

                st.write('**Bảng Comment thu thập được:**')
                st.dataframe(st.session_state.final_df, use_container_width=True)

                with st.container():
                    st.header("Phân tích cảm xúc")
                    with st.spinner('Đang phân tích cảm xúc...'):
                        st.session_state.final_df['Processed Review Text'] = st.session_state.final_df['Review Text'].apply(preprocess_text)
                        
                        # Phân tích cảm xúc
                        def analyze_sentiment(text):
                            inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True)
                            outputs = model_sentiment(**inputs)
                            logits = outputs.logits
                            sentiment = torch.argmax(logits, dim=-1).item()  
                            return "Tích cực" if sentiment == 1 else "Tiêu cực"
                        st.session_state.final_df['Sentiment'] = st.session_state.final_df['Processed Review Text'].apply(analyze_sentiment)

                        st.write('**Bảng đánh giá sau khi được xử lý:**')
                        st.dataframe(st.session_state.final_df[['Review Text', 'Processed Review Text','Rating', 'Sentiment']], use_container_width=True)

                        # Lưu số lượng cảm xúc để vẽ biểu đồ
                        st.session_state.sentiment_counts = st.session_state.final_df['Sentiment'].value_counts()
                    with st.container():
                        st.header("Phân tích cảm xúc")
                        with st.spinner('Đang vẽ biểu đồ'):
                            if not final_df.empty:
                                sentiment_counts = final_df['Sentiment'].value_counts()
                                fig = px.bar(sentiment_counts, 
                                            x=sentiment_counts.index, 
                                            y=sentiment_counts.values, 
                                            labels={'x': 'Loại cảm xúc', 'y': 'Số lượng'},
                                            color=sentiment_counts.index,  
                                            title="Biểu đồ biểu diễn số lượng của từng loại cảm xúc của dữ liệu")
                                st.plotly_chart(fig)
                            else:
                                st.write("Vui lòng trích xuất comment trước.") 
            else:
                st.warning("Vui lòng nhập URL hợp lệ.")

if __name__ == "__main__":
    main()
