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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import itertools
import collections
from nltk import bigrams


st.set_page_config(page_title='Twan-final', layout="wide", initial_sidebar_state="expanded", page_icon="🍵")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Đường dẫn hình ảnh
img_path = Path(__file__).parent / "Webapp_files/image.jpg"
img = get_img_as_base64(img_path)

# Đường dẫn chromedriver
chrome_driver_path = Path(__file__).parent / "chromedriver.exe"

# Thiết lập background cho trang
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1681735342773-d452708e87e7");
background-size: 175%;
background-position: top middel;
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

def save_to_csv(df, file_path=Path(__file__).parent / 'Webapp_files/Reviews_crawled.csv'):
    if os.path.isfile(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
    else:
        combined_df = df.drop_duplicates()

    combined_df.to_csv(file_path, index=False)
    st.success(f"Đã lưu {len(combined_df)} đánh giá vào {file_path}")

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained('mr4/phobert-base-vi-sentiment-analysis')
    model = AutoModelForSequenceClassification.from_pretrained("TwanNDT/phobert-fpt_telecomreview")
    return tokenizer, model

tokenizer_sentiment, model_sentiment = load_sentiment_model()

def main():
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Khởi tạo fig và plt với giá trị None trong session state
    if 'fig' not in st.session_state:
        st.session_state.fig = None
    if 'plt' not in st.session_state:
        st.session_state.plt = plt  # Khởi tạo plt trong session_state
    if 'final_df' not in st.session_state:
        st.session_state.final_df = pd.DataFrame()
    if 'sentiment_counts' not in st.session_state:
        st.session_state.sentiment_counts = None

    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Menu", "Phân tích đánh giá"],
            icons=["house", "database"],
            menu_icon="cpu_fill",
            default_index=0,
        )

    if selected == "Menu":
        st.title("Giới thiệu")
        st.write("Chào mừng bạn đến với ứng dụng phân tích đánh giá của khách hàng về FPT Telecom!")

    elif selected == "Phân tích đánh giá":
        st.title("**Thu thập và phân tích đánh giá**")

        if not st.session_state.final_df.empty:           
            with st.container():
                st.header("Bảng Comment thu thập được:")
            # st.write('**Bảng Comment thu thập được:**')
            st.dataframe(st.session_state.final_df, use_container_width=True)

            # Hiển thị biểu đồ nếu có
            if st.session_state.sentiment_counts is not None and st.session_state.fig is not None:
                st.header("Biểu đồ phân tích cảm xúc")
                st.plotly_chart(st.session_state.fig)
                st.header("Word Cloud từ các đánh giá")
                st.pyplot(st.session_state.plt)
                st.header("Trung bình Rating theo năm")
                st.plotly_chart(st.session_state.fig_rating)

        url_input = st.text_input("Nhập vào URL của chi nhánh cần đánh giá (ngăn cách bởi 'and'):") 

        if st.button("Lấy đánh giá"):
            if url_input: 
                urls = [url.strip() for url in url_input.split('and')]
                driver_path = Path(__file__).parent / "chromedriver.exe"
                service = Service(executable_path=str(driver_path))
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--log-level=3")
                chrome_options.add_argument("--start-maximized")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument('--disable-software-rasterizer')
                
                driver = webdriver.Chrome(service=service, options=chrome_options)
                final_df = pd.DataFrame()

                with st.spinner('Đang lấy đánh giá...'):
                    for url in urls:
                        try:
                            df = collect_reviews_from_url(driver, url)
                            final_df = pd.concat([final_df, df], ignore_index=True)
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý URL {url}: {e}")

                save_to_csv(final_df)
                driver.quit()

                st.session_state.final_df = final_df

                st.write('**Bảng Comment thu thập được:**')
                st.dataframe(st.session_state.final_df, use_container_width=True)

                with st.container():
                    st.header("Phân tích cảm xúc")
                    with st.spinner('Đang phân tích cảm xúc...'):
                        # Xử lý văn bản
                        st.session_state.final_df['Processed Review Text'] = st.session_state.final_df['Review Text'].apply(preprocess_text)
                        inputs = tokenizer_sentiment(st.session_state.final_df['Processed Review Text'].tolist(), return_tensors="pt", truncation=True, padding=True)
                        outputs = model_sentiment(**inputs)
                        logits = outputs.logits
                        sentiments = torch.argmax(logits, dim=-1).numpy()
                        sentiment_labels = ['Tiêu cực', 'Trung tính', 'Tích cực']
                        st.session_state.final_df['Sentiment'] = [sentiment_labels[sentiment] for sentiment in sentiments]
                        st.write('**Bảng đánh giá sau khi được xử lý:**')
                        st.dataframe(st.session_state.final_df[['Review Text', 'Processed Review Text', 'Rating', 'Sentiment']], use_container_width=True)
                        st.session_state.sentiment_counts = st.session_state.final_df['Sentiment'].value_counts()

                # Vẽ biểu đồ
                with st.container():
                    st.header("Visualize")
                    st.write('**Số lượng đánh giá theo cảm xúc:**')
                    with st.spinner('Đang vẽ biểu đồ'):
                        if not st.session_state.final_df.empty:
                            sentiment_counts = st.session_state.sentiment_counts
                            st.session_state.fig = px.bar(sentiment_counts, 
                                                           x=sentiment_counts.index, 
                                                           y=sentiment_counts.values, 
                                                           labels={'x': 'Loại cảm xúc', 'y': 'Số lượng'},
                                                           title="Biểu đồ biểu diễn số lượng của từng loại cảm xúc")
                            st.plotly_chart(st.session_state.fig)

                            col1, col2 = st.columns(2)
                                # Word Cloud cho cảm xúc tiêu cực
                            with col1:
                                # st.header("Word Cloud từ các đánh giá Tiêu cực")
                                st.write('**Word Cloud từ các đánh giá Tiêu cực:**')
                                negative_text = ' '.join(st.session_state.final_df[st.session_state.final_df['Sentiment'] == 'Tiêu cực']['Processed Review Text'])
                                if negative_text:  # Kiểm tra nếu có bình luận tiêu cực
                                    negative_wordcloud = WordCloud(width=400, height=300, background_color='white', max_words=100, colormap='viridis').generate(negative_text)
                                    plt.figure(figsize=(4, 3))
                                    plt.imshow(negative_wordcloud, interpolation='bilinear')
                                    plt.axis('off')
                                    st.pyplot(plt)
                                else:
                                    st.warning("Không có đánh giá tiêu cực nào để vẽ Word Cloud.")

                            # Word Cloud cho cảm xúc tích cực
                            with col2:
                                # st.header("Word Cloud từ các đánh giá Tích cực")
                                st.write('**Word Cloud từ các đánh giá Tích cực:**')
                                positive_text = ' '.join(st.session_state.final_df[st.session_state.final_df['Sentiment'] == 'Tích cực']['Processed Review Text'])
                                if positive_text:  # Kiểm tra nếu có bình luận tích cực
                                    positive_wordcloud = WordCloud(width=400, height=300, background_color='white', max_words=100, colormap='viridis').generate(positive_text)
                                    plt.figure(figsize=(4, 3))
                                    plt.imshow(positive_wordcloud, interpolation='bilinear')
                                    plt.axis('off')
                                    st.pyplot(plt)
                                else:
                                    st.warning("Không có đánh giá tích cực nào để vẽ Word Cloud.")
                                    
                            # Trung bình Rating theo năm
                            st.write("**Trung bình điểm đánh giá Rating theo thời gian**")
                            final_df['Year'] = final_df['Review Time'].apply(convert_to_date)
                            final_df['Rating'] = final_df['Rating'].apply(lambda x: int(x.replace(' sao', '')))
                            avg_rating_per_year = final_df.groupby('Year')['Rating'].mean().reset_index()

                            fig_rating = px.line(avg_rating_per_year, 
                                                  x='Year', 
                                                  y='Rating', 
                                                  title='Trung bình Rating theo năm', 
                                                  labels={'Rating': 'Trung bình Rating', 'Year': 'Năm'},
                                                  markers=True)
                            st.plotly_chart(fig_rating)
                        else:
                            st.warning("Vui lòng trích xuất comment trước.") 
            else:
                st.warning("Vui lòng nhập URL hợp lệ.")

if __name__ == "__main__":
    main()
