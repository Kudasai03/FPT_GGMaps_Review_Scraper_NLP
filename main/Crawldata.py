from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

# Hàm lấy thông tin 
def get_review_summary(result_set):
    rev_dict = {'Review Name': [], 'Review Text': [], 'Review Time': [], 'Rating': []}
    for result in result_set:
        review_name = result.find(class_='d4r55').text if result.find(class_='d4r55') else 'N/A'
        review_text_elements = result.find_all('span', class_='wiI7pd')
        review_text = ' '.join([text_element.get_text(separator=' ', strip=True).replace('\n', ' ') 
                                for text_element in review_text_elements]) if review_text_elements else 'N/A'
        review_time = result.find('span', class_='rsqaWe').text if result.find('span', class_='rsqaWe') else 'N/A'
        star_rating = result.find('span', class_='kvMYJc')['aria-label'] if result.find('span', class_='kvMYJc') else 'N/A'
        rev_dict['Review Name'].append(review_name)
        rev_dict['Review Text'].append(review_text)
        rev_dict['Review Time'].append(review_time)
        rev_dict['Rating'].append(star_rating)
    return pd.DataFrame(rev_dict)

# Hàm nhấn vào mục "Bài đánh giá" 
def click_show_reviews(driver):
    try:
        show_reviews_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH,
            '//*[@id="QA0Szd"]/div/div/div[1]/div[3]/div/div[1]/div/div/div[2]/div/div/div/button[2]/div[2]'))
        )
        show_reviews_button.click()
    except Exception as e:
        print(f"Error clicking on reviews: {e}")

# Hàm xem đầy đủ đánh giá bằng cách bấm vào nút "Thêm" (nếu có)
def expand_reviews(driver):
    try:
        more_buttons = driver.find_elements(By.CLASS_NAME, 'w8nwRe')
        for button in more_buttons:
            driver.execute_script("arguments[0].click();", button)
            time.sleep(1)
    except Exception as e:
        print(f"Error expanding reviews: {e}")

# Hàm cuộc tới cuối cùng của trang để tải hết đánh giá
def scroll_to_bottom(driver, scrollable_div, pause_time=3):
    last_height = driver.execute_script("return arguments[0].scrollHeight;", scrollable_div)
    while True:
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scrollable_div)
        time.sleep(pause_time)
        new_height = driver.execute_script("return arguments[0].scrollHeight;", scrollable_div)
        if new_height == last_height:
            break
        last_height = new_height

# Hàm thu thập đánh giá từ URL
def collect_reviews_from_url(driver, url):
    driver.get(url)
    time.sleep(5)  
    click_show_reviews(driver)  # Nhấn vào nút hiển thị đánh giá

    # Xác định phần tử cuộn
    scrollable_div = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH,
         '//*[@id="QA0Szd"]/div/div/div[1]/div[3]/div/div[1]/div/div/div[3]')))

    # Cuộn để tải toàn bộ đánh giá
    scroll_to_bottom(driver, scrollable_div)
    expand_reviews(driver)  # Xem đầy đủ đánh giá (nếu có)

    # Lấy toàn bộ nội dung sau khi cuộn hết tất cả đánh giá
    response = BeautifulSoup(driver.page_source, 'html.parser')
    review_items = response.find_all('div', class_='jftiEf')
    return get_review_summary(review_items)  

# # Hàm lưu dữ liệu vào CSV
# def save_to_csv(df, file_path='datareview.csv'):
#     if os.path.isfile(file_path):
#         existing_df = pd.read_csv(file_path)
#         combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
#     else:
#         combined_df = df.drop_duplicates()

#     # Lưu dữ liệu vào file CSV
#     combined_df.to_csv(file_path, index=False)
#     print(f"Saved {len(combined_df)} reviews to {file_path}")

# # Nhập danh sách URL từ người dùng trước khi mở trình duyệt
# input_urls = input("Nhập các URL cần thu thập dữ liệu (ngăn cách bởi chữ 'and'): ").strip().split('and')
# urls = [url.strip() for url in input_urls if url.strip()] 

# if not urls:
#     print("Không có URL hợp lệ nào được nhập. Kết thúc chương trình.")
# else:

#     driver_path = r"google-review-scraper-main\chromedriver.exe"

#     service = Service(executable_path=driver_path)
#     chrome_options = Options()
#     chrome_options.add_argument("--log-level=3") 
#     chrome_options.add_argument("--start-maximized")

#     # Khởi tạo driver với Service và Options
#     driver = webdriver.Chrome(service=service, options=chrome_options)

#     # Tạo DataFrame để lưu trữ toàn bộ đánh giá
#     final_df = pd.DataFrame()

#     # Duyệt qua từng URL để thu thập dữ liệu
#     for url in urls:
#         try:
#             df = collect_reviews_from_url(driver, url)
#             final_df = pd.concat([final_df, df], ignore_index=True)
#         except Exception as e:
#             print(f"Error processing URL {url}: {e}")

#     # Lưu dữ liệu vào file CSV
#     save_to_csv(final_df)
#     driver.quit()
# python -m streamlit run d:/Toan/Scrape/google-review-scraper-main/webapp.py