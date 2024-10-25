
import py_vncorenlp

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=r'D:\Toan\Scrape\google-review-scraper-main\VnCoreNLP')
text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
output = rdrsegmenter.word_segment(text)
print(output)
# ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']