# Thesis Personal Assignment - Google Review Analyst

## Objective
The primary goal of this assignment is to develop a robust framework for collecting, processing, and analyzing Google Reviews. This framework will enable us to uncover trends, patterns, and sentiments that can inform business strategies and enhance customer engagement.

This repository contains the code and report for my final project at UEH. For 3 machine learning models including Naive Bayes [[Mosteller and Wallace (1964)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500849)], Maxent [[Berger et al. (1996)](https://dl.acm.org/doi/10.5555/234285.234289)] and XGBoost [[Chen and Guestrin (2016)](https://dl.acm.org/doi/10.1145/2939672.2939785)], we create a custom UI that allows users to directly enter the review and select a machine learning model of their choice. We deploy the PhoBert model [[LDat Quoc Nguyen and Anh Tuan Nguyen (2020)](https://arxiv.org/abs/2003.00744)]

## Methodology

Utilizing Selenium and BeautifulSoup for web scraping, this project captures data from Google Reviews, including review names, texts, ratings, and timestamps. 

## Key Features

- Automated web scraping of Google Reviews
- Data storage in CSV format for easy accessibility
- Comprehensive analysis of customer sentiments
(Acknowledgments: A special thanks to Behitek (https://github.com/behitek) for contributions to preprocessing Vietnamese text, which significantly aided this project.)
- Visualization of review trends and patterns
- Training sentiment prediction models
- Developing an application interface

![Screenshot 2024-10-27 012424](https://github.com/user-attachments/assets/2524fed0-7003-4398-9c4a-db7e99be23da)

## Conclusion

The findings from this project will contribute to a deeper understanding of consumer sentiment in the digital marketplace, providing valuable insights for businesses looking to improve their services and customer relations.
![Screenshot 2024-10-27 012557](https://github.com/user-attachments/assets/e1b02824-5717-43c6-a73c-6f0121331850)

## Implementation

To reproduce the webapp,  you need to clone the repository first:

```
git clone https://github.com/Kudasai03/FPT_GGMaps_Review_Scraper_NLP
```
Then Please Download the VnCoreNLP at (https://github.com/vncorenlp/VnCoreNLP) and then paste is to folder VnCoreNLP to setup it
- `Java 1.8+` (Prerequisite)
- File  `VnCoreNLP-1.2.jar` (27MB) and folder `models` (115MB) are placed in the same working folder.
- `Python 3.6+` if using [a Python wrapper of VnCoreNLP](https://github.com/thelinhbkhn2014/VnCoreNLP_Wrapper). To install this wrapper, users have to run the following command:

    `$ pip3 install py_vncorenlp` 
    
    _A special thanks goes to [Linh The Nguyen](https://github.com/thelinhbkhn2014) for creating this wrapper!_
Install the requirment.txt file and then open the terminal then run the code
```
python -m streamlit run d:/Toan/Scrape/google-review-scraper-main/webapp.py
```
---
