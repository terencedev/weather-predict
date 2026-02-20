# Global Weather Streamlit App

This app builds an interactive dashboard from the Kaggle **Global Weather Repository** CSV:

- Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data

## 1) Get the CSV

Download the dataset from Kaggle and place the CSV file at:

`data/GlobalWeatherRepository.csv`

(You can also upload the CSV directly from the app sidebar.)

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Run

```bash
streamlit run app.py
```

## Features

- Local CSV path or file upload source
- Local CSV can be loaded by browsing a file or entering a path
- AWS endpoint URL source (public or presigned S3 CSV URL)
- Private S3 object support via `s3://bucket/key.csv` using AWS credentials
- Filters for country, condition, and temperature
- KPI cards for rows, countries, average temperature, average humidity
- Charts for country temperature and weather condition mix
- Raw data table
- PM2.5 prediction tab with model metrics (MAE/R2) and one-row prediction form

## AWS endpoint usage

- In the sidebar, choose `Use AWS endpoint URL`, then paste your CSV link.
- Works with public S3 URLs, presigned URLs, or `s3://bucket/key.csv`.
- Optional: set `WEATHER_CSV_URL` env var to prefill the URL input.
- For `s3://...`, ensure your environment has AWS credentials with `s3:GetObject`.
