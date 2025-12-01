# APE Capstone Project

Capstone project for Introduction to Data Science (DS GA 1001) analyzing RateMyProfessor.com data to investigate gender bias in student evaluations and identify factors predicting professor ratings.

## Team

**Members:** Eric Huang, Bruce Zhang


## Project Structure
```
├── data/           # CSV files from RMP
├── src/            # Analysis scripts
├── figures/        # Generated plots
├── reports/        # PDF report
└── notebooks/      # Exploration (optional)
```

## Data

Three datasets with 89,893 professor records each:

**rmpCapstoneNum.csv** - Numerical features including average rating, difficulty, number of ratings, pepper status, would-take-again proportion, online ratings, and gender indicators.

**rmpCapstoneQual.csv** - Professor major, university, and state.

**rmpCapstoneTags.csv** - Raw counts for 20 teaching style tags (tough grader, inspirational, accessible, etc).

## Setup
```bash
pip install -r requirements.txt
python src/analysis.py
```
