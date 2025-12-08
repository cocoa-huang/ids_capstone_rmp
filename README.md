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
python src/preprocessing.py
```

## Preprocessing

The full preprocessing procedure is implemented in src/preprocess.py.

### 1. Column Name Mapping

Based on the official project specification, we manually assigned clear and descriptive column names to each dataset.

**Numerical dataset (8 columns)**
```text
rating, difficulty, rating_count, pepper,
will_retake_pct, online_count, male, female
```

**Qualitative dataset (3 columns)**
```text
major, university, state
```

**Tag dataset (20 columns)**

Each tag was standardized into snake_case for consistency:
```text
tough_grader, good_feedback, respected, lots_to_read,
participation_matters, dont_skip_class_or_will_fail, lots_of_homework,
inspirational, pop_quizzes, accessible, so_many_papers,
clear_grading, hilarious, test_heavy, graded_by_few_things,
amazing_lectures, caring, extra_credit, group_projects,
lecture_heavy
```

### 2. Dataset Loading and Merging

All three CSVs contain exactly 89,893 rows in the same order.
We load each CSV without headers and assign the mappings above:

```text
num = pd.read_csv("../data/rmpCapstoneNum.csv", header=None, names=num_cols)
qual = pd.read_csv("../data/rmpCapstoneQual.csv", header=None, names=qual_cols)
tags = pd.read_csv("../data/rmpCapstoneTags.csv", header=None, names=tag_cols)
```

We then merge horizontally:
```text
df = pd.concat([num, qual, tags], axis=1)
```

This produces one unified dataframe containing all 31 features.

### 3. Removing Invalid Rows (rating_count ≤ 0)

Several rows correspond to professors who received zero ratings on RateMyProfessor.
Because rating, difficulty, and tags become meaningless for such rows, we exclude them:
```text
df = df[df["rating_count"].fillna(0) > 0]
```

This ensures that every remaining row corresponds to a professor with at least one student evaluation.

### 4. Handling Missing Values

We apply minimal, principled imputations based on the semantics of each feature.

**Numerical**
```
online_count → fill missing values with 0   (no online ratings)
will_retake_pct → fill missing values with median
```

**Qualitative**
```
major / university / state → fill with "Unknown"
```

**Ratings**

Not imputed; missing entries are eliminated earlier by the rating_count filter.



**Gender (male/female)**

Created new column ```gender``` with classes ```female``` and ```male``` to denote rows exclusively with ```male == 1``` or ```female == 1```.

### 5. Normalizing Tag Counts

Raw tag counts scale strongly with the number of ratings and are not comparable across professors.
To correct this, we convert each tag into a per-rating frequency:
```text
df[col + "_norm"] = df[col] / df["rating_count"]
```

This produces 20 new normalized tag features (e.g., tough_grader_norm).

These normalized tag variables are the ones used in:

- Q4 (gender difference in tags)

- Q7–9 (regression modeling)
- Q10 (classification)

### 6. Clipping Rating & Difficulty

Ratings on RMP are defined on a 1–5 scale.
We enforce the valid domain:

```text
df["rating"] = df["rating"].clip(1, 5)
df["difficulty"] = df["difficulty"].clip(1, 5)
```

This protects downstream analyses from erroneous values. (After checking, all ratings and difficulties fall within the 1-5 range.)

### 7. Final Output

We save the processed file:

```
rmpCapstoneProcessed.csv             
```
(saved back to /data)

The final dataset after preprocessing contains 51 columns and 70004 rows.


Each contains standardized variable names, normalized tag features, no missing ratings, and consistent formatting.
All further analyses (Questions 1–10) use this processed dataset.
