# -*- coding: utf-8 -*-
# %%
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene
import numpy as np
import random


SEED = 11269807   # Bruce's N-number
np.random.seed(SEED)
random.seed(SEED)
# Load processed dataset
df = pd.read_csv("../data/rmpCapstoneProcessed.csv")

print("Loaded df shape:", df.shape)
print(df.columns)



# %%
def get_gender_df(df):
    """
    Return a filtered dataframe where gender is clearly identified
    as 'male' or 'female'. Rows labeled 'unclear' are removed.
    """
    df_clean = df[df["gender"].isin(["male", "female"])].copy()
    return df_clean



def plot_q1_distribution(df):
    df_gender = df[df["gender"].isin(["male", "female"])]

    plt.figure(figsize=(8,5))
    sns.histplot(data=df_gender, x="rating", hue="gender",
                 bins=20, alpha=0.4, stat="count",
                 palette={"female": "#1f77b4", "male": "#ff7f0e"})
    plt.title("Rating Histogram by Gender")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    





# %%
# Question 1 - Gender Bias in Averaged Ratings


df_gender = get_gender_df(df)

male_ratings = df_gender[df_gender["gender"] == "male"]["rating"]
female_ratings = df_gender[df_gender["gender"] == "female"]["rating"]

print("Sample sizes:")
print("Male:", len(male_ratings))
print("Female:", len(female_ratings))

# Welch t-test
t_stat, p_val = ttest_ind(male_ratings, female_ratings, equal_var=False)

print("\nWelch t-test result:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

print("\nMean ratings:")
print("Male mean:", male_ratings.mean())
print("Female mean:", female_ratings.mean())


plot_q1_distribution(df_gender)


# %%
# Question 2 - Gender variance difference (spread/dispersion)



df_gender = df[df["gender"].isin(["male", "female"])]

male_ratings = df_gender[df_gender["gender"] == "male"]["rating"]
female_ratings = df_gender[df_gender["gender"] == "female"]["rating"]

# Variances
male_var = male_ratings.var()
female_var = female_ratings.var()

print("Male variance:", male_var)
print("Female variance:", female_var)

# Levene test (recommended)
stat, p_val = levene(male_ratings, female_ratings, center='mean')

print("\nLevene test result:")
print("Statistic:", stat)
print("p-value:", p_val)


plt.figure(figsize=(6, 5))
sns.boxplot(data=df_gender, x="gender", y="rating",
            palette={"female": "#1f77b4", "male": "#ff7f0e"})
plt.title("Rating Spread by Gender")
plt.tight_layout()
plt.savefig("../figures/Q2_boxplot.png", dpi=300)
plt.show()





# %%
# Question 3 — Effect sizes + 95% CI

df_gender = df[df["gender"].isin(["male", "female"])]

male = df_gender[df_gender["gender"] == "male"]["rating"].values
female = df_gender[df_gender["gender"] == "female"]["rating"].values

# --- Effect size 1: Cohen's d (mean difference) ---
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)

d = cohens_d(male, female)
print("Cohen's d:", d)

# Bootstrap CI for Cohen's d
N_BOOT = 3000
d_samples = []

for _ in range(N_BOOT):
    male_sample = np.random.choice(male, size=len(male), replace=True)
    female_sample = np.random.choice(female, size=len(female), replace=True)
    d_samples.append(cohens_d(male_sample, female_sample))

d_lower = np.percentile(d_samples, 2.5)
d_upper = np.percentile(d_samples, 97.5)

print(f"95% CI for Cohen's d: [{d_lower}, {d_upper}]")


# --- Effect size 2: Variance ratio ---
var_ratio = np.var(male, ddof=1) / np.var(female, ddof=1)
print("\nVariance ratio (male/female):", var_ratio)

# Bootstrap variance ratio CI
vr_samples = []
for _ in range(N_BOOT):
    male_sample = np.random.choice(male, size=len(male), replace=True)
    female_sample = np.random.choice(female, size=len(female), replace=True)
    vr_samples.append(
        np.var(male_sample, ddof=1) / np.var(female_sample, ddof=1)
    )

vr_lower = np.percentile(vr_samples, 2.5)
vr_upper = np.percentile(vr_samples, 97.5)

print(f"95% CI for variance ratio: [{vr_lower}, {vr_upper}]")




# %%
# Question 4 — Gender differences in tags


df_gender = get_gender_df(df)

tag_cols = [
    "tough_grader", "good_feedback", "respected", "lots_to_read",
    "participation_matters", "dont_skip_class_or_will_fail", "lots_of_homework",
    "inspirational", "pop_quizzes", "accessible", "so_many_papers",
    "clear_grading", "hilarious", "test_heavy", "graded_by_few_things",
    "amazing_lectures", "caring", "extra_credit", "group_projects",
    "lecture_heavy"
]

from scipy.stats import mannwhitneyu

results = []

for tag in tag_cols:
    col = tag + "_norm"

    male_vals = df_gender[df_gender["gender"] == "male"][col]
    female_vals = df_gender[df_gender["gender"] == "female"][col]

    stat, p_val = mannwhitneyu(male_vals, female_vals, alternative="two-sided")

    results.append({
        "tag": tag,
        "male_median": np.median(male_vals),
        "female_median": np.median(female_vals),
        "diff_median": np.median(male_vals) - np.median(female_vals),
        "p_value": p_val
    })

tag_results = pd.DataFrame(results).sort_values("p_value")


print("=== Top 5 most gendered tags (smallest p-values) ===")
print(tag_results.head(5))

print("\n=== Top 5 least gendered tags (largest p-values) ===")
print(tag_results.tail(5))

# %% 
# Question 5




male_diff = df_gender[df_gender["gender"] == "male"]["difficulty"].values
female_diff = df_gender[df_gender["gender"] == "female"]["difficulty"].values

t_stat, p_val = ttest_ind(male_diff, female_diff, equal_var=False)

print("=== Q5: Gender difference in difficulty ===")
print(f"Male mean difficulty:   {male_diff.mean():.4f}")
print(f"Female mean difficulty: {female_diff.mean():.4f}")
print(f"Mean difference (male - female): {male_diff.mean() - female_diff.mean():.4f}")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Question 6 — Effect size (Cohen's d)
def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)

d = cohens_d(male_diff, female_diff)
print("\n=== Q6: Effect size (Cohen's d) ===")
print("Cohen's d:", d)

# Bootstrap CI
N_BOOT = 3000
d_samples = []

for _ in range(N_BOOT):
    male_sample = np.random.choice(male_diff, size=len(male_diff), replace=True)
    female_sample = np.random.choice(female_diff, size=len(female_diff), replace=True)
    d_samples.append(cohens_d(male_sample, female_sample))

d_lower  = np.percentile(d_samples, 2.5)
d_upper  = np.percentile(d_samples, 97.5)

print(f"95% CI for d: [{d_lower:.4f}, {d_upper:.4f}]")




