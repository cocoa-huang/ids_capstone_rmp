# -*- coding: utf-8 -*-
# %%
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
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