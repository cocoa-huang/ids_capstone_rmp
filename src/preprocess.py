import pandas as pd
import numpy as np

def load_and_preprocess():

    # --------- 1. Define column name mapping ---------
    num_cols = [
        "rating", "difficulty", "rating_count", "pepper",
        "will_retake_pct", "online_count", "male", "female"
    ]

    qual_cols = ["major", "university", "state"]

    tag_cols = [
        "tough_grader", "good_feedback", "respected", "lots_to_read",
        "participation_matters", "dont_skip_class_or_will_fail", "lots_of_homework",
        "inspirational", "pop_quizzes", "accessible", "so_many_papers",
        "clear_grading", "hilarious", "test_heavy", "graded_by_few_things",
        "amazing_lectures", "caring", "extra_credit", "group_projects",
        "lecture_heavy"
    ]

    # --------- 2. Load CSVs (no header in raw files) ---------
    num = pd.read_csv("../data/rmpCapstoneNum.csv", header=None, names=num_cols)
    qual = pd.read_csv("../data/rmpCapstoneQual.csv", header=None, names=qual_cols)
    tags = pd.read_csv("../data/rmpCapstoneTags.csv", header=None, names=tag_cols)

    # --------- 3. Merge (all files have same row order) ---------
    df = pd.concat([num, qual, tags], axis=1)

    # --------- 4. Remove rows with 0 or missing rating_count ---------
    df = df[df["rating_count"].fillna(0) > 0]

    # --------- 5. Handle missing values ---------
    # numerical
    df["online_count"] = df["online_count"].fillna(0)

    # qualitative
    df["major"] = df["major"].fillna("Unknown")
    df["university"] = df["university"].fillna("Unknown")
    df["state"] = df["state"].fillna("Unknown")

    # rating & difficulty already removed through rating_count filter

    # --------- 6. Normalize tags ---------
    for col in tag_cols:
        df[col + "_norm"] = df[col] / df["rating_count"]

    
    # --------- 7. Clip rating and difficulty to valid range 1–5 ---------
    # Before clipping stats
    rating_below = (df["rating"] < 1).sum()
    rating_above = (df["rating"] > 5).sum()
    difficulty_below = (df["difficulty"] < 1).sum()
    difficulty_above = (df["difficulty"] > 5).sum()

    print(f"[Clipping] rating < 1: {rating_below}")
    print(f"[Clipping] rating > 5: {rating_above}")
    print(f"[Clipping] difficulty < 1: {difficulty_below}")
    print(f"[Clipping] difficulty > 5: {difficulty_above}")

    # Apply clipping
    df["rating"] = df["rating"].clip(1, 5)
    df["difficulty"] = df["difficulty"].clip(1, 5)

    # After clipping, verify everything is within range
    assert df["rating"].between(1, 5).all(), "Rating contains values outside 1–5 even after clipping!"
    assert df["difficulty"].between(1, 5).all(), "Difficulty contains values outside 1–5 even after clipping!"

    print("[Clipping] Rating & difficulty clipping completed.")


    return df

if __name__ == "__main__":
    df = load_and_preprocess()
    print("Final dataset shape:", df.shape)
    print("Columns:", df.columns)

    df.to_csv("../data/rmpCapstoneProcessed.csv", index=False)