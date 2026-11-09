from unittest import result

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import plotly_express as px

print("all libaries imported")
df= pd.read_csv("social_media_engagement_5000.csv")

print(df.head())

print(df.shape)

print(df.columns)

print(df.dtypes)
df.info()

df["posted_at"] = pd.to_datetime(
    df["posted_at"],
    format="%d-%m-%Y"
)

print(df["posted_at"].head())

likes_array = df["likes"].to_numpy()

print(likes_array[:10])

print("Average Likes:", np.nanmean(likes_array))
print("Maximum Likes:", np.nanmax(likes_array))
print("Minimum Likes:", np.nanmin(likes_array))


# Check missing values in each column

print(df.isnull().sum())

print(df.isna().sum())


# Fill missing numerical values using median

df["age"] = df["age"].fillna(df["age"].median())

df["likes"] = df["likes"].fillna(df["likes"].median())

df["comments"] = df["comments"].fillna(df["comments"].median())

df["shares"] = df["shares"].fillna(df["shares"].median())

print("Missing numerical values handled successfully!")


df["likes"].median()

df["likes"].fillna(...)


# Fill missing categorical values using mode

df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

df["sentiment"] = df["sentiment"].fillna(df["sentiment"].mode()[0])

print("Missing categorical values handled successfully!")


df_without_missing = df.dropna()

print(df_without_missing.shape)

sample = pd.Series([10, None, 30, None, 50])

print(sample.ffill())

print(sample.bfill())

print(df.isnull().sum())

# Count duplicate rows

print("Duplicate records:", df.duplicated().sum())

# Remove duplicate records

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Standardize gender values

df["gender"] = df["gender"].str.strip().str.lower()

# Standardize sentiment values

df["sentiment"] = df["sentiment"].str.strip().str.lower()

print(df["gender"].unique())

print(df["sentiment"].unique())

# Check numerical values

print(df[["likes", "comments", "shares"]].describe())

print("Negative likes:", (df["likes"] < 0).sum())

print("Negative comments:", (df["comments"] < 0).sum())

print("Negative shares:", (df["shares"] < 0).sum())


# Count hashtags in each record

df["hashtag_count"] = df["hashtags"].fillna("").apply(
    lambda x: len(re.findall(r"#\w+", x))
)

print(df[["hashtags", "hashtag_count"]].head(10))

print(df["sentiment"].value_counts())


print("Dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nGender categories:")
print(df["gender"].unique())

print("\nSentiment categories:")
print(df["sentiment"].unique())

print("\nHashtag count:")
print(df[["hashtags", "hashtag_count"]].head())


# Display first five records

print(df.head())

print(df.head(10))

# Display last five records

print(df.tail())

# Display number of rows and columns

print("Dataset Shape:", df.shape)

# Display all column names

print(df.columns)

# Display complete dataset information

df.info()

# Display data types of all columns

print(df.dtypes)

# Generate summary statistics

print(df.describe())

print(df.describe(include="object"))

# Count each post type

print(df["post_type"].value_counts())

print(df["gender"].value_counts())

print(df["country"].value_counts())

print(df["sentiment"].value_counts())

print(df["device_type"].value_counts())

print(df["post_type"].unique())

print(df["gender"].unique())

print(df["sentiment"].unique())

print(df["device_type"].unique())

print("Unique post types:", df["post_type"].nunique())

print("Unique countries:", df["country"].nunique())

print("Unique categories:", df["post_category"].nunique())


# Select numerical columns

numeric_df = df.select_dtypes(include="number")

# Calculate correlation

correlation_matrix = numeric_df.corr()

print(correlation_matrix)
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()

# Calculate average likes for each post type

avg_likes = df.groupby("post_type")["likes"].mean()

print(avg_likes)

# Calculate average impressions by country

country_impressions = df.groupby("country")["impression_count"].mean()

# Sort from highest to lowest

country_impressions = country_impressions.sort_values(
    ascending=False
)

print(country_impressions)

# Calculate average engagement rate by sentiment

sentiment_engagement = df.groupby("sentiment")["engagement_rate"].mean()

print(sentiment_engagement)


# TASK 3: DATA EXPLORATION USING PANDAS

# 1. Display first five records
print("First Five Records")
print(df.head())

# 2. Display last five records
print("\nLast Five Records")
print(df.tail())

# 3. Dataset shape
print("\nDataset Shape")
print(df.shape)

# 4. Column names
print("\nColumn Names")
print(df.columns)

# 5. Dataset information
print("\nDataset Information")
df.info()

# 6. Data types
print("\nData Types")
print(df.dtypes)

# 7. Summary statistics
print("\nSummary Statistics")
print(df.describe())

# 8. Categorical distributions
print("\nPost Type Distribution")
print(df["post_type"].value_counts())

# 9. Unique values
print("\nUnique Post Types")
print(df["post_type"].unique())

# 10. Number of unique values
print("\nUnique Countries Count")
print(df["country"].nunique())

# 11. Correlation matrix
print("\nCorrelation Matrix")
print(df.select_dtypes(include="number").corr())

# 12. Average likes by post type
print("\nAverage Likes by Post Type")
print(df.groupby("post_type")["likes"].mean())

# 13. Average impressions by country
print("\nAverage Impressions by Country")
print(df.groupby("country")["impression_count"].mean().sort_values(ascending=False))


# Create a new column called engagement_score

df["engagement_score"] = (
    df["likes"] +
    df["comments"] +
    df["shares"]
)

# Display the result

print(df[["likes", "comments", "shares", "engagement_score"]].head(10))


# Create log-transformed columns

df["log_likes"] = np.log1p(df["likes"])

df["log_comments"] = np.log1p(df["comments"])

df["log_shares"] = np.log1p(df["shares"])

df["log_impressions"] = np.log1p(df["impression_count"])

# Display original and transformed values

print(df[[
    "likes",
    "log_likes",
    "comments",
    "log_comments",
    "impression_count",
    "log_impressions"
]].head())

# Display hashtag information

print(df[["hashtags", "hashtag_count"]].head(10))
# Group by post type and calculate average engagement

post_type_analysis = df.groupby("post_type")["engagement_score"].mean()

print(post_type_analysis)

post_type_analysis = post_type_analysis.sort_values(
    ascending=False
)

print(post_type_analysis)


# Group data by country

country_analysis = df.groupby("country").agg(
    average_engagement=("engagement_score", "mean"),
    average_engagement_rate=("engagement_rate", "mean"),
    average_impressions=("impression_count", "mean")
)

# Sort by average engagement score

country_analysis = country_analysis.sort_values(
    by="average_engagement",
    ascending=False
)

print(country_analysis)

average_engagement=("engagement_score", "mean")


# Group data by sentiment

sentiment_analysis = df.groupby("sentiment").agg(
    average_likes=("likes", "mean"),
    average_comments=("comments", "mean"),
    average_shares=("shares", "mean"),
    average_engagement=("engagement_score", "mean"),
    average_engagement_rate=("engagement_rate", "mean")
)

print(sentiment_analysis)

students = pd.DataFrame({
    "user_id": [1, 2, 3],
    "name": ["Anu", "Basil", "Arun"]
})

scores = pd.DataFrame({
    "user_id": [1, 2, 3],
    "likes": [100, 200, 150]
})

# Combine using the common user_id column

merged_data = pd.merge(students, scores, on="user_id")

print(merged_data)

left = pd.DataFrame(
    {"name": ["Anu", "Basil"]},
    index=[1, 2]
)

right = pd.DataFrame(
    {"likes": [100, 200]},
    index=[1, 2]
)

joined_data = left.join(right)

print(joined_data)


# TASK 4: DATA WRANGLING

# 1. Create engagement score
df["engagement_score"] = (
    df["likes"] + df["comments"] + df["shares"]
)

# 2. Create log-transformed metrics
df["log_likes"] = np.log1p(df["likes"])
df["log_comments"] = np.log1p(df["comments"])
df["log_shares"] = np.log1p(df["shares"])
df["log_impressions"] = np.log1p(df["impression_count"])

# 3. Display hashtag count
print("Hashtag Count")
print(df[["hashtags", "hashtag_count"]].head())

# 4. Group by post type
print("\nAverage Engagement by Post Type")
print(
    df.groupby("post_type")["engagement_score"]
    .mean()
    .sort_values(ascending=False)
)

# 5. Group by country
print("\nCountry Analysis")
print(
    df.groupby("country").agg(
        average_engagement=("engagement_score", "mean"),
        average_engagement_rate=("engagement_rate", "mean"),
        average_impressions=("impression_count", "mean")
    ).sort_values(
        by="average_engagement",
        ascending=False
    )
)

# 6. Group by sentiment
print("\nSentiment Analysis")
print(
    df.groupby("sentiment").agg(
        average_likes=("likes", "mean"),
        average_comments=("comments", "mean"),
        average_shares=("shares", "mean"),
        average_engagement=("engagement_score", "mean"),
        average_engagement_rate=("engagement_rate", "mean")
    )
)

# 7. Display final dataset shape
print("\nFinal Dataset Shape:", df.shape)

import pandas as pd
import numpy as np

numeric_cols = [
    "likes",
    "comments",
    "shares",
    "watch_time_sec",
    "engagement_rate",
    "follower_count"
]

df[numeric_cols].head()
df[numeric_cols].mean()
df[numeric_cols].median()
df[numeric_cols].mode()
df[numeric_cols].std()
df[numeric_cols].var()
df[numeric_cols].quantile([0.25, 0.50, 0.75])

print("Skewness:")
print(df[numeric_cols].skew())

print("\nKurtosis:")
print(df[numeric_cols].kurt())

statistics_summary = pd.DataFrame({
    "Mean": df[numeric_cols].mean(),
    "Median": df[numeric_cols].median(),
    "Standard Deviation": df[numeric_cols].std(),
    "Variance": df[numeric_cols].var()
})

statistics_summary

df[numeric_cols].count()




