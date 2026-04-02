# ============================================================
# Project: User Activity Pattern Detection using Apache Spark
# Course: B.Tech Big Data Analytics
# Description: Detect user behavior patterns and anomalies
#              from e-commerce activity data using PySpark.
# ============================================================

# -----------------------------------------------
# STEP 1: Initialize Spark Session
# -----------------------------------------------
# A SparkSession is the entry point for PySpark programs.
# It allows us to create DataFrames and execute Spark SQL.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

spark = SparkSession.builder \
    .appName("UserActivityPatternDetection") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")  # Suppress unnecessary log messages

print("=" * 60)
print("  Spark Session Initialized Successfully")
print(f"  Spark Version: {spark.version}")
print("=" * 60)


# -----------------------------------------------
# STEP 2: Load Dataset into Spark
# -----------------------------------------------
# We read the CSV file into a Spark DataFrame.
# inferSchema=True automatically detects column data types.
# header=True uses the first row as column names.

df = spark.read.csv(
    "ecommerce_user_activity.csv",
    header=True,
    inferSchema=True
)

print("\n[STEP 2] Dataset Loaded Successfully")
print(f"  Total Records: {df.count()}")
print(f"  Total Columns: {len(df.columns)}")
print("\n  Schema:")
df.printSchema()

print("\n  Sample Data (5 rows):")
df.show(5, truncate=False)


# -----------------------------------------------
# STEP 3: Data Preprocessing
# -----------------------------------------------
# Preprocessing ensures data quality before analysis.
# 3a) Cast 'time' column to proper timestamp type.
# 3b) Drop any rows with null values (data cleaning).
# 3c) Add a 'date' column extracted from the timestamp.

# 3a) Convert time column to timestamp
df = df.withColumn("time", F.to_timestamp("time", "yyyy-MM-dd HH:mm:ss"))

# 3b) Remove null rows
before_count = df.count()
df = df.dropna()
after_count = df.count()
print(f"\n[STEP 3] Preprocessing Complete")
print(f"  Rows before cleaning : {before_count}")
print(f"  Rows after cleaning  : {after_count}")
print(f"  Rows removed         : {before_count - after_count}")

# 3c) Add derived columns for time-based analysis
df = df.withColumn("date", F.to_date("time")) \
       .withColumn("hour_extracted", F.hour("time")) \
       .withColumn("month", F.month("time"))

# Cache the DataFrame to speed up repeated operations
df.cache()

print("\n  Preprocessed Data (3 rows):")
df.show(3)


# -----------------------------------------------
# STEP 4: Exploratory Analysis
# -----------------------------------------------

print("\n" + "=" * 60)
print("  SECTION A: EXPLORATORY ANALYSIS")
print("=" * 60)

# -----------------------------------------------
# 4A: Most Common User Actions
# -----------------------------------------------
# Group all rows by 'action' and count occurrences.
# This tells us which activities (view/cart/purchase) happen most.

print("\n[4A] Most Common User Actions:")
action_counts = df.groupBy("action") \
                  .agg(F.count("*").alias("total_count")) \
                  .orderBy(F.desc("total_count"))
action_counts.show()

# Percentage share of each action
total_actions = df.count()
action_with_pct = action_counts.withColumn(
    "percentage",
    F.round((F.col("total_count") / total_actions) * 100, 2)
)
print("  Action Distribution with Percentage:")
action_with_pct.show()


# -----------------------------------------------
# 4B: Peak Activity Time (by Hour)
# -----------------------------------------------
# We extract the hour from the timestamp and count
# how many events occurred in each hour of the day.
# This reveals when users are most active.

print("[4B] Peak Activity Hours:")
hourly_activity = df.groupBy("hour") \
                    .agg(F.count("*").alias("activity_count")) \
                    .orderBy(F.desc("activity_count"))
hourly_activity.show()

peak_hour = hourly_activity.first()
print(f"  >> Peak Hour: {peak_hour['hour']}:00 with {peak_hour['activity_count']} events")

# Day-wise activity count
print("\n  Activity by Day of Week:")
day_activity = df.groupBy("day_of_week") \
                 .agg(F.count("*").alias("activity_count")) \
                 .orderBy(F.desc("activity_count"))
day_activity.show()


# -----------------------------------------------
# 4C: Most Active Users
# -----------------------------------------------
# Count total events per user to find the most active ones.
# A very high count per user may indicate bot-like behavior.

print("[4C] Most Active Users (Top 10):")
user_activity = df.groupBy("user_id") \
                  .agg(
                      F.count("*").alias("total_actions"),
                      F.countDistinct("session_id").alias("total_sessions"),
                      F.countDistinct("category").alias("categories_browsed")
                  ) \
                  .orderBy(F.desc("total_actions"))
user_activity.show(10)


# -----------------------------------------------
# 4D: Category Popularity
# -----------------------------------------------
# Count interactions per product category.

print("[4D] Most Popular Product Categories:")
category_counts = df.groupBy("category") \
                    .agg(F.count("*").alias("interactions")) \
                    .orderBy(F.desc("interactions"))
category_counts.show()


# -----------------------------------------------
# STEP 5: Pattern Detection
# -----------------------------------------------
# We identify the common purchase funnel sequence:
# view → cart → purchase
# Users who follow this sequence are "normal" buyers.

print("\n" + "=" * 60)
print("  SECTION B: PATTERN DETECTION")
print("=" * 60)

# -----------------------------------------------
# 5A: Funnel Analysis — View → Cart → Purchase
# -----------------------------------------------
# For each user, count how many times they did each action.
# Then determine what stage of the funnel each user reached.

print("\n[5A] Purchase Funnel Analysis:")

# Pivot table: rows = users, columns = actions, values = count
funnel_df = df.groupBy("user_id", "action") \
              .agg(F.count("*").alias("count")) \
              .groupBy("user_id") \
              .pivot("action", ["view", "cart", "purchase"]) \
              .agg(F.sum("count")) \
              .fillna(0)

# Rename columns for clarity
funnel_df = funnel_df.withColumnRenamed("view", "view_count") \
                     .withColumnRenamed("cart", "cart_count") \
                     .withColumnRenamed("purchase", "purchase_count")

# Classify user funnel stage
funnel_df = funnel_df.withColumn(
    "funnel_stage",
    F.when(F.col("purchase_count") > 0, "Complete (View→Cart→Purchase)")
     .when(F.col("cart_count") > 0, "Partial (View→Cart)")
     .otherwise("Browse Only (View)")
)

print("  User Funnel Breakdown:")
funnel_df.show(20, truncate=False)

# Funnel stage summary
print("  Funnel Stage Summary:")
funnel_summary = funnel_df.groupBy("funnel_stage") \
                           .agg(F.count("*").alias("user_count")) \
                           .orderBy(F.desc("user_count"))
funnel_summary.show()


# -----------------------------------------------
# 5B: Session-Level Sequence Analysis
# -----------------------------------------------
# For each session, collect the sequence of actions
# ordered by time to detect real behavioral patterns.

print("[5B] Action Sequences per Session:")

window_spec = Window.partitionBy("session_id").orderBy("time")
df_ranked = df.withColumn("action_rank", F.rank().over(window_spec))

session_sequences = df.groupBy("session_id", "user_id") \
    .agg(
        F.collect_list(
            F.struct(F.col("time"), F.col("action"))
        ).alias("actions_raw")
    )

# Simpler: collect actions ordered within session as string
df_with_order = df.withColumn("rn", F.row_number().over(window_spec))
session_actions = df.groupBy("session_id", "user_id") \
    .agg(F.concat_ws(" → ", F.collect_list("action")).alias("action_sequence"))

print("  Session Action Sequences:")
session_actions.show(10, truncate=False)

# Count how many sessions have complete view→cart→purchase pattern
complete_funnel_sessions = session_actions.filter(
    F.col("action_sequence").contains("view") &
    F.col("action_sequence").contains("cart") &
    F.col("action_sequence").contains("purchase")
)
print(f"  Sessions with Complete Funnel (view+cart+purchase): {complete_funnel_sessions.count()}")


# -----------------------------------------------
# STEP 6: Anomaly Detection
# -----------------------------------------------
# Anomalies are behaviors that deviate significantly
# from normal patterns. We detect two types:
# (1) Users with too many views and zero purchases
# (2) Users who are highly active beyond normal range

print("\n" + "=" * 60)
print("  SECTION C: ANOMALY DETECTION")
print("=" * 60)

# -----------------------------------------------
# 6A: High View-to-Purchase Ratio (Suspicious Browsing)
# -----------------------------------------------
# Compute how many views each user did without ever purchasing.
# If ratio > threshold, flag as anomalous.

print("\n[6A] High View-to-No-Purchase Ratio Detection:")

user_behavior = df.groupBy("user_id") \
    .agg(
        F.sum(F.when(F.col("action") == "view", 1).otherwise(0)).alias("views"),
        F.sum(F.when(F.col("action") == "cart", 1).otherwise(0)).alias("carts"),
        F.sum(F.when(F.col("action") == "purchase", 1).otherwise(0)).alias("purchases")
    )

# Calculate view-to-purchase ratio
user_behavior = user_behavior.withColumn(
    "view_purchase_ratio",
    F.when(
        F.col("purchases") == 0,
        F.col("views")         # If 0 purchases, ratio = total views (high = anomalous)
    ).otherwise(
        F.round(F.col("views") / F.col("purchases"), 2)
    )
)

# Flag users with 5+ views and 0 purchases as anomalous
anomalous_users = user_behavior.withColumn(
    "anomaly_flag",
    F.when(
        (F.col("views") >= 5) & (F.col("purchases") == 0),
        "HIGH RISK - Never Purchased"
    ).when(
        (F.col("view_purchase_ratio") > 8) & (F.col("purchases") > 0),
        "MODERATE RISK - Low Conversion"
    ).otherwise("NORMAL")
)

print("  User Behavior Analysis:")
anomalous_users.orderBy(F.desc("views")).show(20, truncate=False)

# Show only flagged anomalies
print("  >> Flagged Anomalous Users:")
flagged = anomalous_users.filter(F.col("anomaly_flag") != "NORMAL")
flagged.show(truncate=False)
print(f"  Total Anomalous Users Detected: {flagged.count()} out of {user_behavior.count()}")


# -----------------------------------------------
# 6B: Hyperactive Users (Unusual Session Frequency)
# -----------------------------------------------
# Users with unusually high total actions compared
# to the mean may be bots or programmatic crawlers.

print("\n[6B] Hyperactive / Bot-Like User Detection:")

# Calculate mean and standard deviation of total actions per user
stats = user_behavior.agg(
    F.mean("views").alias("mean_views"),
    F.stddev("views").alias("std_views")
).collect()[0]

mean_v = stats["mean_views"]
std_v  = stats["std_views"]
threshold = mean_v + (2 * std_v)  # 2 standard deviations above mean

print(f"  Mean Views per User  : {round(mean_v, 2)}")
print(f"  Std Dev of Views     : {round(std_v, 2)}")
print(f"  Anomaly Threshold    : {round(threshold, 2)} views")

# Flag users whose views exceed the threshold
hyperactive = user_behavior.withColumn(
    "behavior_type",
    F.when(F.col("views") > threshold, "BOT / HYPERACTIVE SUSPECTED")
     .otherwise("Normal User")
)

print("\n  Hyperactive User Flags:")
hyperactive.filter(F.col("behavior_type") != "Normal User").show(truncate=False)


# -----------------------------------------------
# STEP 7: Final Summary Report
# -----------------------------------------------

print("\n" + "=" * 60)
print("  FINAL ANALYSIS SUMMARY")
print("=" * 60)

total_users    = df.select("user_id").distinct().count()
total_sessions = df.select("session_id").distinct().count()
total_events   = df.count()
normal_users   = anomalous_users.filter(F.col("anomaly_flag") == "NORMAL").count()
anomaly_count  = flagged.count()

print(f"\n  Dataset Statistics:")
print(f"    Total Events Analyzed  : {total_events}")
print(f"    Unique Users           : {total_users}")
print(f"    Unique Sessions        : {total_sessions}")
print(f"\n  Behavior Classification:")
print(f"    Normal Users           : {normal_users}")
print(f"    Anomalous Users        : {anomaly_count}")
print(f"    Anomaly Rate           : {round((anomaly_count/total_users)*100, 1)}%")
print(f"\n  Peak Activity:")
print(f"    Peak Hour              : {peak_hour['hour']}:00")
print(f"    Peak Day               : {day_activity.first()['day_of_week']}")

print("\n  Project Execution Complete!")
print("=" * 60)

spark.stop()