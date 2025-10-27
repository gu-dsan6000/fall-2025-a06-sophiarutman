from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, count, rand
import sys
import os
import glob

spark = SparkSession.builder.appName("Problem1_LogLevelDistribution").getOrCreate()
input_files = glob.glob("data/raw/**/*.log", recursive=True)
output_dir = "data/output/"
os.makedirs(output_dir, exist_ok=True)

df = spark.read.text(input_files).toDF("log_entry")

# Extract log level using regex
level_regex = r"\b(INFO|WARN|ERROR|DEBUG)\b"
df = df.withColumn("log_level", regexp_extract(col("log_entry"), level_regex, 1))

# Count valid log levels
level_counts = (
    df.filter(col("log_level") != "")
        .groupBy("log_level")
        .agg(count("*").alias("count"))
        .orderBy("count", ascending=False)
)

# Save counts
level_counts.coalesce(1).write.csv(os.path.join(output_dir, "problem1_counts.csv"),
                                    header=True, mode="overwrite")

# Sample 10 random log entries
sample_df = df.filter(col("log_level") != "").orderBy(rand()).limit(10)
sample_df.coalesce(1).write.csv(os.path.join(output_dir, "problem1_sample.csv"),
                                header=True, mode="overwrite")

# Compute summary stats
total_lines = df.count()
total_with_levels = df.filter(col("log_level") != "").count()
unique_levels = level_counts.count()

# Collect counts for percentages
counts = {r["log_level"]: r["count"] for r in level_counts.collect()}
summary_lines = [
    f"Total log lines processed: {total_lines}",
    f"Total lines with log levels: {total_with_levels}",
    f"Unique log levels found: {unique_levels}",
    "",
    "Log level distribution:"
]
for level, count_val in counts.items():
    pct = (count_val / total_with_levels) * 100
    summary_lines.append(f"  {level:<5}: {count_val:>10,} ({pct:6.2f}%)")

with open(os.path.join(output_dir, "problem1_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

spark.stop()
