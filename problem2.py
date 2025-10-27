#!/usr/bin/env python3
"""
Problem 2: Cluster Usage Analysis
---------------------------------
Analyzes Spark log data to extract cluster usage statistics and visualize trends.

Usage:
  uv run python problem2.py spark://<MASTER_PRIVATE_IP>:7077 --net-id sr1672
  uv run python problem2.py --skip-spark
"""

import os
import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import glob

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, regexp_extract, to_timestamp, when, lit


OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster Usage Analysis")
    parser.add_argument("spark_url", nargs="?", default=None, help="Spark master URL (e.g., spark://10.0.0.5:7077)")
    parser.add_argument("--net-id", type=str, default="unknown", help="Your NetID (for tagging output files)")
    parser.add_argument("--skip-spark", action="store_true", help="Skip Spark processing and reload from existing CSVs")
    return parser.parse_args()


def run_spark_job(spark_url):
    spark = (
        SparkSession.builder
        .appName("Problem2_ClusterUsage")
        .master(spark_url)
        .getOrCreate()
    )

    # 1. Read all .log files
    input_files = glob.glob("data/raw/**/*.log", recursive=True)

    df = (
        spark.read.text(input_files).toDF("log_entry")
        .withColumn("file_path", F.input_file_name())
    )

    # 2. Extract identifiers from path
    df_parsed = (
        df.withColumn(
            "application_id",
            regexp_extract("file_path", r"(application_\d+_\d+)", 1)
        )
        .withColumn(
            "cluster_id",
            regexp_extract("application_id", r"application_(\d+)_\d+", 1)  # first number
        )
        .withColumn(
            "app_number",
            regexp_extract("application_id", r"application_\d+_(\d+)", 1)  # second number
        )
        .withColumn(
            "container_id",
            regexp_extract("file_path", r"(container_\d+_\d+_\d+_\d+)", 1)
        )
    )
    df_parsed.select("cluster_id").distinct().show()
    print(f"Parsed {df_parsed.count()} log lines")

    # 3. Extract timestamps from log text
    df_ts = df_parsed.withColumn(
        "ts_str",
        regexp_extract(col("log_entry"), r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1)
    )

    # Convert to timestamp safely: empty strings become NULL
    df_ts = df_ts.withColumn(
        "ts",
        when(col("ts_str") != "", to_timestamp(col("ts_str"), "yy/MM/dd HH:mm:ss"))
        .otherwise(lit(None))
    )

    # Keep only lines with timestamps
    df_ts = df_ts.filter(col("ts").isNotNull())

    # 4. Aggregate to get start/end times
    df_containers = (
        df_ts.groupBy("cluster_id", "application_id", "app_number", "container_id")
             .agg(F.min("ts").alias("container_start"),
                  F.max("ts").alias("container_end"))
    )

    df_apps = (
        df_containers.groupBy("cluster_id", "application_id", "app_number")
                     .agg(F.min("container_start").alias("start_time_ts"),
                          F.max("container_end").alias("end_time_ts"))
                     .withColumn("start_time", F.date_format("start_time_ts", "yyyy-MM-dd HH:mm:ss"))
                     .withColumn("end_time", F.date_format("end_time_ts", "yyyy-MM-dd HH:mm:ss"))
                     .select("cluster_id", "application_id", "app_number", "start_time", "end_time")
    )


    # 5. Assign sequential app_number per cluster
    window = Window.partitionBy("cluster_id").orderBy("start_time")
    df_timeline = df_apps.withColumn("app_number_seq", F.row_number().over(window))

    # 6. Save timeline CSV
    timeline_pd = df_timeline.toPandas()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timeline_csv = os.path.join(OUTPUT_DIR, "problem2_timeline.csv")
    timeline_pd.to_csv(timeline_csv, index=False)

    # 7. Cluster summary
    cluster_summary = (
        timeline_pd.groupby("cluster_id", as_index=False)
                   .agg(
                       num_applications=("application_id", "count"),
                       cluster_first_app=("start_time", "min"),
                       cluster_last_app=("end_time", "max"),
                   )
    )

    summary_csv = os.path.join(OUTPUT_DIR, "problem2_cluster_summary.csv")
    cluster_summary.to_csv(summary_csv, index=False)

    # 8. Stats text file
    total_clusters = cluster_summary["cluster_id"].nunique()
    total_apps = len(timeline_pd)
    avg_apps_per_cluster = total_apps / total_clusters if total_clusters else 0

    most_heavy = cluster_summary.sort_values("num_applications", ascending=False)

    stats_text = (
        f"Total unique clusters: {total_clusters}\n"
        f"Total applications: {total_apps}\n"
        f"Average applications per cluster: {avg_apps_per_cluster:.2f}\n\n"
        "Most heavily used clusters:\n"
    )

    for _, row in most_heavy.iterrows():
        stats_text += f"  Cluster {row.cluster_id}: {row.num_applications} applications\n"

    stats_path = os.path.join(OUTPUT_DIR, "problem2_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)


    return timeline_pd, cluster_summary


def generate_visualizations(timeline_pd, cluster_summary):
    # Bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x="cluster_id", y="num_applications", data=cluster_summary, palette="viridis")
    plt.title("Applications per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Applications")
    for index, row in cluster_summary.iterrows():
        plt.text(index, row.num_applications + 0.5, str(row.num_applications), ha='center')
    bar_chart_path = os.path.join(OUTPUT_DIR, "problem2_bar_chart.png")
    plt.tight_layout()
    plt.savefig(bar_chart_path)
    plt.close()

    # Density plot for largest cluster
    largest_cluster = cluster_summary.loc[cluster_summary["num_applications"].idxmax(), "cluster_id"]
    cluster_df = timeline_pd[timeline_pd["cluster_id"] == largest_cluster].copy()
    cluster_df["start_time"] = pd.to_datetime(cluster_df["start_time"])
    cluster_df["end_time"] = pd.to_datetime(cluster_df["end_time"])
    cluster_df["duration_sec"] = (cluster_df["end_time"] - cluster_df["start_time"]).dt.total_seconds()


    plt.figure(figsize=(8, 5))
    sns.histplot(cluster_df["duration_sec"], bins=30, kde=True, log_scale=True)
    plt.title(f"Duration Distribution for Cluster {largest_cluster} (n={len(cluster_df)})")
    plt.xlabel("Duration (seconds, log scale)")
    plt.ylabel("Frequency")
    density_path = os.path.join(OUTPUT_DIR, "problem2_density_plot.png")
    plt.tight_layout()
    plt.savefig(density_path)
    plt.close()


def main():
    args = parse_args()

    if args.skip_spark:
        timeline_pd = pd.read_csv(os.path.join(OUTPUT_DIR, "problem2_timeline.csv"))
        cluster_summary = pd.read_csv(os.path.join(OUTPUT_DIR, "problem2_cluster_summary.csv"))
    else:
        if not args.spark_url:
            sys.exit(1)
        timeline_pd, cluster_summary = run_spark_job(args.spark_url)

    generate_visualizations(timeline_pd, cluster_summary)


if __name__ == "__main__":
    main()
