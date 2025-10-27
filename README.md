# Assignment: Spark Log Analysis on AWS Cluster

This assignment provides hands-on experience with Apache Spark by analyzing real-world Spark cluster logs using PySpark. You will set up a Spark cluster on AWS EC2 and perform distributed data analysis on approximately 2.8 GB of production log data.

## Overview

You will analyze logs from 194 Spark applications running on YARN clusters between 2015-2017, containing 3,852 container log files. The dataset includes ApplicationMaster logs and Executor logs that reveal insights about resource allocation, task execution, performance patterns, and cluster behavior.

For complete dataset documentation, see [DATASET_OVERVIEW.md](DATASET_OVERVIEW.md).

## Learning Objectives

By completing this assignment, you will:
- Set up and manage a distributed Apache Spark cluster on AWS EC2
- Use PySpark to process large-scale log data
- Extract structured information from unstructured log files
- Perform distributed data analysis and aggregations
- Optimize Spark jobs for performance
- Monitor jobs using Spark Web UI
- Apply log parsing and regex techniques at scale

## Prerequisites

- AWS Account with appropriate IAM permissions
- Access to an EC2 instance (Linux) to run the setup commands
- AWS CLI configured with credentials
- Your laptop's public IP address (get from https://ipchicken.com/)
- Basic familiarity with Python and PySpark
- Understanding of distributed computing concepts

## Dataset

**Original Archive:** The dataset is provided as a compressed archive in this repository.

**Structure:**
- 194 application directories
- Each application contains multiple container log files
- Container 000001: ApplicationMaster logs
- Container 000002+: Executor logs
- Total size: ~2.8 GB (uncompressed)

See [DATASET_OVERVIEW.md](DATASET_OVERVIEW.md) for complete documentation.

## Setup Instructions

### 1. Download and Prepare Dataset

The dataset is provided as a compressed archive. You'll need to download, uncompress, and upload it to your own S3 bucket.

```bash
# 1. Create necessary directories
mkdir -p data/raw data/output

# 2. Download the dataset archive from the course S3 bucket
# Note: The bucket uses Requester Pays, so you must include --request-payer requester
aws s3 cp s3://dsan6000-datasets/spark-logs/Spark.tar.gz data/raw/ \
  --request-payer requester

# This will download ~175 MB compressed file

# 3. Extract the archive
cd data/raw/
tar -xzf Spark.tar.gz
cd ../..

# 4. Verify extraction
ls data/raw/application_* | head -10
# Should see directories like: application_1485248649253_0052

# 5. Create your personal S3 bucket (replace YOUR-NET-ID with your actual net ID)
export YOUR_NET_ID="your-net-id"  # e.g., "abc123"
aws s3 mb s3://${YOUR_NET_ID}-assignment-spark-cluster-logs

# 6. Upload the data to S3
aws s3 cp data/raw/ s3://${YOUR_NET_ID}-assignment-spark-cluster-logs/data/ \
  --recursive \
  --exclude "*.gz" \
  --exclude "*.tar.gz"

# 7. Verify upload
aws s3 ls s3://${YOUR_NET_ID}-assignment-spark-cluster-logs/data/ | head -10
# You should see application directories listed
# Note: "Broken pipe" error at the end is normal (caused by head command)

# 8. Save your bucket name for later use
echo "export SPARK_LOGS_BUCKET=s3://${YOUR_NET_ID}-assignment-spark-cluster-logs" >> ~/.bashrc
source ~/.bashrc
```

**Important Notes:**
- The source bucket (`dsan6000-datasets`) uses **Requester Pays**, so always include `--request-payer requester` when downloading
- Once you upload to YOUR personal bucket, you don't need `--request-payer` flag
- The `data/raw/` directory is in `.gitignore` and will not be committed to git
- The compressed archive files (`.gz`, `.tar.gz`) are also ignored
- Only your output files in `data/output/` will be committed

### 2. Install Dependencies

```bash
# Install Java (required for PySpark)
sudo apt update
sudo apt install -y openjdk-17-jdk-headless

# Verify installation
java -version

# Set JAVA_HOME (add to ~/.bashrc for persistence)
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin
```

### 3. Install Python Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 4. Set Up Spark Cluster

Follow [AUTOMATION_README.md](AUTOMATION_README.md) to create your cluster:

```bash
# Run the automated setup script
./setup-spark-cluster.sh 98.192.202.109

# This creates a 4-node cluster (1 master + 3 workers)
# The script will auto-detect your IAM instance profile for S3 access
```

The script will:
- Auto-detect the IAM instance profile from your current EC2 instance
- Create security groups and SSH keys
- Launch 4 EC2 instances (1 master + 3 workers)
- Install and configure Spark on all nodes
- Output cluster configuration to `cluster-config.txt` and `cluster-ips.txt`

**Note:** The script automatically detects the IAM instance profile attached to your current EC2 instance. If no profile is found, it will attempt to use the first available profile, or launch without an IAM role (S3 access may not work).

### 5. Prepare Sample Data for Local Development (Optional but Recommended)

For faster development iteration, create a small sample from the raw data:

```bash
# Copy one application directory as a sample for local testing
mkdir -p data/sample
cp -r data/raw/application_1485248649253_0052 data/sample/

# Now you can test locally before running on the cluster with full dataset
```

## Assignment Problems

Complete **TWO** problems total:

**Development Recommendation:**
While not required for submission, we strongly recommend developing and testing your code locally first (without the cluster) using the sample data in `data/sample/`. This helps you:
- Debug your code faster without cluster overhead
- Iterate quickly on your logic
- Verify your parsing patterns work correctly

Once your code works locally, adapt it for the cluster and run on the full dataset.

---

### Problem 1: Log Level Distribution - 50 Points

Analyze the distribution of log levels (INFO, WARN, ERROR, DEBUG) across all log files. This problem requires basic PySpark operations and simple aggregations.

**Deliverable:**
- `problem1.py`: Python script that runs on the cluster

**Outputs (3 files):**
1. `data/output/problem1_counts.csv`: Log level counts
2. `data/output/problem1_sample.csv`: 10 random sample log entries with their levels
3. `data/output/problem1_summary.txt`: Summary statistics

**Expected output 1 (counts):**
```
log_level,count
INFO,125430
WARN,342
ERROR,89
DEBUG,12
```

**Expected output 2 (sample):**
```
log_entry,log_level
"17/03/29 10:04:41 INFO ApplicationMaster: Registered signal handlers",INFO
"17/03/29 10:04:42 WARN YarnAllocator: Container request...",WARN
...
```

**Expected output 3 (summary):**
```
Total log lines processed: 3,234,567
Total lines with log levels: 3,100,234
Unique log levels found: 4

Log level distribution:
  INFO  :    125,430 (40.45%)
  WARN  :        342 ( 0.01%)
  ...
```

### Problem 2: Cluster Usage Analysis - 50 Points

**⚠️ IMPORTANT: This problem takes approximately 10-20 minutes to run on the cluster.**

Analyze cluster usage patterns to understand which clusters are most heavily used over time. Extract cluster IDs, application IDs, and application start/end times to create a time-series dataset suitable for visualization with Seaborn.

**Key Questions to Answer:**
- How many unique clusters are in the dataset?
- How many applications ran on each cluster?
- Which clusters are most heavily used?
- What is the timeline of application execution across clusters?

**Deliverable:**
- `problem2.py`: Python script that runs on the cluster

**Usage:**
```bash
# Full Spark processing (10-20 minutes)
uv run python problem2.py spark://$MASTER_PRIVATE_IP:7077 --net-id YOUR-NET-ID

# Skip Spark and regenerate visualizations from existing CSVs (fast)
uv run python problem2.py --skip-spark
```

**Outputs (5 files):**
1. `data/output/problem2_timeline.csv`: Time-series data for each application
2. `data/output/problem2_cluster_summary.csv`: Aggregated cluster statistics
3. `data/output/problem2_stats.txt`: Overall summary statistics
4. `data/output/problem2_bar_chart.png`: Bar chart visualization
5. `data/output/problem2_density_plot.png`: Faceted density plot visualization

**Expected output 1 (timeline):**
```
cluster_id,application_id,app_number,start_time,end_time
1485248649253,application_1485248649253_0001,0001,2017-03-29 10:04:41,2017-03-29 10:15:23
1485248649253,application_1485248649253_0002,0002,2017-03-29 10:16:12,2017-03-29 10:28:45
1448006111297,application_1448006111297_0137,0137,2015-11-20 14:23:11,2015-11-20 14:35:22
...
```

**Expected output 2 (cluster summary):**
```
cluster_id,num_applications,cluster_first_app,cluster_last_app
1485248649253,181,2017-03-29 10:04:41,2017-03-29 18:42:15
1472621869829,8,2016-08-30 12:15:30,2016-08-30 16:22:10
...
```

**Expected output 3 (stats):**
```
Total unique clusters: 6
Total applications: 194
Average applications per cluster: 32.33

Most heavily used clusters:
  Cluster 1485248649253: 181 applications
  Cluster 1472621869829: 8 applications
  ...
```

**Visualization Output:**
The script automatically generates two separate visualizations:

1. **Bar Chart** (`problem2_bar_chart.png`):
   - Number of applications per cluster
   - Value labels displayed on top of each bar
   - Color-coded by cluster ID

2. **Density Plot** (`problem2_density_plot.png`):
   - Shows job duration distribution for the **largest cluster** (cluster with most applications)
   - Histogram with KDE overlay
   - **Log scale** on x-axis to handle skewed duration data
   - Sample count (n=X) displayed in title

---

## Development Workflow

### Overview

Follow all instructions for code development and cluster creation from the **lab-spark-cluster** repository. Once you have:
1. Created your Spark cluster following the lab instructions
2. Developed and tested your code locally on your EC2 instance (optionally in a Jupyter notebook)
3. Created a `.py` file ready to run on the cluster

Then proceed to run your analysis on the cluster using the steps below.

**Repeat this workflow for both problems (problem1.py and problem2.py).**

---

### Running Your Analysis on the Cluster

```bash
# From your local machine, load cluster configuration
source cluster-config.txt

# Copy your script to master node
scp -i $KEY_FILE problem1.py ubuntu@$MASTER_PUBLIC_IP:~/

# SSH to master node
ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP

# On the master node:
cd ~/spark-cluster
source cluster-ips.txt

# Run your script on the cluster (IMPORTANT: Use your actual net ID!)
# Replace YOUR-NET-ID with your actual net ID (e.g., abc123)
uv run python ~/problem1.py spark://$MASTER_PRIVATE_IP:7077 --net-id YOUR-NET-ID

# Example:
# uv run python ~/problem1.py spark://$MASTER_PRIVATE_IP:7077 --net-id abc123

# Exit back to your local machine
exit

# Download results to your repo (adjust filenames based on your problem)
# For problem1, download all 3 output files:
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/data/output/problem1_counts.csv data/output/
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/data/output/problem1_sample.csv data/output/
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/data/output/problem1_summary.txt data/output/

# For problem2, download all 5 output files:
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/problem2_timeline.csv data/output/
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/problem2_cluster_summary.csv data/output/
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/problem2_stats.txt data/output/
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/problem2_bar_chart.png data/output/
scp -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/spark-cluster/problem2_density_plot.png data/output/
```

**Important Notes:**
- Replace `problem1.py` with `problem2.py` as needed
- Replace `YOUR-NET-ID` with your actual net ID (e.g., abc123)
- Download all output files generated by each problem
- Repeat this entire process for both problems

---

## Expected Execution Times

- **Local runs** (sample data): 1-5 minutes
- **Cluster runs** (full dataset): 5-20 minutes depending on complexity

Monitor progress in the Spark Web UI:
- Master UI: `http://$MASTER_PUBLIC_IP:8080`
- Application UI: `http://$MASTER_PUBLIC_IP:4040`

## Deliverables

Submit the following to your GitHub repository:

### Required Files

#### 1. Python Scripts (in repository root):
   - `problem1.py` - Your solution for Problem 1: Log Level Distribution
   - `problem2.py` - Your solution for Problem 2: Cluster Usage Analysis

#### 2. Problem 1 Output Files (in `data/output/` directory):
   - `problem1_counts.csv` - Log level counts
   - `problem1_sample.csv` - Sample log entries with levels
   - `problem1_summary.txt` - Summary statistics

#### 3. Problem 2 Output Files (in `data/output/` directory):
   - `problem2_timeline.csv` - Time-series data for each application
   - `problem2_cluster_summary.csv` - Aggregated cluster statistics
   - `problem2_stats.txt` - Overall summary statistics
   - `problem2_bar_chart.png` - Bar chart showing applications per cluster
   - `problem2_density_plot.png` - Faceted density plots showing duration distribution per cluster

#### 4. Analysis Report (`ANALYSIS.md` in repository root):
   - Brief description of your approach for each problem
   - Key findings and insights from the data
   - Performance observations (execution time, optimizations)
   - Screenshots of Spark Web UI showing job execution
   - Explanation of the visualizations generated in Problem 2

### Do NOT Submit:
- Local test files or outputs
- Configuration files (`cluster-config.txt`, `cluster-ips.txt`, `*.pem`)
- Raw data files (`data/raw/`)
- Compressed archives (`*.tar.gz`, `*.gz`)

**Note:** Configuration files are in `.gitignore` for security reasons.

---

## Summary of Deliverables

**Total Files to Submit:** 10 files

| File | Location | Description |
|------|----------|-------------|
| `problem1.py` | Repository root | Problem 1 script |
| `problem2.py` | Repository root | Problem 2 script |
| `problem1_counts.csv` | `data/output/` | Problem 1 log level counts |
| `problem1_sample.csv` | `data/output/` | Problem 1 sample entries |
| `problem1_summary.txt` | `data/output/` | Problem 1 statistics |
| `problem2_timeline.csv` | `data/output/` | Problem 2 timeline data |
| `problem2_cluster_summary.csv` | `data/output/` | Problem 2 cluster summary |
| `problem2_stats.txt` | `data/output/` | Problem 2 statistics |
| `problem2_bar_chart.png` | `data/output/` | Problem 2 bar chart |
| `problem2_density_plot.png` | `data/output/` | Problem 2 density plots |
| `ANALYSIS.md` | Repository root | Your analysis report |

---

## Grading Rubric

**Total: 100 Points**

### Problem 1: Log Level Distribution (50 points)
- Correct implementation (30 points)
  - Accurate log level extraction and counting (15 points)
  - Proper sample generation (10 points)
  - Correct summary statistics (5 points)
- Code quality and documentation (10 points)
- Output file completeness (10 points)
  - All 3 required output files present and correctly formatted

### Problem 2: Cluster Usage Analysis (50 points)
- Correct implementation (30 points)
  - Accurate cluster and application ID extraction (10 points)
  - Proper timeline data generation (10 points)
  - Correct aggregations and statistics (10 points)
- Visualization quality (10 points)
  - Both visualizations generated correctly with Seaborn
  - Bar chart shows applications per cluster with clear labels
  - Density plot shows job duration distribution for largest cluster
  - Log scale properly applied to handle skewed data
  - Charts are clear, labeled, and informative
- Code quality and documentation (5 points)
- Output file completeness (5 points)
  - All 5 required output files present and correctly formatted

### Analysis Report (50 points bonus)
- Comprehensive analysis of findings (20 points)
  - Clear description of approach for each problem
  - Key insights and patterns discovered in the data
  - Discussion of cluster usage patterns and trends
- Performance analysis (15 points)
  - Execution time observations
  - Optimization strategies employed
  - Comparison of local vs cluster performance
- Documentation quality (10 points)
  - Screenshots of Spark Web UI showing job execution
  - Well-formatted markdown with clear sections
  - Professional presentation
- Additional insights (5 points)
  - Novel visualizations beyond requirements
  - Deeper analysis of the data
  - Creative problem-solving approaches

**Maximum total with bonus: 150 points**

## Hints and Tips

### Parsing Log Files

Use PySpark's `regexp_extract` for parsing:

```python
from pyspark.sql.functions import regexp_extract, col

logs_parsed = logs_df.select(
    regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
    regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('log_level'),
    regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)\s+([^:]+):', 2).alias('component'),
    col('value').alias('message')
)
```

### Extracting Application and Container IDs

```python
from pyspark.sql.functions import input_file_name

# Extract from file path
df = logs_df.withColumn('file_path', input_file_name())
df = df.withColumn('application_id',
    regexp_extract('file_path', r'application_(\d+_\d+)', 0))
df = df.withColumn('container_id',
    regexp_extract('file_path', r'(container_\d+_\d+_\d+_\d+)', 1))
```

### Performance Optimization

- Use `broadcast()` for small lookup tables
- Cache intermediate DataFrames: `df.cache()`
- Avoid collecting large datasets to driver
- Use DataFrame operations instead of RDDs
- Partition data appropriately: `repartition()`

### Handling Timestamps

```python
from pyspark.sql.functions import to_timestamp

df = df.withColumn('timestamp',
    to_timestamp('timestamp', 'yy/MM/dd HH:mm:ss'))
```

## Cleanup

**IMPORTANT:** Always clean up your cluster to avoid unnecessary AWS charges!

```bash
# Automated cleanup
./cleanup-spark-cluster.sh

# Or manual cleanup:
# 1. Terminate EC2 instances
# 2. Delete key pair
# 3. Delete security group
```

## Repository Structure

```
assignment-spark-cluster/
├── README.md                      # This file - assignment instructions
├── DATASET_OVERVIEW.md            # Complete dataset documentation
├── AUTOMATION_README.md           # Cluster setup guide
├── ANALYSIS.md                    # Your analysis report (to be created)
├── setup-spark-cluster.sh         # Automated cluster creation script
├── cleanup-spark-cluster.sh       # Automated cleanup script
├── pyproject.toml                 # Python dependencies
├── .gitignore                     # Git ignore rules
├── data/
│   ├── raw/                       # Raw log files (NOT committed - in .gitignore)
│   │   ├── Spark.tar.gz           # Downloaded archive (NOT committed)
│   │   └── application_*/         # Extracted log directories (NOT committed)
│   ├── sample/                    # Small sample for local testing
│   │   └── application_*/         # Sample application logs
│   └── output/                    # Your CSV results (COMMIT THESE!)
│       ├── problem1_X_local.csv
│       └── problem1_X_cluster.csv
├── cluster-files/                 # Cluster configuration templates
├── src/                           # Source code utilities
├── problem1_X_local.py            # Your solution scripts (to be created)
└── problem1_X_cluster.py          # Your cluster scripts (to be created)
```

**Important Notes:**
- `data/raw/` is in `.gitignore` - never commit large datasets
- `data/output/` is NOT in `.gitignore` - commit your CSV results here
- Configuration files with secrets (`.pem`, `cluster-config.txt`) are in `.gitignore`

## Cost Estimate

**Instance type:** t3.large (4 vCPU, 8 GB RAM)
**Number of instances:** 4 (1 master + 3 workers)
**Estimated cost:** $0.33/hour

**Expected total cost:** $3-5 for completing the assignment (assuming 10-15 hours including cluster time)

**Remember to terminate your cluster when not actively using it!**

## Troubleshooting

### Common Issues

1. **Java not found:**
   ```bash
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
   ```

2. **Spark session fails:**
   - Verify cluster is running: check Master UI
   - Check master URL format: `spark://PRIVATE_IP:7077`
   - Ensure security group allows worker communication

3. **S3 access denied:**
   - Verify IAM role attached to EC2 instances
   - Check S3 bucket permissions

4. **Parsing errors:**
   - Handle malformed log lines gracefully
   - Use `try/except` in UDFs for robust parsing

### Getting Help

1. Review [DATASET_OVERVIEW.md](DATASET_OVERVIEW.md) for log format details
2. Check Spark logs: `$SPARK_HOME/logs/`
3. Monitor Spark Web UI for job progress and errors
4. Refer to [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)

## Security Notes

- Never commit `.pem` files or AWS credentials
- Limit security group access to your IP only
- Use IAM roles for S3 access (not access keys)
- Always clean up resources when done

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Spark on YARN Architecture](https://spark.apache.org/docs/latest/running-on-yarn.html)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)

## Academic Integrity

This is an individual assignment. You may discuss high-level concepts with classmates, but all code must be your own work. Copying code from others or online sources without attribution is prohibited.

## Submission

Submit your completed assignment by pushing to your GitHub repository:

```bash
git add .
git commit -m "Complete Spark log analysis assignment"
git push origin main
```

Ensure your repository includes all deliverables listed above.

## License

MIT License - See LICENSE file for details.
