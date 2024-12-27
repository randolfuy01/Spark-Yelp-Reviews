"""
    Data preprocessing for the Yelp data: initial conversion from JSON to Parquet format.
    Considerations:
        - Convert large JSON datasets into Parquet for efficient storage and processing.
        - Leverage Apache Spark for handling large datasets with SQL-style operations.
"""

from pyspark.sql import SparkSession
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize the SparkSession
    spark = (
        SparkSession.builder.appName("Yelp Data Preprocessing")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "6g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "4g")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.sql.files.maxPartitionBytes", "128MB")
        .getOrCreate()
    )

    # File paths to process
    file_paths = [
        "./raw_format/yelp_academic_dataset_tip.json",
        "./raw_format/yelp_academic_dataset_review.json",
        "./raw_format/yelp_academic_dataset_business.json",
        "./raw_format/yelp_academic_dataset_checkin.json",
        "./raw_format/yelp_academic_dataset_user.json",
    ]

    logger.info("Starting Yelp data preprocessing script.")

    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File {path} does not exist. Skipping...")
            continue

        try:
            logger.info(f"Processing file: {path}")

            # Read JSON file into a DataFrame
            dataframe = spark.read.json(path)
            record_count = dataframe.count()
            logger.info(f"Loaded {record_count} records from {path}")

            # Generate output path
            file_name = os.path.basename(path).replace(".json", "")
            output_path = f"./parquet_format/{file_name}.parquet"

            # Write the DataFrame as a Parquet file
            dataframe.write.mode("overwrite").parquet(output_path)
            logger.info(f"Finished writing Parquet file to {output_path}")

        except Exception as e:
            logger.error(f"Failed to process file {path}: {e}", exc_info=True)

    logger.info("Data preprocessing complete.")


if __name__ == "__main__":
    main()
