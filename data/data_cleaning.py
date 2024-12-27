"""
    Parquet loaded data aggregation and assimilation
    - Focus on restaurant-related data for Yelp reviews.
    - Clean and prepare the data for further analysis or model training.
"""

from pyspark.sql import SparkSession
import os
import logging
import shutil


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting data cleaning script")

    # Initialize the SparkSession
    spark = (
        SparkSession.builder.appName("Yelp Data Cleaning")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "4g")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    # Load the Parquet files
    try:
        logger.info("Loading Parquet files")
        business_df = spark.read.parquet("./data/parquet_format/business.parquet")
        review_df = spark.read.parquet(
            "./data/parquet_format/yelp_academic_dataset_review.parquet"
        )
    except Exception as e:
        logger.error(f"Failed to load Parquet files: {e}")
        return

    # Filter and clean data
    logger.info("Filtering business data")
    business_df = business_df.select(
        "business_id", "name", "review_count", "attributes", "categories"
    )
    review_df = review_df.select("business_id", "stars", "text")

    # Query for restaurant-related businesses
    logger.info("Filtering for restaurants and food-related businesses")
    business_df.createOrReplaceTempView("business")
    business_df = spark.sql(
        """
        SELECT business_id, name, review_count
        FROM business
        WHERE LOWER(categories) LIKE '%restaurant%' OR LOWER(categories) LIKE '%food%'
        """
    )

    # Join DataFrames
    logger.info("Joining business and review data")
    joined_df = business_df.join(review_df, "business_id", "inner").select(
        "name", "stars", "text"
    )

    # Drop null values
    logger.info("Dropping rows with null values")
    joined_df = joined_df.dropna()

    # Count rows saved
    rows_saved = joined_df.count()
    logger.info(f"Rows saved: {rows_saved}")

    # Save to CSV
    output_path = "./data/csv_format/joined_data"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    logger.info("Saving to CSV")
    joined_df.repartition(1).write.csv(output_path, header=True)

    logger.info("Data cleaning complete")


if __name__ == "__main__":
    main()