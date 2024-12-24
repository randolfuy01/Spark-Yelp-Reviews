""" 
    Parquet loaded data aggregation and assimilation
        - Restaurant information is most important for determining success rate
        - Remove unnecessary tags
        - Concatenate necessary fields 
"""

from pyspark.sql import SparkSession


def main():

    print("Data cleaning script")
    # Initialize the SparkSession with necessary configurations (adjust as needed)
    spark: SparkSession.builder = (
        SparkSession.builder.config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "4g")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    # Load the necessary parquet files
    print("Loading parquet files")
    business_df = spark.read.parquet("./data/parquet_format/business.parquet")
    review_df = spark.read.parquet(
        "./data/parquet_format/yelp_academic_dataset_review.parquet"
    )

    # Filter out the necessary columns
    print("Filtering out necessary columns")
    business_df = business_df.select(
        "business_id", "name", "review_count", "attributes", "categories"
    )
    review_df = review_df.select("business_id", "stars", "text")

    print("Joining dataframes")
    business_df.createOrReplaceTempView("business")
    business_df = spark.sql(
        """
    SELECT business_id, name, review_count, attributes, categories
    FROM business
    WHERE categories LIKE '%Restaurant%' OR categories LIKE '%Food%'
    """
    )

    business_df = business_df.drop("categories").drop("attributes")

    # Join dataframes
    joined_df = business_df.join(review_df, "business_id", "inner").select(
        "name", "stars", "text"
    )

    # Drop rows where any of the values are null
    joined_df = joined_df.filter(
        joined_df.name.isNotNull()
        & joined_df.stars.isNotNull()
        & joined_df.text.isNotNull()
    )

    rows_saved = joined_df.count()
    print(f"Rows saved: {rows_saved}")

    # Save to CSV in chunks
    print("Saving to CSV")
    joined_df.write.csv("./data/csv_format/joined_data", header=True)

    print("Data cleaning complete")


if __name__ == "__main__":
    main()
