from transformers import AutoTokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)

@udf(ArrayType(IntegerType()))
def tokenize(text):
    """
    Tokenize the text using the Longformer tokenizer.
    """
    try:
        if text:
            return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=4096)
        return []
    except Exception as e:
        logger.error(f"Error tokenizing text: {text}, Error: {e}")
        return []

def main():
    logger.info("Text encoding script started.")

    # Initialize SparkSession
    spark = (
        SparkSession.builder.appName("Text Encoder")
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

    # Load the CSV files
    input_path = "./csv_format/joined_data"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    logger.info("Loading CSV data...")
    data_frame = spark.read.csv(input_path, header=True, inferSchema=True)

    # Check if 'text' column exists
    if "text" not in data_frame.columns:
        raise ValueError("The CSV file does not contain a 'text' column.")

    # Tokenize the text data
    logger.info("Encoding text data...")
    data_frame = data_frame.withColumn("tokenized_text", tokenize(data_frame.text))

    # Cache and debug
    data_frame = data_frame.cache()
    if data_frame.count() > 0:
        logger.info(f"Sample text: {data_frame.select('text').first()}")
        logger.info(f"Sample tokenized text: {data_frame.select('tokenized_text').first()}")
    else:
        logger.warning("No data available in the DataFrame.")

    # Repartition the data to a single file
    logger.info("Repartitioning data to a single file...")
    data_frame = data_frame.coalesce(1)

    # Save the tokenized data to Parquet
    output_path = "./encoded_format/encoded_data.parquet"
    logger.info("Saving encoded repartitioned data...")
    data_frame.write.parquet(output_path, mode="overwrite")

    logger.info("Text encoding and saving complete.")

if __name__ == "__main__":
    main()
