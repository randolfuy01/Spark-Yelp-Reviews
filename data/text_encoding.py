from transformers import AutoTokenizer

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


@udf(ArrayType(IntegerType()))
def tokenize(text):
    """
    Tokenize the text using the BERT tokenizer
    """
    if text:
        return tokenizer.encode(text, add_special_tokens=True)
    return []


def main():
    print("Text encoding script")

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
    print("Loading CSV data...")
    data_frame = spark.read.csv(
        "./csv_format/joined_data", header=True, inferSchema=True
    )

    # Check if 'text' column exists
    if "text" not in data_frame.columns:
        raise ValueError("The CSV file does not contain a 'text' column.")

    # Tokenize the text data
    print("Encoding text data...")
    data_frame = data_frame.withColumn("tokenized_text", tokenize(data_frame.text))

    # Print sample text and tokenzied text
    print(f"Sample text: {data_frame.select('text').first()}")
    print(f"Sample tokenized text: {data_frame.select('tokenized_text').first()}")

    # Repartition the data to a single file
    print("Repartitioning data to a single file...")
    data_frame = data_frame.coalesce(1)

    # Save the tokenized data to Parquet
    print("Saving encoded repartitioned data...")
    data_frame.write.parquet("./encoded_format/encoded_data.parquet", mode="overwrite")

    print("Text encoding and saving complete.")


if __name__ == "__main__":
    main()
