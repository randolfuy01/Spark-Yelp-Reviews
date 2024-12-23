""" 
    Data preprocessing for the yelp data initial conversion from json formatted data into parquet
    Things taken into consideration:

        - Dataset is as json data, we want to shrink the data by turning it into 
            csv format and saving that locally instead

        - Large data amounts makes it so that the preferred method of extracting data
            is by leveraging Apache spark (pyspark) for SQL queries to clean data

"""
from pyspark.sql import SparkSession
import os
import shutil

def main():

    # Initialize the SparkSession with necessary configurations
    spark: SparkSession.builder = SparkSession.builder \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.memory", "6g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "4g") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
        .getOrCreate()

    file_paths = ["./raw_format/yelp_academic_dataset_tip.json", "./raw_format/yelp_academic_dataset_review.json",
                   "./raw_format/yelp_academic_dataset_business.json", "./raw_format/yelp_academic_dataset_checkin.json",
                   "./raw_format/yelp_academic_dataset_user.json"
                ]
    
    for path in file_paths:
        dataframe = spark.read.json(path)
        # Get the file name (without the directory) to use in the output path
        file_name = path.split("/")[-1].replace(".json", "")

        # Define the output Parquet path using the file name
        output_path = f"./parquet_format/{file_name}"

        # Write the DataFrame as a Parquet file
        dataframe.coalesce(1).write.mode("overwrite").parquet(output_path)

        # Find and move the single Parquet file to a desired location
        for root, files in os.walk(output_path):
            for file in files:
                if file.endswith(".parquet"):
                    shutil.move(os.path.join(root, file), f"{output_path}.parquet")

        shutil.rmtree(output_path)

if __name__ == "__main__":
    main()