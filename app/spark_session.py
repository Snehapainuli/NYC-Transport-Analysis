# ✅ spark_session.py
from pyspark.sql import SparkSession

def get_spark_session():
    spark = SparkSession.builder \
        .appName("NYC Transport ML App") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    return spark

# Create global instance
spark = get_spark_session()
spark.sparkContext.setLogLevel("ERROR")