from pyspark.sql.functions import *
from pyspark.sql.window import Window


# Logger

def log(msg):
    print(f"[INFO] {msg}")

# Base path
base = "/Volumes/workspace/default/hospitality/"


# Bronze Layer

try:
    log("Starting Bronze Layer")

    customers = spark.read.csv(base+"customers.csv", header=True, inferSchema=True)
    hotels = spark.read.csv(base+"hotels.csv", header=True, inferSchema=True)
    bookings = spark.read.csv(base+"bookings.csv", header=True, inferSchema=True)
    payments = spark.read.csv(base+"payments.csv", header=True, inferSchema=True)

    customers.write.format("delta").mode("overwrite").save(base+"bronze/customers")
    hotels.write.format("delta").mode("overwrite").save(base+"bronze/hotels")
    bookings.write.format("delta").mode("overwrite").save(base+"bronze/bookings")
    payments.write.format("delta").mode("overwrite").save(base+"bronze/payments")

    log("Bronze Layer Completed")

except Exception as e:
    log(f"Bronze Layer Failed: {e}")



# Silver Layer

try:
    log("Starting Silver Layer")

    cust = customers.dropDuplicates().fillna({"email":"unknown@gmail.com"})

    df = bookings.join(cust, "customer_id")\
        .join(hotels, "hotel_id")\
        .join(payments, "booking_id")

    df = df.withColumn("booking_date", col("booking_date").cast("date"))\
        .withColumn("day", date_format("booking_date", "E"))\
        .withColumn("dynamic_price",
            when(col("day").isin("Sat","Sun"), col("base_price") * 1.5)
            .otherwise(col("base_price"))
        )

    
    # Data Quality Checks
    
    assert df.filter(col("customer_id").isNull()).count() == 0
    assert df.filter(col("booking_id").isNull()).count() == 0

    df.write.format("delta")\
        .partitionBy("booking_date")\
        .mode("overwrite")\
        .save(base+"silver/sales")

    log("Silver Layer Completed")

except Exception as e:
    log(f"Silver Layer Failed: {e}")



# Gold Layer

try:
    log("Starting Gold Layer")

    df = spark.read.format("delta").load(base+"silver/sales")

    # 1. Daily Revenue
    df.groupBy("booking_date")\
        .agg(sum("dynamic_price").alias("total_revenue"))\
        .write.mode("overwrite").save(base+"gold/daily_revenue")

    # 2. Customer Lifetime Value (CLV)
    df.groupBy("customer_id")\
        .agg(sum("dynamic_price").alias("clv"))\
        .write.mode("overwrite").save(base+"gold/clv")

    # 3. Revenue per Hotel
    df.groupBy("hotel_name")\
        .agg(sum("dynamic_price").alias("total_revenue"))\
        .write.mode("overwrite").save(base+"gold/hotel_revenue")

    # 4. City-wise Revenue
    df.groupBy("city")\
        .agg(sum("dynamic_price").alias("total_revenue"))\
        .write.mode("overwrite").save(base+"gold/city_revenue")

    
    # Optimization
    
    spark.sql(f"OPTIMIZE delta.`{base}silver/sales` ZORDER BY (customer_id)")

    log("Gold Layer Completed")

except Exception as e:
    log(f"Gold Layer Failed: {e}")
