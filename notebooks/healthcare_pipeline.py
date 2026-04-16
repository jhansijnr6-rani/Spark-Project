from pyspark.sql.functions import *
from pyspark.sql.window import Window


# Logger

def log(msg):
    print(f"[INFO] {msg}")

# Base path
base = "/Volumes/workspace/default/healthcare/"


# Bronze Layer

try:
    log("Starting Bronze Layer")

    patients = spark.read.csv(base+"patients.csv", header=True, inferSchema=True)
    visits = spark.read.csv(base+"visits.csv", header=True, inferSchema=True)
    treatments = spark.read.csv(base+"treatments.csv", header=True, inferSchema=True)

    patients.write.format("delta").mode("overwrite").save(base+"bronze/patients")
    visits.write.format("delta").mode("overwrite").save(base+"bronze/visits")
    treatments.write.format("delta").mode("overwrite").save(base+"bronze/treatments")

    log("Bronze Layer Completed")

except Exception as e:
    log(f"Bronze Layer Failed: {e}")



# Silver Layer

try:
    log("Starting Silver Layer")

    pat = patients.dropDuplicates()
    vis = visits.withColumn("visit_date", col("visit_date").cast("date"))
    trt = treatments.dropDuplicates()

    df = vis.join(pat, "patient_id")\
            .join(trt, "visit_id")

    
    # Readmission Logic
    
    window_spec = Window.partitionBy("patient_id").orderBy("visit_date")

    df = df.withColumn("prev_visit", lag("visit_date").over(window_spec))\
           .withColumn("days_diff", datediff("visit_date", "prev_visit"))\
           .withColumn("readmission",
               when(col("days_diff") <= 30, 1).otherwise(0)
           )

    
    # Risk Classification
    
    df = df.withColumn("risk_level",
        when(col("cost") > 5000, "High")
        .when(col("cost") > 2000, "Medium")
        .otherwise("Low")
    )

    
    # Data Quality Checks
    
    assert df.filter(col("patient_id").isNull()).count() == 0
    assert df.filter(col("visit_id").isNull()).count() == 0

    df.write.format("delta")\
        .partitionBy("visit_date")\
        .mode("overwrite")\
        .save(base+"silver/patient_data")

    log("Silver Layer Completed")

except Exception as e:
    log(f"Silver Layer Failed: {e}")



# Gold Layer

try:
    log("Starting Gold Layer")

    df = spark.read.format("delta").load(base+"silver/patient_data")

    # 1. Readmission Rate
    df.groupBy()\
      .agg(avg("readmission").alias("readmission_rate"))\
      .write.mode("overwrite").save(base+"gold/readmission_rate")

    # 2. Patient Total Cost
    df.groupBy("patient_id")\
      .agg(sum("cost").alias("total_cost"))\
      .write.mode("overwrite").save(base+"gold/patient_cost")

    # 3. Risk Distribution
    df.groupBy("risk_level")\
      .count()\
      .write.mode("overwrite").save(base+"gold/risk_distribution")

    # 4. Department-wise Cost
    df.groupBy("department")\
      .agg(sum("cost").alias("total_cost"))\
      .write.mode("overwrite").save(base+"gold/department_cost")

    
    # Optimization
    
    spark.sql(f"OPTIMIZE delta.`{base}silver/patient_data` ZORDER BY (patient_id)")

    log("Gold Layer Completed")

except Exception as e:
    log(f"Gold Layer Failed: {e}")
