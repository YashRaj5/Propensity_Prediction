# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC To support our propensity scoring efforts, we will make use of the [The Complete Journey](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey) dataset, published to Kaggle by Dunnhumby. The dataset consists of numerous files identifying household purchasing activity in combination with various promotional campaigns for about 2,500 households over a nearly 2 year period. The schema of the overall dataset may be represented as follows:
# MAGIC <img src = "https://brysmiwasb.blob.core.windows.net/demos/images/segmentation_journey_schema3.png">
# MAGIC The purpose of this notebook is to establish access to the raw data, make a few adjustments to the transactional data that will serve as the focus of our feature engineering efforts, and explore these data to inform our future work.

# COMMAND ----------

# DBTITLE 1,Download the Data
# MAGIC %run "./data_download"

# COMMAND ----------

# DBTITLE 1,Retrieve Configuration Values
# MAGIC %run "./00_config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as f

# COMMAND ----------

# DBTITLE 1,Resetting the Database Environment
# In this first step, we will reset the database environment within which our data assets will be housed by deleting any existing copies of the database and recreating it anew:
_ =  spark.sql('DROP DATABASE IF EXISTS {0} CASCADE'.format(config['database']))

# COMMAND ----------

# DBTITLE 1,Drop the Feature Store Table if Exists
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()
try:
  fs.drop_table(
    name=f"{config['database']}.propensity_features" # throws value error if Feature Store table does not exist
  )
except ValueError: 
  pass

# COMMAND ----------

# DBTITLE 1,Create Database if Not Exists
_ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database']))

# COMMAND ----------

# DBTITLE 1,Set the current database
spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Access the Data
# MAGIC To make the data available for our analysis, it is [downloaded](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey), extracted and uploaded to the bronze folder of a [cloud-storage mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) named /mnt/propensity. Each file in the dataset is a comma-separated values file with a header which can be read to a table as follows:
# MAGIC
# MAGIC **NOTE** The name of the mount point can be altered from within the 00 notebook housing configurations. We have automated this downloading step for you and use a /tmp/propensity/ storage path throughout this accelerator.

# COMMAND ----------

# DBTITLE 1,Define Function to Create Tables
def create_table(database_name, table_name, dbfs_file_path):
    
  # drop table if exists
  _ = spark.sql('DROP TABLE IF EXISTS `{0}`.`{1}`'.format(database_name, table_name))
 
  # read data from input file
  df = ( 
    spark
     .read
       .csv(
         dbfs_file_path,
         header=True,
         inferSchema=True
         )
    )
  
  # convert day integers to actual dates
  # for each column
  for c in df.columns:
    # if column is an integer day column
    if c.lower().endswith('day'):
      # convert it to a date value
      df = df.withColumn(c, f.expr(f"date_add('2018-01-01', cast({c} as int)-1)"))
  
  # write data to table
  _ = (
    df
     .write
       .format('delta')
       .mode('overwrite')
       .option('overwriteSchema', 'true')
       .saveAsTable('`{0}`.`{1}`'.format(database_name, table_name))
     )

# COMMAND ----------

# MAGIC %md
# MAGIC It's important to note the dates used in this data set are not proper dates. Instead, they are integer values ranging from 1 to 711 where 1 represents the first date in the range and 711 indicates the last. To make it easier to explore how time-oriented values would be employed in later steps, our function converts these to actual dates by making day 1 equal to January 1, 2018 and adjusting the other values relative to this. The use of this specific date is arbitrary and doesn't reflect any known dates associated with the original dataset.
# MAGIC
# MAGIC From there, we might convert the CSV files to accessible tables as follows:

# COMMAND ----------

# DBTITLE 1,Create Tables
create_table( config['database'], 'transactions', '{0}/bronze/transaction_data.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'products', '{0}/bronze/product.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'households', '{0}/bronze/hh_demographic.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'coupons', '{0}/bronze/coupon.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'campaigns', '{0}/bronze/campaign_desc.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'coupon_redemptions', '{0}/bronze/coupon_redempt.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'campaigns_households', '{0}/bronze/campaign_table.csv'.format(config['dbfs_mount']))
create_table( config['database'], 'causal_data', '{0}/bronze/causal_data.csv'.format(config['dbfs_mount']))

# COMMAND ----------

# DBTITLE 1,Verify Tables Exist
display(
    spark.sql('SHOW TABLES IN {0}'.format(config['database']))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Adjust Transactional Data
# MAGIC The transactional data will be the focal point of our analysis. It contains information about what was purchased by each household and when along with various discounts applied at the time of purchase. Some of this information is presented in a manner that is not easily consumable. As such, we will implement some simple logic to sum discounts and combine these with amounts paid to recreate list pricing and make other simply adjustments that make the transactional data a bit easier to consume:

# COMMAND ----------

# DBTITLE 1,Create Adjusted Transactions Table
# MAGIC %sql
# MAGIC  
# MAGIC DROP TABLE IF EXISTS transactions_adj;
# MAGIC  
# MAGIC CREATE TABLE transactions_adj
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     household_key,
# MAGIC     basket_id,
# MAGIC     week_no,
# MAGIC     day,
# MAGIC     trans_time,
# MAGIC     store_id,
# MAGIC     product_id,
# MAGIC     amount_list,
# MAGIC     campaign_coupon_discount,
# MAGIC     manuf_coupon_discount,
# MAGIC     manuf_coupon_match_discount,
# MAGIC     total_coupon_discount,
# MAGIC     instore_discount,
# MAGIC     amount_paid,
# MAGIC     units
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       household_key,
# MAGIC       basket_id,
# MAGIC       week_no,
# MAGIC       day,
# MAGIC       trans_time,
# MAGIC       store_id,
# MAGIC       product_id,
# MAGIC       COALESCE(sales_value - retail_disc - coupon_disc - coupon_match_disc,0.0) as amount_list,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_match_disc,0.0) = 0.0 THEN -1 * COALESCE(coupon_disc,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as campaign_coupon_discount,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_match_disc,0.0) != 0.0 THEN -1 * COALESCE(coupon_disc,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as manuf_coupon_discount,
# MAGIC       -1 * COALESCE(coupon_match_disc,0.0) as manuf_coupon_match_discount,
# MAGIC       -1 * COALESCE(coupon_disc - coupon_match_disc,0.0) as total_coupon_discount,
# MAGIC       COALESCE(-1 * retail_disc,0.0) as instore_discount,
# MAGIC       COALESCE(sales_value,0.0) as amount_paid,
# MAGIC       quantity as units
# MAGIC     FROM transactions
# MAGIC     );

# COMMAND ----------

# DBTITLE 1,Review Adjusted Transactions Data
# MAGIC %sql SELECT * FROM transactions_adj;

# COMMAND ----------


