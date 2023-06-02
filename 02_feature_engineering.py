# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC In this notebook we will leverage transactional data associated with individual households to generate features with which we will train our model and, later, perform inference, i.e. make predictions. Our goal is to predict the likelihood a household will purchase products from a given product category, i.e. commodity-designation, in the next 30 days.
# MAGIC
# MAGIC In an operationalized workflow, we would receive new data into the lakehouse on a periodic, i.e. daily or more frequent basis. As that data arrives, we might recalculate features for propensity scoring and store these for the purpose of making predictions, i.e. performing inference, about the future period. As these features age, they at some point become useful for training new models. This happens at the point that enough new data arrives that we can derive labels for the period they were built to predict.
# MAGIC
# MAGIC To simulate this workflow, we will calculate features for each of the last 30-days of our dataset. We will establish our workflow logic at the top of this notebook and then define a loop at the bottom to persist these data for later use. As part of this, we will be persisting our data to the Databricks Feature Store, a capability in the Databricks platform which simplifies the persistence and retrieval of features.
# MAGIC
# MAGIC **NOTE** In this notebook, we are deriving features exclusively from our transactional sales data in order to keep things simple. The dataset provides access to customer demographic and promotional campaign data from which additional features would typically be derived.

# COMMAND ----------

# DBTITLE 1,Retrieve Configuration Vlaues
# MAGIC %run "./00_config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.window import Window
 
from databricks.feature_store import FeatureStoreClient
 
from datetime import timedelta

# COMMAND ----------

# DBTITLE 1,Setting current Database
spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Feature Generation Logic
# MAGIC Our first step is to define a function to generate features from a dataframe of transactional data passed to it. In our function, we are deriving a generic set of features from the last 30, 60 and 90 day periods of the transactional data as well as from a 30-day period (aligned with the labels we wish to predict) from 1-year back. This is not exhaustive of what we could derive from these data but should give a since of how we might approach feature generation.
# MAGIC
# MAGIC **NOTE** If we were to derive metrics from across the entire set of data, it would be important to adust the starting date from which metrics might be derived so that features generated on different days are calculated from a consistent range of dates.
# MAGIC
# MAGIC It's important to note that we will be deriving these features first from the household level and then from the household-commodity level. The include_commodity argument is used to control which of these levels is employed at feature generation. In later steps, these features will be combined so that each instance in our feature set will contain features for a given household as well as for that household in combination with one of the 308 commodities against which we may calculate propensity scores:

# COMMAND ----------

# DBTITLE 1,Define Function to Derive Features
def get_features(df, include_commodity=False, window=None):
  
  '''
  This function derives a number of features from our transactional data.
  These data are grouped by either just the household_key or the household_key
  and commodity_desc field and are filtered based on a window prescribed
  with the function call.
  
  df: the dataframe containing household transaction history
  
  include_commodity: controls whether data grouped on:
     household_key (include_commodity=False) or 
     household_key and commodity_desc (include_commodity=True)
  
  window: one of four supported string values:
    '30d': derive metrics from the last 30 days of the dataset
    '60d': derive metrics from the last 60 days of the dataset
    '90d': derive metrics from the last 90 days of the dataset
    '1yr': derive metrics from the 30 day period starting 1-year
           prior to the end of the dataset. this aligns with the
           period from which our labels are derived.
  '''
  
  # determine how to group transaction data for metrics calculations
  grouping_fields = ['household_key']
  grouping_suffix = ''
  if include_commodity: 
    grouping_fields += ['commodity_desc']
    grouping_suffix = '_cmd'
    
  # get list of distinct grouping items in the original dataframe
  anchor_df = transactions.select(grouping_fields).distinct()
  
  # identify when dataset starts and ends
  min_day, max_day = (
    df
      .groupBy()
        .agg(
          f.min('day').alias('min_day'), 
          f.max('day').alias('max_day')
          )
      .collect()
    )[0]    
  
  ## print info to support validation
  #print('{0}:\t{1} days in original set between {2} and {3}'.format(window, (max_day - min_day).days + 1, min_day, max_day))
  
  # adjust min and max days based on specified window   
  if window == '30d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=30-1)
    
  elif window == '60d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=60-1)
    
  elif window == '90d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=90-1)
    
  elif window == '1yr':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=365-1)
    max_day = min_day + timedelta(days=30-1)
    
  else:
    raise Exception('unknown window definition')
  
  # determine the number of days in the window
  days_in_window = (max_day - min_day).days + 1
  
  ## print to help with date math validation
  #print('{0}:\t{1} days in adjusted set between {2} and {3}'.format(window, days_in_window, min_day, max_day))
  
  # convert dates to strings to make remaining steps easier
  max_day = max_day.strftime('%Y-%m-%d')
  min_day = min_day.strftime('%Y-%m-%d')
  
  # derive summary features from set
  summary_df = (
    df
      .filter(f.expr(f"day between '{min_day}' and '{max_day}'")) # constrain to window
      .groupBy(grouping_fields)
        .agg(
          
          # summary metrics
          f.countDistinct('day').alias('days'), 
          f.countDistinct('basket_id').alias('baskets'),
          f.count('product_id').alias('products'), 
          f.count('*').alias('line_items'),
          f.sum('amount_list').alias('amount_list'),
          f.sum('instore_discount').alias('instore_discount'),
          f.sum('campaign_coupon_discount').alias('campaign_coupon_discount'),
          f.sum('manuf_coupon_discount').alias('manuf_coupon_discount'),
          f.sum('total_coupon_discount').alias('total_coupon_discount'),
          f.sum('amount_paid').alias('amount_paid'),
          
          # unique days with activity
          f.countDistinct(f.expr('case when instore_discount >0 then day else null end')).alias('days_with_instore_discount'),
          f.countDistinct(f.expr('case when campaign_coupon_discount >0 then day else null end')).alias('days_with_campaign_coupon_discount'),
          f.countDistinct(f.expr('case when manuf_coupon_discount >0 then day else null end')).alias('days_with_manuf_coupon_discount'),
          f.countDistinct(f.expr('case when total_coupon_discount >0 then day else null end')).alias('days_with_total_coupon_discount'),
          
          # unique baskets with activity
          f.countDistinct(f.expr('case when instore_discount >0 then basket_id else null end')).alias('baskets_with_instore_discount'),
          f.countDistinct(f.expr('case when campaign_coupon_discount >0 then basket_id else null end')).alias('baskets_with_campaign_coupon_discount'),
          f.countDistinct(f.expr('case when manuf_coupon_discount >0 then basket_id else null end')).alias('baskets_with_manuf_coupon_discount'),
          f.countDistinct(f.expr('case when total_coupon_discount >0 then basket_id else null end')).alias('baskets_with_total_coupon_discount'),          
    
          # unique products with activity
          f.countDistinct(f.expr('case when instore_discount >0 then product_id else null end')).alias('products_with_instore_discount'),
          f.countDistinct(f.expr('case when campaign_coupon_discount >0 then product_id else null end')).alias('products_with_campaign_coupon_discount'),
          f.countDistinct(f.expr('case when manuf_coupon_discount >0 then product_id else null end')).alias('products_with_manuf_coupon_discount'),
          f.countDistinct(f.expr('case when total_coupon_discount >0 then product_id else null end')).alias('products_with_total_coupon_discount'),          
    
          # unique line items with activity
          f.sum(f.expr('case when instore_discount >0 then 1 else null end')).alias('line_items_with_instore_discount'),
          f.sum(f.expr('case when campaign_coupon_discount >0 then 1 else null end')).alias('line_items_with_campaign_coupon_discount'),
          f.sum(f.expr('case when manuf_coupon_discount >0 then 1 else null end')).alias('line_items_with_manuf_coupon_discount'),
          f.sum(f.expr('case when total_coupon_discount >0 then 1 else null end')).alias('line_items_with_total_coupon_discount')          
          )    
    
      # per-day ratios
      .withColumn(f'baskets_per_day', f.expr('baskets/days'))
      .withColumn(f'products_per_day{window_suffix}', f.expr('products/days'))
      .withColumn(f'line_items_per_day', f.expr('line_items/days'))
      .withColumn(f'amount_list_per_day', f.expr('amount_list/days'))
      .withColumn(f'instore_discount_per_day', f.expr('instore_discount/days'))
      .withColumn(f'campaign_coupon_discount_per_day', f.expr('campaign_coupon_discount/days'))
      .withColumn(f'manuf_coupon_discount_per_day', f.expr('manuf_coupon_discount/days'))
      .withColumn(f'total_coupon_discount_per_day', f.expr('total_coupon_discount/days'))
      .withColumn(f'amount_paid_per_day', f.expr('amount_paid/days'))
      .withColumn(f'days_with_instore_discount_per_days', f.expr('days_with_instore_discount/days'))
      .withColumn(f'days_with_campaign_coupon_discount_per_days', f.expr('days_with_campaign_coupon_discount/days'))
      .withColumn(f'days_with_manuf_coupon_discount_per_days', f.expr('days_with_manuf_coupon_discount/days'))
      .withColumn(f'days_with_total_coupon_discount_per_days', f.expr('days_with_total_coupon_discount/days'))
    
      # per-day-in-set ratios
      .withColumn(f'days_to_days_in_set', f.expr(f'days/{days_in_window}'))
      .withColumn(f'baskets_per_days_in_set', f.expr(f'baskets/{days_in_window}'))
      .withColumn(f'products_to_days_in_set', f.expr(f'products/{days_in_window}'))
      .withColumn(f'line_items_per_days_in_set', f.expr(f'line_items/{days_in_window}'))
      .withColumn(f'amount_list_per_days_in_set', f.expr(f'amount_list/{days_in_window}'))
      .withColumn(f'instore_discount_per_days_in_set', f.expr(f'instore_discount/{days_in_window}'))
      .withColumn(f'campaign_coupon_discount_per_days_in_set', f.expr(f'campaign_coupon_discount/{days_in_window}'))
      .withColumn(f'manuf_coupon_discount_per_days_in_set', f.expr(f'manuf_coupon_discount/{days_in_window}'))
      .withColumn(f'total_coupon_discount_per_days_in_set', f.expr(f'total_coupon_discount/{days_in_window}'))
      .withColumn(f'amount_paid_per_days_in_set', f.expr(f'amount_paid/{days_in_window}'))
      .withColumn(f'days_with_instore_discount_per_days_in_set', f.expr(f'days_with_instore_discount/{days_in_window}'))
      .withColumn(f'days_with_campaign_coupon_discount_per_days_in_set', f.expr(f'days_with_campaign_coupon_discount/{days_in_window}'))
      .withColumn(f'days_with_manuf_coupon_discount_per_days_in_set', f.expr(f'days_with_manuf_coupon_discount/{days_in_window}'))
      .withColumn(f'days_with_total_coupon_discount_per_days_in_set', f.expr(f'days_with_total_coupon_discount/{days_in_window}'))
 
      # per-basket ratios
      .withColumn('products_per_basket', f.expr('products/baskets'))
      .withColumn('line_items_per_basket', f.expr('line_items/baskets'))
      .withColumn('amount_list_per_basket', f.expr('amount_list/baskets'))      
      .withColumn('instore_discount_per_basket', f.expr('instore_discount/baskets'))  
      .withColumn('campaign_coupon_discount_per_basket', f.expr('campaign_coupon_discount/baskets')) 
      .withColumn('manuf_coupon_discount_per_basket', f.expr('manuf_coupon_discount/baskets'))
      .withColumn('total_coupon_discount_per_basket', f.expr('total_coupon_discount/baskets'))    
      .withColumn('amount_paid_per_basket', f.expr('amount_paid/baskets'))
      .withColumn('baskets_with_instore_discount_per_baskets', f.expr('baskets_with_instore_discount/baskets'))
      .withColumn('baskets_with_campaign_coupon_discount_per_baskets', f.expr('baskets_with_campaign_coupon_discount/baskets'))
      .withColumn('baskets_with_manuf_coupon_discount_per_baskets', f.expr('baskets_with_manuf_coupon_discount/baskets'))
      .withColumn('baskets_with_total_coupon_discount_per_baskets', f.expr('baskets_with_total_coupon_discount/baskets'))
      
      # per-product ratios
      .withColumn('line_items_per_product', f.expr('line_items/products'))
      .withColumn('amount_list_per_product', f.expr('amount_list/products'))      
      .withColumn('instore_discount_per_product', f.expr('instore_discount/products'))  
      .withColumn('campaign_coupon_discount_per_product', f.expr('campaign_coupon_discount/products')) 
      .withColumn('manuf_coupon_discount_per_product', f.expr('manuf_coupon_discount/products'))
      .withColumn('total_coupon_discount_per_product', f.expr('total_coupon_discount/products'))    
      .withColumn('amount_paid_per_product', f.expr('amount_paid/products'))
      .withColumn('products_with_instore_discount_per_product', f.expr('products_with_instore_discount/products'))
      .withColumn('products_with_campaign_coupon_discount_per_product', f.expr('products_with_campaign_coupon_discount/products'))
      .withColumn('products_with_manuf_coupon_discount_per_product', f.expr('products_with_manuf_coupon_discount/products'))
      .withColumn('products_with_total_coupon_discount_per_product', f.expr('products_with_total_coupon_discount/products'))
      
      # per-line_item ratios
      .withColumn('amount_list_per_line_item', f.expr('amount_list/line_items'))      
      .withColumn('instore_discount_per_line_item', f.expr('instore_discount/line_items'))  
      .withColumn('campaign_coupon_discount_per_line_item', f.expr('campaign_coupon_discount/line_items')) 
      .withColumn('manuf_coupon_discount_per_line_item', f.expr('manuf_coupon_discount/line_items'))
      .withColumn('total_coupon_discount_per_line_item', f.expr('total_coupon_discount/line_items'))    
      .withColumn('amount_paid_per_line_item', f.expr('amount_paid/line_items'))
      .withColumn('products_with_instore_discount_per_line_item', f.expr('products_with_instore_discount/line_items'))
      .withColumn('products_with_campaign_coupon_discount_per_line_item', f.expr('products_with_campaign_coupon_discount/line_items'))
      .withColumn('products_with_manuf_coupon_discount_per_line_item', f.expr('products_with_manuf_coupon_discount/line_items'))
      .withColumn('products_with_total_coupon_discount_per_line_item', f.expr('products_with_total_coupon_discount/line_items'))    
    
      # amount_list ratios
      .withColumn('campaign_coupon_discount_to_amount_list', f.expr('campaign_coupon_discount/amount_list'))
      .withColumn('manuf_coupon_discount_to_amount_list', f.expr('manuf_coupon_discount/amount_list'))
      .withColumn('total_coupon_discount_to_amount_list', f.expr('total_coupon_discount/amount_list'))
      .withColumn('amount_paid_to_amount_list', f.expr('amount_paid/amount_list'))
      )
 
 
  # derive days-since metrics
  dayssince_df = (
    df
      .filter(f.expr(f"day <= '{max_day}'"))
      .groupBy(grouping_fields)
        .agg(
          f.min(f.expr(f"'{max_day}' - case when instore_discount >0 then day else '{min_day}' end")).alias('days_since_instore_discount'),
          f.min(f.expr(f"'{max_day}' - case when campaign_coupon_discount >0 then day else '{min_day}' end")).alias('days_since_campaign_coupon_discount'),
          f.min(f.expr(f"'{max_day}' - case when manuf_coupon_discount >0 then day else '{min_day}' end")).alias('days_since_manuf_coupon_discount'),
          f.min(f.expr(f"'{max_day}' - case when total_coupon_discount >0 then day else '{min_day}' end")).alias('days_since_total_coupon_discount')
          )
      )
  
  # combine metrics with anchor set to form return set 
  ret_df = (
    anchor_df
      .join(summary_df, on=grouping_fields, how='leftouter')
      .join(dayssince_df, on=grouping_fields, how='leftouter')
    )
  
  # rename fields based on control parameters
  for c in ret_df.columns:
    if c not in grouping_fields: # don't rename grouping fields
      ret_df = ret_df.withColumn(c, f.col(c).cast(DoubleType())) # cast all metrics as doubles to avoid confusion as categoricals
      ret_df = ret_df.withColumnRenamed(c,f'{c}{grouping_suffix}{window_suffix}')
 
  return ret_df

# COMMAND ----------


