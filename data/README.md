# Dataset Description
You are provided with historical sales data for selected Rohlik inventory and date. IDs, sales, total orders and price columns are altered to keep the real values confidential. Some features are not available in test as they are not known at the moment of making the prediction. The task is to forecast the sales column for a given id, constructed from unique_id and date (e. g. id 1226_2024-06-03 from unique_id 1226 and date 2024-06-03), for the test set.

## Files
* sales_train.csv - training set containing the historical sales data for given date and inventory with selected features described below
* sales_test.csv - full testing set
inventory.csv - additional information about inventory like its product (same products across all warehouses share same product unique id and name, but have different unique id)
* solution_format.csv - full submission file in the correct format
* calendar.csv - calendar containing data about holidays or warehouse specific events, some columns are already in the train data but there are additional rows in this file for dates where some warehouses could be closed due to public holiday or Sunday (and therefore they are not in the train set)

## sales_train.csv and sales_test.csv

* unique_id - unique id for inventory
* date - date
* warehouse - warehouse name
* total_orders - historical orders for selected Rohlik warehouse known also for test set as a part of this challenge
* sales - Target value, sales volume (either in pcs or kg) adjusted by availability. The sales with lower availability than 1 are already adjusted to full potential sales by Rohlik internal logic. There might be missing dates both in train and test for a given inventory due to various reasons. This column is missing in test.csv as it is target variable.
* sell_price_main - sell price
* availability - proportion of the day that the inventory was available to customers. The inventory doesn't need to be available at all times. A value of 1 means it was available for the entire day. This column is missing in test.csv as it is not known at the moment of making the prediction.
* type_0_discount, type_1_discount, … - Rohlik is running different types of promo sale actions, these show the percentage of the original price during promo ((original price - current_price) / original_price). Multiple discounts with different type can be run at the same time, but always the highest possible discount among these types is used for sales. Negative discount value should be interpreted as no discount.

## inventory.csv

* unique_id - inventory id for a single keeping unit
* product_unique_id - product id, inventory in each warehouse has the same product unique id (same products across all warehouses has the same product id, but different unique id)
* name - inventory id for a single keeping unit
* L1_category_name, L2_category_name, … - name of the internal category, the higher the number, the more granular information is present
* warehouse - warehouse name

## calendar.csv

* warehouse - warehouse name
* date - date
* holiday_name - name of public holiday if any
* holiday - 0/1 indicating the presence of holidays
* shops_closed - public holiday with most of the shops or large part of shops closed
* winter_school_holidays - winter school holidays
* school_holidays - school holidays

# test_weights.csv

* unique_id - inventory id for a single keeping unit
* weight - weight used for final metric computation