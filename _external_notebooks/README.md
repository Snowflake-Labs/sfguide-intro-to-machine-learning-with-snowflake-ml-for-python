## Using an External IDE
Want to try out the [Getting Started with Snowflake ML for Python Quickstart](https://quickstarts.snowflake.com/guide/intro_to_machine_learning_with_snowpark_ml_for_python/index.html?index=..%2F..index#0) using an external IDE? This folder contains three notebooks that are ready-to-use in your Python environment:
- [1_snowflake_ml_data_ingest.ipynb](./1_snowflake_ml_data_ingest.ipynb)
- [2_snowflake_ml_feature_transformations.ipynb](./2_snowflake_ml_feature_transformations.ipynb)
- [3_snowflake_ml_model_training_inference.ipynb](./3_snowflake_ml_model_training_inference.ipynb)

### Prerequisites
- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed
    - Download the [git repo](https://github.com/Snowflake-Labs/sfguide-intro-to-machine-learning-with-snowpark-ml-for-python)
- [Anaconda](https://www.anaconda.com/) installed
- [Python 3.11](https://www.python.org/downloads/) installed
    - Note that you will be creating a Python environment with 3.11
- A Snowflake account with [Anaconda Packages enabled by ORGADMIN](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-packages.html#using-third-party-packages-from-anaconda). If you do not have a Snowflake account, you can register for a [free trial account](https://signup.snowflake.com/).
- A Snowflake account login with a role that has the ability to create database, schema, tables, stages, user-defined functions, and stored procedures. If not, you will need to register for a free trial or use a different role.

### Set up the Snowflake environment
To get started, first login to Snowsight and run the following [setup.sql](./setup.sql) in a SQL worksheet (we need  to create the database, warehouse, schema, etc. that we will use for our ML project).
```sql
USE ROLE SYSADMIN;
CREATE OR REPLACE WAREHOUSE ML_HOL_WH; --by default, this creates an XS Standard Warehouse
CREATE OR REPLACE DATABASE ML_HOL_DB;
CREATE OR REPLACE SCHEMA ML_HOL_SCHEMA;
CREATE OR REPLACE STAGE ML_HOL_ASSETS; --to store model assets

-- create csv format
CREATE FILE FORMAT IF NOT EXISTS ML_HOL_DB.ML_HOL_SCHEMA.CSVFORMAT 
    SKIP_HEADER = 1 
    TYPE = 'CSV';

-- create external stage with the csv format to stage the diamonds dataset
CREATE STAGE IF NOT EXISTS ML_HOL_DB.ML_HOL_SCHEMA.DIAMONDS_ASSETS 
    FILE_FORMAT = ML_HOL_DB.ML_HOL_SCHEMA.CSVFORMAT 
    URL = 's3://sfquickstarts/intro-to-machine-learning-with-snowpark-ml-for-python/diamonds.csv';
    -- https://sfquickstarts.s3.us-west-1.amazonaws.com/intro-to-machine-learning-with-snowpark-ml-for-python/diamonds.csv

LS @DIAMONDS_ASSETS;

```

## Set up your Python environment

- Download and install the miniconda installer from https://conda.io/miniconda.html. (OR, you may use any other Python environment with Python 3.11, for example, virtualenv).
- Open a new terminal window and execute the following commands in the same terminal window:
    - Create the conda environment.
        ```
        conda env create -f conda_env.yml 
        ```
    - Activate the conda environment.
        ``` 
        conda activate snowpark-ml-hol 
        ```
    - Optionally start notebook server:
        ``` 
        $ jupyter notebook &> /tmp/notebook.log & 
        ```
- Update [connection.json](./connection.json) with your Snowflake account details and credentials. Here's a sample based on the object names we created in the last step:
    ```
    {
    "account"   : "<your_account_identifier_goes_here>",
    "user"      : "<your_username_goes_here>",
    "password"  : "<your_password_goes_here>",
    "role"      : "ACCOUNTADMIN",
    "warehouse" : "ML_HOL_WH",
    "database"  : "ML_HOL_DB",
    "schema"    : "ML_HOL_SCHEMA"
    }
    ```

    > **Note:** For the account parameter above, specify your account identifier and do not include the snowflakecomputing.com domain name. Snowflake automatically appends this when creating the connection. For more details on that, refer to the documentation.
