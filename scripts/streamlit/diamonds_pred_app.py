import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, avg, min, max
import pandas as pd
import altair as alt
import joblib
import os

# --- App Title and Description ---
st.title("Diamond Price Prediction")
st.write(
    "This application predicts the price of a diamond based on its characteristics. "
    "Use the sliders and dropdowns below to enter the diamond's features and click 'Predict Price' to see the estimated value."
)

# --- Snowflake Session ---
session = get_active_session()

# --- Database and Schema Information ---
# This assumes the user has run the setup.sql and notebooks from the repository
DB_NAME = 'ML_HOL_DB'
SCHEMA_NAME = 'ML_HOL_SCHEMA'
STAGE_NAME = 'ML_HOL_ASSETS'
# Using the raw training data table as it contains the unprocessed features
TRAINING_TABLE = 'DIAMONDS_TRAIN' 
MODEL_NAME = 'DIAMONDS_PRICE_PREDICTION'
MODEL_VERSION = 'v1'
PIPELINE_FILE_NAME = 'preprocessing_pipeline.joblib.gz'
LOCAL_PIPELINE_PATH = f'/tmp/{PIPELINE_FILE_NAME}'

# --- Load Preprocessing Pipeline ---
# This function downloads the pipeline from stage and caches it for reuse.
@st.cache_resource
def load_pipeline():
    """Downloads the preprocessing pipeline file from stage and loads it into memory."""
    try:
        session.file.get(f'@{STAGE_NAME}/{PIPELINE_FILE_NAME}', '/tmp')
        pipeline = joblib.load(LOCAL_PIPELINE_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load preprocessing pipeline from stage '@{STAGE_NAME}'.")
        st.error(f"Please ensure '{PIPELINE_FILE_NAME}' exists in the stage. Error: {e}")
        return None

try:
    with st.spinner("Loading preprocessing pipeline..."):
        preprocessing_pipeline = load_pipeline()
    if preprocessing_pipeline:
        st.success("Preprocessing pipeline loaded successfully!")
    else:
        st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the pipeline: {e}")
    st.stop()


# --- Feature Input from User ---
st.sidebar.header("Diamond Features")

# Define the logical order for categorical features to be used in UI and for encoding
CATEGORICAL_ORDER = {
    "CUT": ["FAIR", "GOOD", "VERY_GOOD", "PREMIUM", "IDEAL"],
    "COLOR": ["J", "I", "H", "G", "F", "E", "D"],
    "CLARITY": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
}

def get_feature_ranges(table_name):
    """Gets the min and max for numerical columns and distinct values for categorical columns."""
    # User should input raw features; the pipeline will create derived features like TABLE_PCT
    numerical_features = ['CARAT', 'DEPTH', 'TABLE_PCT', 'X', 'Y', 'Z']
    categorical_features = ['CUT', 'COLOR', 'CLARITY']

    feature_ranges = {}
    df = session.table(table_name)

    # Get min/max for numerical features
    for feature in numerical_features:
        result_row = df.agg(min(col(feature)).alias("MIN_VAL"), max(col(feature)).alias("MAX_VAL")).collect()[0]
        min_val = result_row["MIN_VAL"]
        max_val = result_row["MAX_VAL"]
        feature_ranges[feature] = (float(min_val), float(max_val))

    # Get distinct values for categorical features and sort them logically
    for feature in categorical_features:
        feature_ranges[feature] = CATEGORICAL_ORDER[feature]

    return feature_ranges

try:
    with st.spinner("Loading feature ranges from Snowflake..."):
        feature_ranges = get_feature_ranges(TRAINING_TABLE)

    # Create input widgets for raw features
    carat = st.sidebar.slider("Carat", feature_ranges['CARAT'][0], feature_ranges['CARAT'][1], float(feature_ranges['CARAT'][0] + feature_ranges['CARAT'][1]) / 2)
    depth = st.sidebar.slider("Depth", feature_ranges['DEPTH'][0], feature_ranges['DEPTH'][1], float(feature_ranges['DEPTH'][0] + feature_ranges['DEPTH'][1]) / 2)
    table = st.sidebar.slider("TABLE_PCT", feature_ranges['TABLE_PCT'][0], feature_ranges['TABLE_PCT'][1], float(feature_ranges['TABLE_PCT'][0] + feature_ranges['TABLE_PCT'][1]) / 2)
    x = st.sidebar.slider("X (Length in mm)", feature_ranges['X'][0], feature_ranges['X'][1], float(feature_ranges['X'][0] + feature_ranges['X'][1]) / 2)
    y = st.sidebar.slider("Y (Width in mm)", feature_ranges['Y'][0], feature_ranges['Y'][1], float(feature_ranges['Y'][0] + feature_ranges['Y'][1]) / 2)
    z = st.sidebar.slider("Z (Depth in mm)", feature_ranges['Z'][0], feature_ranges['Z'][1], float(feature_ranges['Z'][0] + feature_ranges['Z'][1]) / 2)

    cut = st.sidebar.selectbox("Cut", feature_ranges['CUT'])
    color = st.sidebar.selectbox("Color", feature_ranges['COLOR'])
    clarity = st.sidebar.selectbox("Clarity", feature_ranges['CLARITY'])

except Exception as e:
    st.error(f"Could not load feature ranges. Make sure the table '{TRAINING_TABLE}' exists in '{DB_NAME}.{SCHEMA_NAME}'. Error: {e}")
    st.stop()


# --- Prediction ---
if st.sidebar.button("Predict Price", type="primary"):
    with st.spinner("Preprocessing input and predicting price..."):
        try:
            # --- FEATURE ENGINEERING VIA PIPELINE ---
            # 1. Create a pandas DataFrame from user inputs with the raw feature names
            raw_features_df = pd.DataFrame(
                [[carat, cut, color, clarity, depth, table, x, y, z]],
                columns=['CARAT', 'CUT', 'COLOR', 'CLARITY', 'DEPTH', 'TABLE_PCT', 'X', 'Y', 'Z']
            )

            # 2. Run the raw features through the loaded preprocessing pipeline
            processed_features_df = preprocessing_pipeline.fit(raw_features_df).transform(raw_features_df)
            
            # 3. Get the processed feature values for the SQL query
            sql_column_order = ['CUT_OE', 'COLOR_OE', 'CLARITY_OE', 'CARAT', 'DEPTH', 'TABLE_PCT', 'X', 'Y', 'Z']
            feature_values = processed_features_df[sql_column_order].iloc[0].values.tolist()
            values_str = ', '.join(map(str, feature_values))
            
            # --- PREDICTION ---
            # Construct the SQL to call the prediction model from the registry
            # The model expects features in the order output by the pipeline
            prediction_sql = f"""
                WITH model_version_alias AS MODEL {MODEL_NAME} VERSION {MODEL_VERSION}
                SELECT model_version_alias!predict(
                    t.CUT_OE, t.COLOR_OE, t.CLARITY_OE, t.CARAT, t.DEPTH, t.TABLE_PCT, t.X, t.Y, t.Z
                )['output_feature_0'] as PREDICTION
                FROM (
                    VALUES ({values_str})
                ) AS t(CUT_OE, COLOR_OE, CLARITY_OE, CARAT, DEPTH, TABLE_PCT, X, Y, Z)
            """

            result_df = session.sql(prediction_sql).collect()
            predicted_price = result_df[0]['PREDICTION']

            st.metric("Predicted Diamond Price", f"${float(predicted_price):,.2f}")

            # --- Visualizations ---
            st.subheader("Feature Comparison")

            # Get average values from the training data for visualization
            training_df_vis = session.table(TRAINING_TABLE)
            # FIX: The column name 'TABLE' is a reserved SQL keyword and must be quoted.
            avg_values = training_df_vis.select(
                avg("CARAT").alias("Avg Carat"),
                avg("DEPTH").alias("Avg Depth"),
                avg(col("TABLE_PCT")).alias("Avg Table"),
                avg("X").alias("Avg X"),
                avg("Y").alias("Avg Y"),
                avg("Z").alias("Avg Z")
            ).to_pandas()

            # Prepare data for charting using raw user inputs for interpretability
            user_input_data = {
                'Feature': ['Carat', 'Depth', 'Table_PCT', 'X', 'Y', 'Z'],
                'Value': [carat, depth, table, x, y, z],
                'Source': 'Your Input'
            }
            avg_data = {
                'Feature': ['Carat', 'Depth', 'Table_PCT', 'X', 'Y', 'Z'],
                'Value': [
                    avg_values['Avg Carat'][0],
                    avg_values['Avg Depth'][0],
                    avg_values['Avg Table'][0],
                    avg_values['Avg X'][0],
                    avg_values['Avg Y'][0],
                    avg_values['Avg Z'][0]
                ],
                'Source': 'Training Set Average'
            }

            chart_data = pd.concat([pd.DataFrame(user_input_data), pd.DataFrame(avg_data)])

            # Create the chart
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Feature:N', sort=None),
                y=alt.Y('Value:Q'),
                color='Source:N',
                tooltip=['Feature', 'Value', 'Source']
            ).properties(
                title="Your Diamond vs. The Average Diamond"
            )

            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction. Please ensure the model '{MODEL_NAME}' version '{MODEL_VERSION}' is in the Model Registry and the pipeline file is in stage '@{STAGE_NAME}'. Error: {e}")

else:
    st.info("Adjust the features in the sidebar and click 'Predict Price'.")

# --- Instructions for Setup ---
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Setup Instructions:**
    1. Run the `setup.sql` script from the [GitHub repository](https://github.com/Snowflake-Labs/sfguide-intro-to-machine-learning-with-snowflake-ml-for-python) to create the database, schema, and tables.
    2. Run through the notebooks in the repository to train the model, create the preprocessing pipeline, and deploy them.
    3. Ensure this Streamlit app is running in a Snowflake environment with access to the created database and schema.
    """
)
