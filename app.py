import streamlit as st
import pandas as pd
import numpy as np
import joblib
from transformers import ColumnStringConverter, AttributeCombiner

# Load the model
final_pipeline = joblib.load('final_pipeline_model.joblib')
preprocessing_pipeline = final_pipeline.named_steps['preprocessing']
# Define the Streamlit app
def main():
    st.title("Click-Through Rate Prediction App")
    st.subheader("Unlock the potential of your email campaigns with AI-driven predictions!")
    numerical_features=['day_of_week','subject_len', 'body_len', 'mean_paragraph_len','no_of_CTA', 'mean_CTA_len', 'no_of_image','no_of_quotes','no_of_emoticons']
    categorical_features=['is_discount', 'is_price', 'is_urgency', 'is_personalised',  'is_weekend', 'sender', 'category', 'product', 'target_audience']
    one_hot_features=['times_of_day']

    # Upload file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Filter the DataFrame for selected columns
        selected_columns = numerical_features + categorical_features + one_hot_features
        df_filtered = df[selected_columns]

        # Preprocess data
        X = df_filtered  # Assuming the target column is not included in X
        st.write(X)
        if st.button('Predict'):
            # X_transformed = final_pipeline.named_steps['preprocessing'].transform(X)
            # Make predictions
            predictions = final_pipeline.predict(X)

            # Add predictions to DataFrame
            df_filtered['click_rate'] = predictions

            # Display original dataset with predictions
            st.write(df_filtered)


if __name__ == "__main__":
    main()
