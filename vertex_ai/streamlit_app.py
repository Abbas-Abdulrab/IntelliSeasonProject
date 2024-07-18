import streamlit as st
import requests
import pandas as pd
import json
from streamlit.components.v1 import iframe
from streamlit_option_menu import option_menu
from utils.data_loading import load_csv 
from utils.data_cleaning import DataCleaner
import os
import threading
from flask_server import app

def fetch_data():
    try:

        response = requests.get('http://localhost:5000/data')
        st.write(f"Response status code: {response.status_code}")
        st.write(f"Response headers: {response.headers}")
        st.write(f"Response content preview: {response.text[:500]}")  # Display first 500 characters for preview

        if response.status_code == 200:
            try:
                data = response.json()
                st.write("Data successfully fetched from server:")
                st.json(data)  # Display JSON data for debugging

                # Ensure the JSON data is properly parsed into a list of dictionaries
                if isinstance(data, str):
                    data = json.loads(data)

                # Flatten the JSON data manually
                flat_data = []
                for item in data:
                    # Flatten the nested 'predicted_PSData' field
                    if 'predicted_PSData' in item and isinstance(item['predicted_PSData'], dict):
                        item['predicted_PSData'] = item['predicted_PSData']['value']
                    flat_data.append(item)

                # Convert the flattened data to a DataFrame
                df = pd.DataFrame(flat_data)
                st.write("DataFrame created from JSON data:")
                # st.dataframe(df)  # Display DataFrame for debugging
                return df
            except ValueError as e:
                st.error(f"Failed to decode JSON response: {e}")
                return None
        elif response.status_code == 204:
            st.error("No data found.")
            return None
        else:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None
    except ValueError as e:
        st.error(f"Failed to decode JSON response: {e}")
        return None

def main():
    st.title("BigQuery Data Visualization")

    with st.sidebar:
        st.header("Options Menu")
        selected = option_menu(
            'IntelliSeason', ["Auto Forecast", "Compare Forecast", "History"], 
            icons=['play-btn', 'search', 'info-circle'], menu_icon='intersect', default_index=0
        )


    if selected == "Auto Forecast":
        st.subheader("Auto Forecast")
        specific_dir = os.path.join(os.getcwd(), 'uploads')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if not os.path.exists(specific_dir):
            os.makedirs(specific_dir)

        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            cleaner = DataCleaner(df)
            
            if df is not None:
                file_path = os.path.join(specific_dir, uploaded_file.name)
                cleaned_file_path = os.path.join(specific_dir, f"cleaned_{uploaded_file.name}")
                df.to_csv(cleaned_file_path, index=False)
                
                st.write(f"Cleaned file saved to: {cleaned_file_path}")
                
                date_column = st.selectbox("Select date column", df.columns)
                target_column = st.selectbox("Select column to forecast", df.columns)
                time_series_identifier = st.selectbox("Select time series identifier", df.columns)
                additional_columns = st.multiselect("Select additional columns for forecasting", df.columns.difference([date_column, target_column, time_series_identifier]))
                # time_series_identifier = st.selectbox("Select time series identifier", df.columns)
                period = st.number_input("Forecast Period (days)", min_value=1, value=30)
                # optimization_objective = st.selectbox("Optimization Objective", ["minimize-rmse", "minimize-mae"])
                # budget_milli_node_hours = st.number_input("Budget (milli node hours)", min_value=100, step=100)
                df = cleaner.clean_data(date_column)
                if st.button('Start AutoML Training'):
                    data = {
                        'file_path': cleaned_file_path,
                        'target_column': target_column,
                        # 'optimization_objective': optimization_objective,
                        # 'budget_milli_node_hours': budget_milli_node_hours,
                        'period': period,
                        'date_column': date_column,
                        'time_series_identifier': time_series_identifier,
                        'additional_columns': additional_columns
                    }

                    response = requests.post('http://127.0.0.1:5000/automl', data=data)

                    if response.status_code == 200:
                        st.success(response.text)
                    else:
                        st.error(f"Error: {response.text}")   

    elif selected == "History":
        df = fetch_data()
        if df is not None:
            st.write("Available columns:", df.columns.tolist())

            required_columns = ['PSData', 'predicted_PSData', 'Date', 'predicted_on_Date']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"Missing columns in the dataset: {', '.join(missing_columns)}. Please check the available columns above.")
            else:
                # Convert 'Date' column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                    st.write("Converted 'Date' column to datetime")

                # Convert 'predicted_on_Date' column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df['predicted_on_Date']):
                    df['predicted_on_Date'] = pd.to_datetime(df['predicted_on_Date'], dayfirst=True)
                    st.write("Converted 'predicted_on_Date' column to datetime")

                # Ensure PSData and predicted_PSData are numeric
                df['PSData'] = pd.to_numeric(df['PSData'], errors='coerce')
                df['predicted_PSData'] = pd.to_numeric(df['predicted_PSData'], errors='coerce')

                # Set 'Date' column as the index
                df.set_index('Date', inplace=True)
                st.write("Set 'Date' column as index")

                st.write("Actual vs Predicted Data")
                st.line_chart(df[['PSData', 'predicted_PSData']])

                st.write("Detailed Data")
                st.dataframe(df)


if __name__ == '__main__':
    main()