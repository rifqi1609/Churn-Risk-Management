import streamlit as st
import pandas as pd
import numpy as np
import dill
from datetime import datetime

# Price Transformation
def add_price_features(X, price_df):
    types = ['var', 'fix']
    agg_dict = {}

    for t in types:
        off_col = f'price_off_peak_{t}'
        mid_col = f'price_mid_peak_{t}'
        peak_col = f'price_peak_{t}'
        
        mid_diff_key = f'mid_off_{t}_diff'
        peak_diff_key = f'peak_off_{t}_diff'
        
        price_df[mid_diff_key] = np.where(price_df[mid_col] > 0, 
                                            price_df[mid_col] - price_df[off_col], 0)
        
        price_df[peak_diff_key] = np.where(price_df[peak_col] > 0, 
                                             price_df[peak_col] - price_df[off_col], 0)

        agg_dict[mid_diff_key] = ['mean', 'max']
        agg_dict[peak_diff_key] = ['mean', 'max']

    grouped_prices = price_df.groupby('id').agg(agg_dict)

    grouped_prices.columns = [f"{stat}_{col}" for col, stat in grouped_prices.columns]
    grouped_prices = grouped_prices.reset_index()

    X = pd.merge(X, grouped_prices, on='id', how='left')

    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    return X

# Categorical Transformation
def transform_categorical_features(X):
    channel_mapping = {
        'foosdfpfkusacimwkcsosbicdxkicaua': 'Direct Sales',
        'usilxuppasemubllopkaafesmlibmsdf': 'Online',
        'lmkebamcaaclubfxadlmueccxoimlema': 'Broker',
        'ewpakwlliwisiwduibdlfmalxowmwpci': 'Group Partnership',
        'epumfxlbckeskwekxbiuasklxalciiuu': 'Other Channels',
        'fixdbufsefwooaasfcxdxadsiekoceaa': 'Other Channels',
        'sddiedcslfslkckwlfkdpoeeailfpeds': 'Other Channels',
        'MISSING': 'Untracked Channels'
    }
    X['channel_sales'] = X['channel_sales'].replace(channel_mapping)
    
    origin_mapping = {
        'kamkkxfxxuwbdslkwifmmcsiusiuosws': 'A (New Year)',
        'lxidpiddsbxsbosboudacockeimpuepw': 'B (Strategic Partnership)',
        'ldkssxwpmemidmecebumciepifcamkci': 'C (Exclusive Offers)',
        'MISSING': 'Other Origins',
        'usapbepcfoloekilkwsdiboslwaxobdp': 'Other Origins',
        'ewxeelcelemmiwuafmddpobolfuxioce': 'Other Origins'
    }
    X['origin_up'] = X['origin_up'].replace(origin_mapping)
    
    return X

# Determine Intervention Actions
def intervention_actions (row):
    decile = row['Churn Decile']
    value_segment = row['Customer Value Quantiles']
    
    # High Risk
    if decile <= 7:
        if value_segment == 'High':
            return "1 - Personalized Intervention"
        else:
            return "3 - Business as Usual"

    # Low Risk
    elif decile >= 9:
        if value_segment == 'High':
            return "2 - Special Loyalty Program"
        else:
            return "3 - Business as Usual"

    # Mid Risk
    else:
        return "3 - Business as Usual"
    
# Data Type Transformation
def transform_date_features(X):
    date_columns = ['date_activ', 'date_end', 'date_modif_prod', 'date_renewal']
    ref_date = datetime(2026, 1, 1)
    
    for col in date_columns:
        dates = pd.to_datetime(X[col], format='%Y-%m-%d')
        
        year_diff = ref_date.year - dates.dt.year
        month_diff = ref_date.month - dates.dt.month
        months = year_diff * 12 + month_diff
        months -= (ref_date.day < dates.dt.day).astype(int)
        
        if col == 'date_activ':
            X['months_activ'] = months
        elif col == 'date_end':
            X['months_to_end'] = -months
        elif col == 'date_modif_prod':
            X['months_modif_prod'] = months
        elif col == 'date_renewal':
            X['months_renewal'] = months
            
    remove = [
        'date_activ', 
        'date_end', 
        'date_modif_prod', 
        'date_renewal',
        'id']
    
    X = X.drop(columns=remove)
    return X

@st.cache_resource
def load_model():
    from datetime import datetime
    import __main__
    __main__.datetime = datetime

    with open('deployment/final_model.pkl', 'rb') as f:
        model = dill.load(f)

    for name, step in model.steps:
        if hasattr(step, 'func') and step.func is not None:
            step.func.__globals__['datetime'] = datetime

    return model

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Customer Churn Prediction")
st.markdown("Let me help you handle potential churned customer.")

st.sidebar.header("Input Data")
upload_file = st.sidebar.file_uploader("Upload Customer Data", type=['csv'])

if upload_file is not None:
    X_inf = pd.read_csv(upload_file)
    st.write("### Data uploaded:", X_inf.head())
    
    if st.button("Predict Churn"):
        with st.spinner('Processing...'):
            try:
                price_df = pd.read_csv('deployment/clean_price_data.csv') 
                
                # Load Model
                model = load_model()
                
                # Prediction Process
                prediction_prob = model.predict_proba(X_inf)[:, 1]
                X_inf['Churn_Probability']=prediction_prob

                X_inf = add_price_features(X_inf, price_df)
                X_inf = transform_categorical_features(X_inf)
        
                # Grouping Bin
                prob_bins = [
                    -np.inf,
                    0.122897,
                    0.181997,
                    0.227403,
                    0.275547,
                    0.331891,
                    0.383967,
                    0.438906,
                    0.508089,
                    0.606386,
                    np.inf]

                decile_labels = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

                X_inf['Churn Decile'] = pd.cut(
                    X_inf['Churn_Probability'], 
                    bins=prob_bins, 
                    labels=decile_labels)

                margin_bins = [
                    -np.inf,
                    93.99,
                    207.57,
                    422.38,
                    np.inf]

                margin_labels = ['Low', 'Lower-Mid', 'Upper-Mid', 'High']

                X_inf['Customer Value Quantiles'] = pd.cut(
                    X_inf['net_margin'], 
                    bins=margin_bins, 
                    labels=margin_labels)
                
                # Data Preparation
                X_inf = add_price_features(X_inf, price_df)
                X_inf = transform_categorical_features(X_inf)

                # Recommend Actions
                X_inf['Recommedation Action'] = X_inf.apply(intervention_actions, axis=1)
                X_inf = X_inf.sort_values(by='Recommedation Action', ascending=True).reset_index(drop=True)
                
                # Summary
                st.success("Process Done!")
                st.write("### Churn Handling Recommendation:", X_inf[['id', 'Recommedation Action']])
                
                # Download Button
                csv = X_inf.to_csv(index=False).encode('utf-8')
                st.download_button("Download Recommendation Action", csv, "recommendation_action.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Please upload customer data on sidebar for prediction.")