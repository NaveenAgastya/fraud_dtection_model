import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st

# Function to simulate transaction data
def simulate_data():
    np.random.seed(42)

    # Simulate 1000 normal transactions
    normal_transactions = np.random.normal(loc=50, scale=15, size=(1000, 2))
    normal_df = pd.DataFrame(normal_transactions, columns=['amount', 'time_spent'])

    # Simulate 50 fraudulent transactions
    fraud_transactions = np.random.normal(loc=150, scale=30, size=(50, 2))
    fraud_df = pd.DataFrame(fraud_transactions, columns=['amount', 'time_spent'])

    # Combine into a single dataset
    transactions_df = pd.concat([normal_df, fraud_df], ignore_index=True)
    transactions_df['is_fraud'] = [0] * 1000 + [1] * 50

    return transactions_df

# Function to build and apply the anomaly detection model
def detect_anomalies(transactions_df):
    X = transactions_df[['amount', 'time_spent']]
    model = IsolationForest(contamination=0.05)  # Expecting 5% fraud cases
    model.fit(X)
    transactions_df['anomaly_score'] = model.decision_function(X)
    transactions_df['is_anomaly'] = model.predict(X)
    transactions_df['is_anomaly'] = transactions_df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)

    return transactions_df

# Main Streamlit app
def main():
    st.title("AI-Powered Fraud Detection Demo")

    st.sidebar.header("Simulation Settings")
    st.sidebar.write("Use the settings below to customize the data simulation:")

    # Slider for data size
    num_normal = st.sidebar.slider("Number of Normal Transactions", 500, 2000, 1000)
    num_fraud = st.sidebar.slider("Number of Fraudulent Transactions", 10, 200, 50)

    if st.sidebar.button("Simulate Data"):
        # Simulate data based on user input
        transactions_df = simulate_data()
        transactions_df = detect_anomalies(transactions_df)

        st.subheader("Simulated Transactions")
        st.write(transactions_df.head())

        st.subheader("Real-Time Monitoring Simulation")
        for i in range(len(transactions_df)):
            transaction = transactions_df.iloc[i]
            if transaction['is_anomaly']:
                st.write(f"**Suspicious transaction detected:** Amount = {transaction['amount']:.2f}, Time Spent = {transaction['time_spent']:.2f}")

        st.subheader("Summary")
        st.write(f"Total Transactions: {len(transactions_df)}")
        st.write(f"Detected Anomalies: {transactions_df['is_anomaly'].sum()}")

if __name__ == "__main__":
    main()
