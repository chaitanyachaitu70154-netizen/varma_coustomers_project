import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Customer Segmentation System")
st.write("Customer Purchase Behavior Analysis")

uploaded_file = st.file_uploader(
    "Upload customers.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required = ["AnnualIncome", "SpendingScore"]

    if all(col in df.columns for col in required):
        data = df[required].dropna()

        if len(data) >= 3:
            scaled = StandardScaler().fit_transform(data)

            model = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )

            result = data.copy()
            result["Segment"] = model.fit_predict(scaled)

            st.subheader("Customer Data")
            st.dataframe(result)

            st.subheader("Customer Groups")
            st.bar_chart(
                result["Segment"].value_counts().sort_index()
            )

            st.subheader("Income vs Spending")
            st.scatter_chart(
                result,
                x="AnnualIncome",
                y="SpendingScore",
                color="Segment"
            )

            st.download_button(
                "Download Results CSV",
                result.to_csv(index=False),
                "segmented_customers.csv",
                "text/csv"
            )
        else:
            st.error("At least 3 valid customer records are required.")
    else:
        st.error(
            "CSV must contain AnnualIncome and SpendingScore columns."
        )
else:
    st.info("Upload your customers.csv file to start.")