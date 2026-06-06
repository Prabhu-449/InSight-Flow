import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="InsightFlow", page_icon="📊", layout="wide")
# Dashboard Title and Description
st.title("📊 InsightFlow")

st.markdown("""
## ML Model Experimentation & Analytics Dashboard

Upload datasets, perform EDA, run ML models,
customer segmentation, and A/B testing simulations.
""")
# st.write("This dashboard allows you to upload your datasets, perform exploratory data analysis (EDA), run machine learning models, conduct customer segmentation, and simulate A/B testing scenarios. Use the sidebar to filter and explore your data.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Initialize selected_region to "All" to avoid reference before assignment
selected_region = "All"

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    
    st.sidebar.header("🔍 Filters")

    if "Region" in df.columns:

        selected_region = st.sidebar.selectbox(
            "Select Region",
            ["All"] + list(df["Region"].unique())
        )

    if selected_region != "All":
        df = df[df["Region"] == selected_region]

    st.success("Dataset uploaded successfully!")

    st.write("File Name:", uploaded_file.name)
    
#DATA PREVIEW
    st.markdown("---")

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

#DATA EXPLORATION
    st.markdown("---")

    st.subheader("📊 Dataset Information")

    st.write("Rows and Columns:", df.shape)

    st.write("Column Names:")

    st.write(df.columns.tolist())

    st.markdown("---")

    st.subheader("🧹 Missing Values")

    st.write(df.isnull().sum())

    st.markdown("---")

    st.subheader("📌 Data Types")

    st.write(df.dtypes.astype(str))

    st.markdown("---")

    st.subheader("📈 Statistical Summary")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        st.write(numeric_df.describe())
    else:
        st.warning("No numeric columns found.")

#DATA VISUALIZATION
    st.markdown("---")

    st.subheader("📊 Data Visualization")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_columns) > 0:

        selected_column = st.selectbox("Select numeric column", numeric_columns)

        fig, ax = plt.subplots()

        ax.hist(df[selected_column])

        ax.set_title(f"Distribution of {selected_column}")

        ax.set_xlabel(selected_column)

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

    else:
        st.warning("No numeric columns available.")

#KPIs
    st.markdown("---")

    st.subheader("📌 Business KPIs")            

    if "Sales" in df.columns:
        total_sales = df["Sales"].sum()
    else:
        total_sales = 0

    if "Profit" in df.columns:
        total_profit = df["Profit"].sum()
    else:
        total_profit = 0

    if "Discount" in df.columns:
        avg_discount = df["Discount"].mean()
    else:
        avg_discount = 0

    total_orders = len(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
    "💰 Total Sales",
    f"${total_sales:,.2f}"
    )

    col2.metric(
    "📈 Total Profit",
    f"${total_profit:,.2f}"
    )

    col3.metric(
    "🏷️ Avg Discount",
    f"{avg_discount:.2f}"
    )

    col4.metric(
    "📦 Total Orders",
    total_orders
    )
    
#BAR CHART FOR SALES BY CATEGORY
    st.markdown("---")

    st.subheader("📊 Sales by Category")

    if "Category" in df.columns and "Sales" in df.columns:

        category_sales = df.groupby(
            "Category"
        )["Sales"].sum()

        fig2, ax2 = plt.subplots()

        ax2.bar(
            category_sales.index,
            category_sales.values
        )

        ax2.set_title("Sales by Category")

        ax2.set_xlabel("Category")

        ax2.set_ylabel("Total Sales")

        st.pyplot(fig2)

    else:
        st.warning(
            "Category or Sales column not found."
        )
        
#PIE CHART FOR SALES DISTRIBUTION BY REGION
    st.markdown("---")

    st.subheader("🥧 Sales Distribution by Region")

    if "Region" in df.columns and "Sales" in df.columns:

        region_sales = df.groupby("Region")["Sales"].sum()

        fig3, ax3 = plt.subplots()

        ax3.pie(
            region_sales.values,
            labels=region_sales.index,
            autopct="%1.1f%%"
        )

        ax3.set_title("Sales Distribution by Region")

        st.pyplot(fig3)

    else:
        st.warning("Region or Sales column not found.")
        
        st.markdown("---")
        
#BAR CHART FOR TOP 10 PRODUCTS BY SALES
    st.subheader("🏆 Top 10 Products by Sales")

    if "Product Name" in df.columns and "Sales" in df.columns:

        top_products = df.groupby(
            "Product Name"
        )["Sales"].sum().sort_values(
            ascending=False
        ).head(10)

        fig4, ax4 = plt.subplots(
            figsize=(10, 6)
        )

        ax4.barh(
            top_products.index,
            top_products.values
        )

        ax4.set_title(
            "Top 10 Products by Sales"
        )

        ax4.set_xlabel("Sales")

        ax4.set_ylabel("Product Name")

        st.pyplot(fig4)

    else:
        st.warning(
            "Product Name or Sales column not found."
        )
        
#REGRESSION MODEL AND GRAPH
    st.markdown("---")

    st.subheader("🤖 Regression Model: Predict Profit")

    required_columns = [
        "Sales",
        "Discount",
        "Quantity",
        "Category",
        "Sub-Category",
        "Region",
        "Segment",
        "Ship Mode",
        "State",
        "Profit"
    ]

    if all(column in df.columns for column in required_columns):

        model_df = df[required_columns].copy()

        model_df = model_df[
            (model_df["Profit"] > -500) &
            (model_df["Profit"] < 5000)
        ]

        features = model_df[
            [
                "Sales",
                "Discount",
                "Quantity",
                "Category",
                "Sub-Category",
                "Region",
                "Segment",
                "Ship Mode",
                "State"
            ]
        ]

        features = features.fillna("Unknown")

        X = pd.get_dummies(
            features,
            drop_first=True
        )

        y = model_df["Profit"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", f"{mae:.2f}")

        with col2:
            st.metric("MSE", f"{mse:.2f}")

        with col3:
            st.metric("R² Score", f"{r2:.2f}")

        fig5, ax5 = plt.subplots(figsize=(8, 5))

        ax5.scatter(y_test, predictions)

        ax5.set_xlabel("Actual Profit")
        ax5.set_ylabel("Predicted Profit")
        ax5.set_title("Actual vs Predicted Profit")

        st.pyplot(fig5)
            
#FEATURE IMPORTANCE
    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    st.dataframe(importance_df)

    fig6, ax6 = plt.subplots(figsize=(12, 8))

    ax6.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax6.set_title("Top 10 Important Features")
    ax6.set_xlabel("Importance Score")
    ax6.set_ylabel("Features")

    ax6.invert_yaxis()

    plt.tight_layout()

    st.pyplot(fig6)