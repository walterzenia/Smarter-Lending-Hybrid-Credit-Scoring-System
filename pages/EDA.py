"""
EDA Page - Exploratory Data Analysis
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from apps.utils import (
    load_data, plot_distribution, plot_correlation_heatmap,
    plot_target_distribution, get_data_summary
)

st.set_page_config(page_title="EDA - Loan Default", page_icon="", layout="wide")

def show():
    st.title("Exploratory Data Analysis")
    st.markdown("Explore and analyze loan application data to uncover patterns and insights.")
    
    st.markdown("---")
    
    # Data Loading Section
    st.subheader(" Data Source")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        data_option = st.radio(
            "Choose data source:",
            ["Upload CSV", "Use Sample Data"],
            horizontal=True
        )
    
    df = None
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f" Loaded {len(df)} rows and {len(df.columns)} columns")
    else:
        # Load sample data
        sample_path = "data/smoke_engineered.csv"
        if Path(sample_path).exists():
            df = load_data(sample_path)
            st.info(f" Using sample dataset: {sample_path}")
        else:
            # Try application_train.csv
            alt_path = "data/application_train.csv"
            if Path(alt_path).exists():
                df = load_data(alt_path)
                st.info(f" Using sample dataset: {alt_path}")
                # Sample for performance
                if len(df) > 50000:
                    df = df.sample(50000, random_state=42)
                    st.warning(" Large dataset - sampled 50,000 rows for performance")
            else:
                st.error("No sample data found. Please upload a CSV file.")
    
    if df is not None:
        st.markdown("---")
        
        # Data Summary
        st.subheader(" Dataset Overview")

        summary = get_data_summary(df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", summary['Rows'])
        col2.metric("Columns", summary['Columns'])
        col3.metric("Missing Values", summary['Missing Values'])
        col4.metric("Duplicates", summary['Duplicate Rows'])
        col5.metric("Memory", summary['Memory Usage'])
        
        # Preview
        st.markdown("####  Data Preview")
        n_rows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)
        
        st.markdown("---")
        
        # Statistical Summary
        st.subheader(" Statistical Summary")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Numeric Features**")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            st.markdown("**Categorical Features**")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                cat_summary = pd.DataFrame({
                    'Column': cat_cols,
                    'Unique Values': [df[col].nunique() for col in cat_cols],
                    'Most Frequent': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols]
                })
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical columns found in dataset")
        
        st.markdown("---")
        
        # Visualizations
        st.subheader(" Data Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution Analysis", "Target Analysis", "Correlation Matrix", "Custom Filters"])
        
        with tab1:
            st.markdown("#### Distribution Plots")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select feature to visualize", numeric_cols, key="dist_col")
                
                if selected_col:
                    fig = plot_distribution(df, selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{df[selected_col].mean():.2f}")
                    col2.metric("Median", f"{df[selected_col].median():.2f}")
                    col3.metric("Std Dev", f"{df[selected_col].std():.2f}")
                    col4.metric("Missing", f"{df[selected_col].isnull().sum()}")
            else:
                st.warning("No numeric columns available for distribution plot")
        
        with tab2:
            st.markdown("#### Target Variable Analysis")
            
            if 'TARGET' in df.columns:
                fig = plot_target_distribution(df, 'TARGET')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Default rate
                default_rate = df['TARGET'].mean() * 100
                st.metric("Default Rate", f"{default_rate:.2f}%")
                
                # Insights
                st.markdown("#####  Insights")
                if default_rate < 10:
                    st.success(" Low default rate - dataset is imbalanced (class 0 dominant)")
                elif default_rate < 30:
                    st.info("ℹ Moderate default rate - relatively balanced dataset")
                else:
                    st.warning(" High default rate - may need special attention")
                
                # Comparison by feature
                st.markdown("##### Default Rate by Feature")
                compare_col = st.selectbox(
                    "Compare default rate across:",
                    [col for col in df.columns if df[col].nunique() < 20 and col != 'TARGET'],
                    key="compare_col"
                )
                
                if compare_col:
                    comparison = df.groupby(compare_col)['TARGET'].agg(['mean', 'count'])
                    comparison.columns = ['Default Rate', 'Count']
                    comparison['Default Rate'] = comparison['Default Rate'] * 100
                    st.dataframe(comparison.sort_values('Default Rate', ascending=False), use_container_width=True)
            else:
                st.info("No TARGET column found in dataset. Upload labeled data to see default analysis.")
        
        with tab3:
            st.markdown("#### Correlation Heatmap")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Limit for performance
                max_cols = st.slider("Maximum features to include", 10, min(50, len(numeric_cols)), 20)
                
                # Select top correlated features with TARGET if available
                if 'TARGET' in numeric_cols:
                    correlations = df[numeric_cols].corr()['TARGET'].abs().sort_values(ascending=False)
                    selected_features = correlations.head(max_cols).index.tolist()
                else:
                    selected_features = numeric_cols[:max_cols]
                
                with st.spinner("Generating correlation heatmap..."):
                    fig = plot_correlation_heatmap(df[selected_features])
                    st.pyplot(fig)
                
                # Show top correlations with TARGET
                if 'TARGET' in numeric_cols:
                    st.markdown("##### Top Correlations with TARGET")
                    top_corr = df[numeric_cols].corr()['TARGET'].abs().sort_values(ascending=False)[1:11]
                    st.dataframe(pd.DataFrame(top_corr).reset_index().rename(
                        columns={'index': 'Feature', 'TARGET': 'Correlation'}
                    ), use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        with tab4:
            st.markdown("#### Custom Data Filters")
            
            # Numeric filters
            st.markdown("##### Filter by Numeric Range")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                filter_col = st.selectbox("Select column to filter", numeric_cols, key="filter_num")
                
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                
                range_vals = st.slider(
                    f"Select range for {filter_col}",
                    min_val, max_val, (min_val, max_val)
                )
                
                filtered_df = df[(df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])]
                
                st.info(f"Filtered: {len(filtered_df)} rows (from {len(df)})")
                st.dataframe(filtered_df.head(20), use_container_width=True)
                
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    " Download Filtered Data",
                    csv,
                    "filtered_data.csv",
                    "text/csv"
                )
        
        st.markdown("---")
        
        # Key Insights
        st.subheader(" Key Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("#### Dataset Characteristics")
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_pct > 20:
                st.warning(f" High missing values ({missing_pct:.1f}%) - consider imputation strategies")
            elif missing_pct > 5:
                st.info(f"ℹ Moderate missing values ({missing_pct:.1f}%) detected")
            else:
                st.success(f" Low missing values ({missing_pct:.1f}%)")
            
            if summary['Duplicate Rows'] > 0:
                st.warning(f" {summary['Duplicate Rows']} duplicate rows detected")
        
        with insights_col2:
            st.markdown("#### Feature Distribution")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                skewed_features = []
                for col in numeric_cols:
                    skew = df[col].skew()
                    if abs(skew) > 1:
                        skewed_features.append(col)
                
                if skewed_features:
                    st.info(f"ℹ {len(skewed_features)} features are highly skewed - may benefit from transformation")
                else:
                    st.success(" Features show reasonable distribution")

if __name__ == "__main__":
    show()
else:
    show()
