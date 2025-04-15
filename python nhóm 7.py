import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set the styling for matplotlib
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')
sns.set_style("whitegrid")

# Load the dataset
df = pd.read_csv('Sample  Superstore 1.csv', encoding='cp1252')

# Data preprocessing
def preprocess_data(df):
    # Convert date columns to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Create derived date fields
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Quarter'] = df['Order Date'].dt.quarter
    
    # Calculate fulfillment time (days between order and shipment)
    df['Fulfillment Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Calculate profit ratio
    df['Profit Ratio'] = df['Profit'] / df['Sales']
    
    # Calculate unit cost (for contribution margin analysis)
    df['Unit Cost'] = df['unit price'] - (df['Profit'] / df['Quantity'])
    
    # For RFM analysis, we need the most recent date in the dataset
    df['Recency'] = (df['Order Date'].max() - df['Order Date']).dt.days
    
    return df

df = preprocess_data(df)
print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

#####################################
### 1. RFM CUSTOMER SEGMENTATION ###
#####################################

def rfm_analysis(df):
    print("\n=== CUSTOMER RFM ANALYSIS ===")
    # Group by customer
    rfm = df.groupby('Customer ID').agg({
        'Recency': 'min',
        'Order ID': 'nunique',
        'Sales': 'sum'
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # Create RFM quartiles
    rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['Frequency'], 4, labels=[1, 2, 3, 4])
    rfm['M'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])
    
    # Calculate RFM Score
    rfm['RFM Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    # Create RFM Segments
    rfm['Customer Segment'] = 'Low Value'
    rfm.loc[rfm['RFM Score'].isin(['444', '443', '434', '344']), 'Customer Segment'] = 'Champions'
    rfm.loc[rfm['RFM Score'].isin(['433', '434', '443', '343', '334']), 'Customer Segment'] = 'Loyal Customers'
    rfm.loc[rfm['RFM Score'].isin(['332', '333', '342', '432', '423']), 'Customer Segment'] = 'Potential Loyalists'
    rfm.loc[rfm['RFM Score'].isin(['311', '411', '331', '421']), 'Customer Segment'] = 'New Customers'
    rfm.loc[rfm['RFM Score'].isin(['212', '213', '221', '222', '223']), 'Customer Segment'] = 'At Risk Customers'
    rfm.loc[rfm['RFM Score'].isin(['111', '112', '121', '122', '123', '132', '211', '311']), 'Customer Segment'] = 'Churned Customers'
    
    # Visualize customer segments
    plt.figure(figsize=(12, 6))
    segment_counts = rfm['Customer Segment'].value_counts().sort_values(ascending=False)
    segment_values = rfm.groupby('Customer Segment')['Monetary'].sum().reindex(segment_counts.index)
    
    # Create subplots with 2 axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Number of customers by segment
    sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax1, palette='viridis')
    ax1.set_title('Number of Customers by RFM Segment', fontsize=16)
    ax1.set_xlabel('Segment', fontsize=14)
    ax1.set_ylabel('Number of Customers', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Total monetary value by segment
    sns.barplot(x=segment_values.index, y=segment_values.values, ax=ax2, palette='magma')
    ax2.set_title('Total Sales Value by RFM Segment', fontsize=16)
    ax2.set_xlabel('Segment', fontsize=14)
    ax2.set_ylabel('Total Sales ($)', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.savefig('rfm_customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Merge RFM segments back to the original dataframe
    customer_segments = rfm[['Customer ID', 'Customer Segment']]
    
    return rfm, customer_segments

rfm_results, customer_segments = rfm_analysis(df)
print(f"RFM Analysis completed. Identified {len(rfm_results['Customer Segment'].unique())} customer segments.")

#####################################
### 2. PRODUCT CATEGORY ANALYSIS ###
#####################################

def product_category_analysis(df):
    print("\n=== PRODUCT CATEGORY ANALYSIS ===")
    
    # Category level analysis
    category_perf = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique,
        'Product ID': pd.Series.nunique
    }).reset_index()
    
    category_perf['Profit Margin %'] = (category_perf['Profit'] / category_perf['Sales'] * 100).round(2)
    category_perf['Avg Order Value'] = (category_perf['Sales'] / category_perf['Order ID']).round(2)
    category_perf.sort_values('Sales', ascending=False, inplace=True)
    
    # Subcategory level analysis
    subcategory_perf = df.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    subcategory_perf['Profit Margin %'] = (subcategory_perf['Profit'] / subcategory_perf['Sales'] * 100).round(2)
    subcategory_perf.sort_values(['Category', 'Sales'], ascending=[True, False], inplace=True)
    
    # Visualization 1: Category Performance Overview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Sales and Profit by Category
    sns.barplot(x='Category', y='Sales', data=category_perf, ax=ax1, palette='Blues_d')
    ax1_twin = ax1.twinx()
    sns.barplot(x='Category', y='Profit', data=category_perf, ax=ax1_twin, alpha=0.5, palette='Reds_d')
    
    ax1.set_title('Sales and Profit by Category', fontsize=16)
    ax1.set_ylabel('Sales ($)', fontsize=14)
    ax1_twin.set_ylabel('Profit ($)', fontsize=14, color='red')
    ax1.tick_params(axis='x', rotation=0)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Profit Margin by Category
    sns.barplot(x='Category', y='Profit Margin %', data=category_perf, ax=ax2, palette='RdYlGn')
    ax2.set_title('Profit Margin % by Category', fontsize=16)
    ax2.set_ylabel('Profit Margin %', fontsize=14)
    ax2.tick_params(axis='x', rotation=0)
    
    # Add a horizontal line at y=0 for reference
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('product_category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 2: Subcategory Performance
    plt.figure(figsize=(16, 10))
    
    # Create a grouped bar chart for subcategories
    subcategory_plot = subcategory_perf.pivot(index='Sub-Category', columns='Category', values='Sales')
    subcategory_plot.plot(kind='barh', stacked=False, figsize=(16, 10))
    plt.title('Sales by Sub-Category and Category', fontsize=16)
    plt.xlabel('Sales ($)', fontsize=14)
    plt.ylabel('Sub-Category', fontsize=14)
    plt.legend(title='Category')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('subcategory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_perf, subcategory_perf

category_perf, subcategory_perf = product_category_analysis(df)
print(f"Product Category Analysis completed. Analyzed {len(category_perf)} categories and {len(subcategory_perf)} subcategories.")

#################################
### 3. DISCOUNT IMPACT ANALYSIS ###
#################################

def discount_impact_analysis(df):
    print("\n=== DISCOUNT IMPACT ANALYSIS ===")
    
    # Create discount bins
    df['Discount Bin'] = pd.cut(df['Discount'], 
                               bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], 
                               labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%'])
    
    # For items with no discount, create a separate category
    df.loc[df['Discount'] == 0, 'Discount Bin'] = 'No Discount'
    
    # Analyze the impact of discounts on sales, profit, and quantity
    discount_impact = df.groupby('Discount Bin').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    # Calculate derived metrics
    discount_impact['Profit Margin %'] = (discount_impact['Profit'] / discount_impact['Sales'] * 100).round(2)
    discount_impact['Avg Order Value'] = (discount_impact['Sales'] / discount_impact['Order ID']).round(2)
    discount_impact['Profit per Order'] = (discount_impact['Profit'] / discount_impact['Order ID']).round(2)
    
    # Create a custom order for the discount bins
    discount_order = ['No Discount', '0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%']
    discount_impact['Discount Bin'] = pd.Categorical(discount_impact['Discount Bin'], categories=discount_order, ordered=True)
    discount_impact.sort_values('Discount Bin', inplace=True)
    
    # Visualize discount impact
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Sales by Discount Bin
    sns.barplot(x='Discount Bin', y='Sales', data=discount_impact, ax=axes[0, 0], palette='YlOrRd')
    axes[0, 0].set_title('Total Sales by Discount Level', fontsize=16)
    axes[0, 0].set_xlabel('Discount Level', fontsize=14)
    axes[0, 0].set_ylabel('Total Sales ($)', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Profit by Discount Bin
    sns.barplot(x='Discount Bin', y='Profit', data=discount_impact, ax=axes[0, 1], palette='YlGnBu')
    axes[0, 1].set_title('Total Profit by Discount Level', fontsize=16)
    axes[0, 1].set_xlabel('Discount Level', fontsize=14)
    axes[0, 1].set_ylabel('Total Profit ($)', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Profit Margin % by Discount Bin
    sns.barplot(x='Discount Bin', y='Profit Margin %', data=discount_impact, ax=axes[1, 0], palette='RdYlGn')
    axes[1, 0].set_title('Profit Margin % by Discount Level', fontsize=16)
    axes[1, 0].set_xlabel('Discount Level', fontsize=14)
    axes[1, 0].set_ylabel('Profit Margin %', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Profit per Order by Discount Bin
    sns.barplot(x='Discount Bin', y='Profit per Order', data=discount_impact, ax=axes[1, 1], palette='PuBuGn')
    axes[1, 1].set_title('Profit per Order by Discount Level', fontsize=16)
    axes[1, 1].set_xlabel('Discount Level', fontsize=14)
    axes[1, 1].set_ylabel('Profit per Order ($)', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discount_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Category-specific discount analysis
    category_discount = df.groupby(['Category', 'Discount Bin']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
    }).reset_index()
    
    category_discount['Profit Margin %'] = (category_discount['Profit'] / category_discount['Sales'] * 100).round(2)
    category_discount['Discount Bin'] = pd.Categorical(category_discount['Discount Bin'], categories=discount_order, ordered=True)
    category_discount.sort_values(['Category', 'Discount Bin'], inplace=True)
    
    # Plot category-specific discount impact
    plt.figure(figsize=(16, 10))
    g = sns.FacetGrid(category_discount, col='Category', height=6, aspect=1.2)
    g.map_dataframe(sns.barplot, x='Discount Bin', y='Profit Margin %', palette='RdYlGn')
    g.set_titles('{col_name}')
    g.set_axis_labels('Discount Level', 'Profit Margin %')
    
    # Add horizontal line at y=0
    for ax in g.axes.flat:
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('category_discount_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return discount_impact, category_discount

discount_impact, category_discount = discount_impact_analysis(df)
print(f"Discount Impact Analysis completed. Analyzed {len(discount_impact)} discount levels.")

################################################
### 4. GEOGRAPHIC SALES & PROFIT PERFORMANCE ###
################################################

def geographic_analysis(df):
    print("\n=== GEOGRAPHIC PERFORMANCE ANALYSIS ===")
    
    # State-level performance
    state_perf = df.groupby('State').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique,
        'Customer ID': pd.Series.nunique
    }).reset_index()
    
    state_perf['Profit Margin %'] = (state_perf['Profit'] / state_perf['Sales'] * 100).round(2)
    state_perf['Sales per Customer'] = (state_perf['Sales'] / state_perf['Customer ID']).round(2)
    state_perf.sort_values('Sales', ascending=False, inplace=True)
    
    # Region-level performance
    region_perf = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique,
        'Customer ID': pd.Series.nunique
    }).reset_index()
    
    region_perf['Profit Margin %'] = (region_perf['Profit'] / region_perf['Sales'] * 100).round(2)
    region_perf.sort_values('Sales', ascending=False, inplace=True)
    
    # Top 10 and Bottom 10 States by Profit
    top_states = state_perf.sort_values('Profit', ascending=False).head(10)
    bottom_states = state_perf.sort_values('Profit').head(10)
    
    # Visualization: Regional Performance
    plt.figure(figsize=(16, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))
    
    # Sales and Profit by Region
    sns.barplot(x='Region', y='Sales', data=region_perf, ax=ax1, palette='Blues_d')
    ax1_twin = ax1.twinx()
    sns.barplot(x='Region', y='Profit', data=region_perf, ax=ax1_twin, alpha=0.5, palette='Reds_d')
    
    ax1.set_title('Sales and Profit by Region', fontsize=16)
    ax1.set_ylabel('Sales ($)', fontsize=14)
    ax1_twin.set_ylabel('Profit ($)', fontsize=14, color='red')
    ax1.tick_params(axis='x', rotation=0)
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Top and Bottom States by Profit
    combined_states = pd.concat([top_states, bottom_states])
    colors = ['green'] * 10 + ['red'] * 10
    
    sns.barplot(x='State', y='Profit', data=combined_states, ax=ax2, palette=colors)
    ax2.set_title('Top 10 and Bottom 10 States by Profit', fontsize=16)
    ax2.set_xlabel('State', fontsize=14)
    ax2.set_ylabel('Profit ($)', fontsize=14)
    ax2.tick_params(axis='x', rotation=90)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('geographic_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # State-Category performance heatmap
    state_category = df.groupby(['State', 'Category']).agg({
        'Profit': 'sum'
    }).reset_index()
    
    # Filter to include only states with significant data
    top_20_states = state_perf.nlargest(20, 'Sales')['State'].tolist()
    state_category = state_category[state_category['State'].isin(top_20_states)]
    
    # Create pivot table for heatmap
    state_category_pivot = state_category.pivot(index='State', columns='Category', values='Profit')
    
    plt.figure(figsize=(12, 16))
    sns.heatmap(state_category_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.0f', 
                linewidths=0.5, cbar_kws={'label': 'Profit ($)'})
    plt.title('Profit by State and Category (Top 20 States by Sales)', fontsize=16)
    plt.tight_layout()
    plt.savefig('state_category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return state_perf, region_perf

state_perf, region_perf = geographic_analysis(df)
print(f"Geographic Analysis completed. Analyzed {len(state_perf)} states across {len(region_perf)} regions.")

######################################
### 5. TIME SERIES TREND ANALYSIS ###
######################################

def time_series_analysis(df):
    print("\n=== TIME SERIES ANALYSIS ===")
    
    # Create date-based aggregations
    # Monthly trends
    monthly_trends = df.groupby(['Order Year', 'Order Month']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    # Create a proper date column for plotting
    monthly_trends['Date'] = pd.to_datetime(monthly_trends['Order Year'].astype(str) + '-' + 
                                          monthly_trends['Order Month'].astype(str) + '-01')
    monthly_trends.sort_values('Date', inplace=True)
    
    # Calculate rolling averages (3-month)
    monthly_trends['Sales_3MA'] = monthly_trends['Sales'].rolling(window=3).mean()
    monthly_trends['Profit_3MA'] = monthly_trends['Profit'].rolling(window=3).mean()
    
    # Quarterly trends
    quarterly_trends = df.groupby(['Order Year', 'Order Quarter']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    quarterly_trends['Quarter'] = quarterly_trends['Order Year'].astype(str) + '-Q' + quarterly_trends['Order Quarter'].astype(str)
    quarterly_trends.sort_values(['Order Year', 'Order Quarter'], inplace=True)
    
    # Seasonality analysis (monthly)
    monthly_seasonality = df.groupby('Order Month').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly_seasonality['Month'] = monthly_seasonality['Order Month'].map(month_names)
    monthly_seasonality['Month'] = pd.Categorical(monthly_seasonality['Month'], 
                                                categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                                                ordered=True)
    monthly_seasonality.sort_values('Month', inplace=True)
    
    # Visualize time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # Monthly trend with rolling average
    sns.lineplot(x='Date', y='Sales', data=monthly_trends, ax=ax1, marker='o', linewidth=2, label='Monthly Sales')
    sns.lineplot(x='Date', y='Sales_3MA', data=monthly_trends, ax=ax1, linewidth=3, linestyle='--', label='3-Month Moving Avg')
    
    ax1.set_title('Monthly Sales Trend with 3-Month Moving Average', fontsize=16)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Sales ($)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Monthly seasonality
    sns.barplot(x='Month', y='Sales', data=monthly_seasonality, ax=ax2, palette='viridis')
    
    ax2.set_title('Monthly Sales Seasonality', fontsize=16)
    ax2.set_xlabel('Month', fontsize=14)
    ax2.set_ylabel('Total Sales ($)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Category-specific seasonal analysis
    category_monthly = df.groupby(['Category', 'Order Month']).agg({
        'Sales': 'sum'
    }).reset_index()
    
    category_monthly['Month'] = category_monthly['Order Month'].map(month_names)
    category_monthly['Month'] = pd.Categorical(category_monthly['Month'], 
                                             categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                                             ordered=True)
    category_monthly.sort_values(['Category', 'Month'], inplace=True)
    
    plt.figure(figsize=(16, 10))
    g = sns.FacetGrid(category_monthly, col='Category', height=6, aspect=1.2)
    g.map_dataframe(sns.barplot, x='Month', y='Sales', palette='viridis')
    g.set_titles('{col_name}')
    g.set_axis_labels('Month', 'Sales ($)')
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('category_seasonality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return monthly_trends, quarterly_trends, monthly_seasonality

monthly_trends, quarterly_trends, monthly_seasonality = time_series_analysis(df)
print(f"Time Series Analysis completed. Analyzed trends across {len(monthly_trends)} months.")

#########################################
### 6. CUSTOMER SEGMENT PROFITABILITY ###
#########################################

def customer_segment_analysis(df):
    print("\n=== CUSTOMER SEGMENT ANALYSIS ===")
    
    # Analyze segment performance
    segment_perf = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique,
        'Customer ID': pd.Series.nunique
    }).reset_index()
    
    segment_perf['Profit Margin %'] = (segment_perf['Profit'] / segment_perf['Sales'] * 100).round(2)
    segment_perf['Sales per Customer'] = (segment_perf['Sales'] / segment_perf['Customer ID']).round(2)
    segment_perf['Profit per Customer'] = (segment_perf['Profit'] / segment_perf['Customer ID']).round(2)
    segment_perf['Orders per Customer'] = (segment_perf['Order ID'] / segment_perf['Customer ID']).round(2)
    segment_perf.sort_values('Sales', ascending=False, inplace=True)
    
    # Segment-Category performance
    segment_category = df.groupby(['Segment', 'Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    segment_category['Profit Margin %'] = (segment_category['Profit'] / segment_category['Sales'] * 100).round(2)
    
    # Visualize segment performance
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Sales and Profit by Segment
    sns.barplot(x='Segment', y='Sales', data=segment_perf, ax=axes[0, 0], palette='Blues_d')
    ax1_twin = axes[0, 0].twinx()
    sns.barplot(x='Segment', y='Profit', data=segment_perf, ax=ax1_twin, alpha=0.5, palette='Reds_d')
    
    axes[0, 0].set_title('Sales and Profit by Segment', fontsize=16)
    axes[0, 0].set_xlabel('Segment', fontsize=14)
    axes[0, 0].set_ylabel('Sales ($)', fontsize=14)
    ax1_twin.set_ylabel('Profit ($)', fontsize=14, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Profit Margin % by Segment
    sns.barplot(x='Segment', y='Profit Margin %', data=segment_perf, ax=axes[0, 1], palette='RdYlGn')
    axes[0, 1].set_title('Profit Margin % by Segment', fontsize=16)
    axes[0, 1].set_xlabel('Segment', fontsize=14)
    axes[0, 1].set_ylabel('Profit Margin %', fontsize=14)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Sales per Customer by Segment
    sns.barplot(x='Segment', y='Sales per Customer', data=segment_perf, ax=axes[1, 0], palette='YlOrBr')
    axes[1, 0].set_title('Sales per Customer by Segment', fontsize=16)
    axes[1, 0].set_xlabel('Segment', fontsize=14)
    axes[1, 0].set_ylabel('Sales per Customer ($)', fontsize=14)
    
    # Orders per Customer by Segment
    sns.barplot(x='Segment', y='Orders per Customer', data=segment_perf, ax=axes[1, 1], palette='PuBuGn')
    axes[1, 1].set_title('Orders per Customer by Segment', fontsize=16)
    axes[1, 1].set_xlabel('Segment', fontsize=14)
    axes[1, 1].set_ylabel('Orders per Customer', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('segment_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Segment-Category heatmap
    segment_category_pivot = segment_category.pivot(index='Segment', columns='Category', values='Profit Margin %')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(segment_category_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.1f', 
                linewidths=0.5, cbar_kws={'label': 'Profit Margin %'})
    plt.title('Profit Margin % by Segment and Category', fontsize=16)
    plt.tight_layout()
    plt.savefig('segment_category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return segment_perf, segment_category

segment_perf, segment_category = customer_segment_analysis(df)
print(f"Customer Segment Analysis completed. Analyzed {len(segment_perf)} customer segments.")

###############################
### 7. ABC INVENTORY ANALYSIS ###
###############################

def abc_inventory_analysis(df):
    print("\n=== ABC INVENTORY ANALYSIS ===")
    
    # Group by Product ID for inventory analysis
    product_perf = df.groupby('Product ID').agg({
        'Product Name': 'first',
        'Category': 'first',
        'Sub-Category': 'first',
        'Sales': 'sum',
        'Quantity': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    # Calculate profit per unit and sort
    product_perf['Profit per Unit'] = product_perf['Profit'] / product_perf['Quantity']
    product_perf.sort_values('Sales', ascending=False, inplace=True)
    
    # Calculate cumulative percentage
    product_perf['Sales_Percent'] = product_perf['Sales'] / product_perf['Sales'].sum() * 100
    product_perf['Cumulative_Percent'] = product_perf['Sales_Percent'].cumsum()
    
    # Assign ABC classes
    product_perf['ABC_Class'] = 'C'
    product_perf.loc[product_perf['Cumulative_Percent'] <= 70, 'ABC_Class'] = 'A'
    product_perf.loc[(product_perf['Cumulative_Percent'] > 70) & 
                     (product_perf['Cumulative_Percent'] <= 90), 'ABC_Class'] = 'B'
    
    # Summarize by ABC class
    abc_summary = product_perf.groupby('ABC_Class').agg({
        'Product ID': 'count',
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    abc_summary['Products_Percent'] = (abc_summary['Product ID'] / abc_summary['Product ID'].sum() * 100).round(1)
    abc_summary['Sales_Percent'] = (abc_summary['Sales'] / abc_summary['Sales'].sum() * 100).round(1)
    abc_summary['Profit_Percent'] = (abc_summary['Profit'] / abc_summary['Profit'].sum() * 100).round(1)
    abc_summary['Quantity_Percent'] = (abc_summary['Quantity'] / abc_summary['Quantity'].sum() * 100).round(1)
    
    # Ensure correct order for displaying ABC classes
    abc_summary['ABC_Class'] = pd.Categorical(abc_summary['ABC_Class'], categories=['A', 'B', 'C'], ordered=True)
    abc_summary.sort_values('ABC_Class', inplace=True)
    
    # Visualize ABC Analysis
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Pareto Chart (cumulative sales percentage)
    product_sample = product_perf.head(100)  # Take top 100 products for readability
    
    ax1 = axes[0, 0]
    ax1.bar(range(len(product_sample)), product_sample['Sales_Percent'], color='lightblue')
    ax1.set_title('Pareto Chart: Product Sales Distribution', fontsize=16)
    ax1.set_xlabel('Product Rank', fontsize=14)
    ax1.set_ylabel('Sales Percentage (%)', fontsize=14)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(product_sample)), product_sample['Cumulative_Percent'], 'r-', linewidth=2)
    ax1_twin.set_ylabel('Cumulative Percentage (%)', fontsize=14, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Add reference lines for ABC classes
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1_twin.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='A Class (70%)')
    ax1_twin.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='B Class (90%)')
    ax1_twin.legend(fontsize=12)
    
    # ABC Class Distribution (# of products)
    ax2 = axes[0, 1]
    sns.barplot(x='ABC_Class', y='Product ID', data=abc_summary, ax=ax2, palette='viridis')
    ax2.set_title('Number of Products by ABC Class', fontsize=16)
    ax2.set_xlabel('ABC Class', fontsize=14)
    ax2.set_ylabel('Number of Products', fontsize=14)
    
    # Add percentage labels
    for i, row in enumerate(abc_summary.itertuples()):
        ax2.text(i, row.Product_ID/2, f"{row.Products_Percent}%", 
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # Sales by ABC Class
    ax3 = axes[1, 0]
    sns.barplot(x='ABC_Class', y='Sales', data=abc_summary, ax=ax3, palette='YlOrRd')
    ax3.set_title('Sales by ABC Class', fontsize=16)
    ax3.set_xlabel('ABC Class', fontsize=14)
    ax3.set_ylabel('Sales ($)', fontsize=14)
    
    # Add percentage labels
    for i, row in enumerate(abc_summary.itertuples()):
        ax3.text(i, row.Sales/2, f"{row.Sales_Percent}%", 
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # Profit by ABC Class
    ax4 = axes[1, 1]
    sns.barplot(x='ABC_Class', y='Profit', data=abc_summary, ax=ax4, palette='RdYlGn')
    ax4.set_title('Profit by ABC Class', fontsize=16)
    ax4.set_xlabel('ABC Class', fontsize=14)
    ax4.set_ylabel('Profit ($)', fontsize=14)
    
    # Add percentage labels
    for i, row in enumerate(abc_summary.itertuples()):
        y_pos = max(row.Profit/2, 50)  # Ensure text visibility even with low profit
        ax4.text(i, y_pos, f"{row.Profit_Percent}%", 
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('abc_inventory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Category-wise ABC Distribution
    category_abc = product_perf.groupby(['Category', 'ABC_Class']).agg({
        'Product ID': 'count',
        'Sales': 'sum'
    }).reset_index()
    
    category_abc_pivot = category_abc.pivot(index='Category', columns='ABC_Class', values='Product ID')
    category_abc_pivot.fillna(0, inplace=True)
    
    plt.figure(figsize=(12, 8))
    category_abc_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), 
                           colormap='viridis')
    plt.title('Distribution of ABC Products by Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Number of Products', fontsize=14)
    plt.legend(title='ABC Class')
    
    plt.tight_layout()
    plt.savefig('category_abc_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return product_perf, abc_summary

product_perf, abc_summary = abc_inventory_analysis(df)
print(f"ABC Inventory Analysis completed. Classified {len(product_perf)} products into ABC categories.")

####################################
### 8. PRODUCT AFFINITY ANALYSIS ###
####################################

def product_affinity_analysis(df):
    print("\n=== PRODUCT AFFINITY ANALYSIS ===")
    
    # Create a basket dataset (Order ID - Sub-Category combinations)
    baskets = df.groupby(['Order ID', 'Sub-Category'])['Sales'].sum().reset_index()
    
    # Create a pivot table to convert to transaction format
    transactions = baskets.pivot_table(index='Order ID', 
                                      columns='Sub-Category', 
                                      values='Sales', 
                                      aggfunc='sum', 
                                      fill_value=0)
    
    # Convert to binary (0,1) to indicate purchase/no purchase
    transactions_binary = transactions.applymap(lambda x: 1 if x > 0 else 0)
    
    # Calculate support for each sub-category
    subcategory_support = transactions_binary.sum() / len(transactions_binary)
    subcategory_support = subcategory_support.sort_values(ascending=False)
    
    # Find co-occurrence of sub-categories
    cooccurrence_matrix = transactions_binary.T.dot(transactions_binary)
    np.fill_diagonal(cooccurrence_matrix.values, 0)  # Remove self-pairs
    
    # Calculate lift (measure of association)
    support_diag = np.diag(subcategory_support.values)
    lift_matrix = cooccurrence_matrix / len(transactions_binary) / np.dot(subcategory_support.values.reshape(-1, 1), 
                                                                      subcategory_support.values.reshape(1, -1))
    np.fill_diagonal(lift_matrix.values, 0)
    
    # Get top subcategory pairs based on lift
    pairs = []
    for i in range(len(lift_matrix.columns)):
        for j in range(i+1, len(lift_matrix.columns)):
            pairs.append({
                'Sub-Category 1': lift_matrix.index[i],
                'Sub-Category 2': lift_matrix.columns[j],
                'Lift': lift_matrix.iloc[i, j],
                'Co-occurrence': cooccurrence_matrix.iloc[i, j]
            })
    
    pairs_df = pd.DataFrame(pairs)
    top_pairs = pairs_df.sort_values('Lift', ascending=False).head(20)
    
    # Visualize product affinity
    plt.figure(figsize=(14, 10))
    
    # Product Subcategory Support (Popularity)
    top_support = subcategory_support.head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_support.index, y=top_support.values, palette='Blues_d')
    plt.title('Top 10 Product Sub-Categories by Support (Popularity)', fontsize=16)
    plt.xlabel('Sub-Category', fontsize=14)
    plt.ylabel('Support (Percentage of Orders)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('subcategory_support.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top pairs by lift
    plt.figure(figsize=(16, 10))
    heatmap_size = min(15, len(lift_matrix))  # Limit size of heatmap for readability
    top_subcategories = subcategory_support.head(heatmap_size).index
    
    lift_heatmap = lift_matrix.loc[top_subcategories, top_subcategories]
    
    mask = np.triu(np.ones_like(lift_heatmap, dtype=bool))
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(lift_heatmap, annot=True, fmt='.2f', cmap='YlGnBu', 
                mask=mask, square=True, linewidths=.5, cbar_kws={'label': 'Lift Ratio'})
    
    plt.title('Product Affinity Analysis (Lift Ratio) - Top Sub-Categories', fontsize=16)
    plt.tight_layout()
    plt.savefig('product_affinity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar chart for top product pairs
    plt.figure(figsize=(16, 12))
    
    pair_labels = [f"{row['Sub-Category 1']} + {row['Sub-Category 2']}" for _, row in top_pairs.head(10).iterrows()]
    
    sns.barplot(x='Lift', y=pair_labels, data=top_pairs.head(10), palette='viridis')
    plt.title('Top 10 Product Sub-Category Pairs by Lift Ratio', fontsize=16)
    plt.xlabel('Lift Ratio', fontsize=14)
    plt.ylabel('Product Pair', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('top_product_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return top_pairs, subcategory_support

top_pairs, subcategory_support = product_affinity_analysis(df)
print(f"Product Affinity Analysis completed. Identified {len(top_pairs)} significant product pairs.")

#################################################
### 9. PRICE SENSITIVITY & ELASTICITY ANALYSIS ###
#################################################

def price_sensitivity_analysis(df):
    print("\n=== PRICE SENSITIVITY ANALYSIS ===")
    
    # Filter dataset to focus on high-volume sub-categories
    subcategory_counts = df['Sub-Category'].value_counts()
    high_volume_subcategories = subcategory_counts[subcategory_counts > 100].index.tolist()
    
    price_sensitivity_df = df[df['Sub-Category'].isin(high_volume_subcategories)].copy()
    
    # Group data by unit price bins
    price_sensitivity_df['Price_Bin'] = pd.cut(price_sensitivity_df['unit price'], 
                                             bins=[0, 50, 100, 200, 500, 1000, 5000], 
                                             labels=['0-50', '51-100', '101-200', '201-500', '501-1000', '1001+'])
    
    # Analyze discount effect at different price points
    price_sensitivity = price_sensitivity_df.groupby(['Price_Bin', 'Discount_Bin']).agg({
        'Sales': 'sum',
        'Quantity': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    price_sensitivity['Units per Order'] = (price_sensitivity['Quantity'] / price_sensitivity['Order ID']).round(2)
    
    # For elasticity analysis, look at 0 discount versus high discount
    elasticity_data = price_sensitivity_df.groupby(['Sub-Category', 'Discount_Bin']).agg({
        'unit price': 'mean',
        'Quantity': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    elasticity_data['Units per Order'] = (elasticity_data['Quantity'] / elasticity_data['Order ID']).round(2)
    
    # Filter to focus on specific discount levels for comparison
    high_discount = elasticity_data[elasticity_data['Discount_Bin'].isin(['31-40%', '41-50%', '51-100%'])]
    no_discount = elasticity_data[elasticity_data['Discount_Bin'] == 'No Discount']
    
    # Visualize price sensitivity
    # Plot 1: Units per order at different price points and discount levels
    plt.figure(figsize=(16, 10))
    
    g = sns.catplot(x='Price_Bin', y='Units per Order', hue='Discount_Bin', 
                   data=price_sensitivity, kind='bar', height=8, aspect=1.5, palette='viridis')
    
    g.set_xlabels('Price Range ($)', fontsize=14)
    g.set_ylabels('Units per Order', fontsize=14)
    plt.title('Price Sensitivity: Units per Order by Price Range and Discount Level', fontsize=16)
    plt.legend(title='Discount Level')
    
    plt.tight_layout()
    plt.savefig('price_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Sub-category specific price sensitivity
    top_subcategories = subcategory_counts.nlargest(6).index.tolist()
    subcategory_prices = price_sensitivity_df[price_sensitivity_df['Sub-Category'].isin(top_subcategories)]
    
    plt.figure(figsize=(20, 16))
    
    g = sns.FacetGrid(subcategory_prices, col='Sub-Category', col_wrap=3, height=6)
    g.map_dataframe(sns.scatterplot, x='unit price', y='Discount', 
                   size='Quantity', sizes=(20, 200), alpha=0.6)
    
    g.set_titles('{col_name}')
    g.set_axis_labels('Unit Price ($)', 'Discount %')
    
    plt.tight_layout()
    plt.savefig('subcategory_price_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return price_sensitivity

price_sensitivity = price_sensitivity_analysis(df)
print("Price Sensitivity Analysis completed.")

#################################################
### 10. COMPREHENSIVE DASHBOARD (MAIN METRICS) ###
#################################################

def create_dashboard(df, category_perf, segment_perf, monthly_trends):
    print("\n=== CREATING COMPREHENSIVE METRICS DASHBOARD ===")
    
    # Calculate key performance indicators
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    overall_margin = (total_profit / total_sales * 100).round(2)
    total_customers = df['Customer ID'].nunique()
    total_orders = df['Order ID'].nunique()
    avg_order_value = (total_sales / total_orders).round(2)
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(22, 28))
    gs = fig.add_gridspec(6, 2)
    
    # 1. KPI Summary Section
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.text(0.5, 0.8, 'E-COMMERCE PERFORMANCE DASHBOARD', fontsize=24, 
             weight='bold', ha='center', va='center')
    
    # Add KPI boxes
    kpi_metrics = [
        {'label': 'Total Sales', 'value': f'${total_sales:,.2f}'},
        {'label': 'Total Profit', 'value': f'${total_profit:,.2f}'},
        {'label': 'Profit Margin', 'value': f'{overall_margin}%'},
        {'label': 'Customers', 'value': f'{total_customers:,}'},
        {'label': 'Orders', 'value': f'{total_orders:,}'},
        {'label': 'Avg Order Value', 'value': f'${avg_order_value:,.2f}'}
    ]
    
    for i, kpi in enumerate(kpi_metrics):
        x_pos = 0.1 + 0.15 * i
        rect = plt.Rectangle((x_pos-0.06, 0.3), 0.12, 0.3, facecolor='lightgray', alpha=0.3)
        ax1.add_patch(rect)
        ax1.text(x_pos, 0.5, kpi['value'], fontsize=16, weight='bold', ha='center')
        ax1.text(x_pos, 0.4, kpi['label'], fontsize=12, ha='center')
    
    # 2. Sales Trend
    ax2 = fig.add_subplot(gs[1, :])
    sns.lineplot(x='Date', y='Sales', data=monthly_trends, ax=ax2, marker='o', linewidth=2)
    ax2.set_title('Monthly Sales Trend', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Sales ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Category Performance
    ax3 = fig.add_subplot(gs[2, 0])
    sns.barplot(x='Category', y='Sales', data=category_perf, ax=ax3, palette='Blues_d')
    ax3.set_title('Sales by Category', fontsize=16)
    ax3.set_xlabel('Category', fontsize=12)
    ax3.set_ylabel('Sales ($)', fontsize=12)
    ax3.tick_params(axis='x', rotation=0)
    
    # 4. Segment Performance
    ax4 = fig.add_subplot(gs[2, 1])
    sns.barplot(x='Segment', y='Sales', data=segment_perf, ax=ax4, palette='Greens_d')
    ax4.set_title('Sales by Customer Segment', fontsize=16)
    ax4.set_xlabel('Segment', fontsize=12)
    ax4.set_ylabel('Sales ($)', fontsize=12)
    
    # 5. Profit Margins by Category and Segment
    ax5 = fig.add_subplot(gs[3, 0])
    sns.barplot(x='Category', y='Profit Margin %', data=category_perf, ax=ax5, palette='RdYlGn')
    ax5.set_title('Profit Margin % by Category', fontsize=16)
    ax5.set_xlabel('Category', fontsize=12)
    ax5.set_ylabel('Profit Margin %', fontsize=12)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, 1])
    sns.barplot(x='Segment', y='Profit Margin %', data=segment_perf, ax=ax6, palette='RdYlGn')
    ax6.set_title('Profit Margin % by Customer Segment', fontsize=16)
    ax6.set_xlabel('Segment', fontsize=12)
    ax6.set_ylabel('Profit Margin %', fontsize=12)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Geographical Analysis
    region_data = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    ax7 = fig.add_subplot(gs[4, 0])
    sns.barplot(x='Region', y='Sales', data=region_data, ax=ax7, palette='Purples_d')
    ax7.set_title('Sales by Region', fontsize=16)
    ax7.set_xlabel('Region', fontsize=12)
    ax7.set_ylabel('Sales ($)', fontsize=12)
    
    # 7. Discount Impact
    discount_data = df.groupby('Discount_Bin').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    discount_order = ['No Discount', '0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%']
    discount_data['Discount_Bin'] = pd.Categorical(discount_data['Discount_Bin'], 
                                                 categories=discount_order, 
                                                 ordered=True)
    discount_data.sort_values('Discount_Bin', inplace=True)
    discount_data['Profit Margin %'] = (discount_data['Profit'] / discount_data['Sales'] * 100).round(2)
    
    ax8 = fig.add_subplot(gs[4, 1])
    sns.barplot(x='Discount_Bin', y='Profit Margin %', data=discount_data, ax=ax8, palette='RdYlGn')
    ax8.set_title('Profit Margin % by Discount Level', fontsize=16)
    ax8.set_xlabel('Discount Level', fontsize=12)
    ax8.set_ylabel('Profit Margin %', fontsize=12)
    ax8.tick_params(axis='x', rotation=45)
    ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 8. RFM Segment Analysis (if available)
    try:
        customer_rfm = df.merge(rfm_results[['Customer ID', 'Customer Segment']], on='Customer ID', how='left')
        rfm_segment_perf = customer_rfm.groupby('Customer Segment').agg({
            'Sales': 'sum',
            'Customer ID': pd.Series.nunique
        }).reset_index()
        
        rfm_segment_perf['Sales per Customer'] = (rfm_segment_perf['Sales'] / rfm_segment_perf['Customer ID']).round(2)
        
        ax9 = fig.add_subplot(gs[5, 0])
        sns.barplot(x='Customer Segment', y='Sales', data=rfm_segment_perf, ax=ax9, palette='viridis')
        ax9.set_title('Sales by RFM Customer Segment', fontsize=16)
        ax9.set_xlabel('Customer Segment', fontsize=12)
        ax9.set_ylabel('Sales ($)', fontsize=12)
        ax9.tick_params(axis='x', rotation=45)
        
        ax10 = fig.add_subplot(gs[5, 1])
        sns.barplot(x='Customer Segment', y='Sales per Customer', data=rfm_segment_perf, ax=ax10, palette='plasma')
        ax10.set_title('Sales per Customer by RFM Segment', fontsize=16)
        ax10.set_xlabel('Customer Segment', fontsize=12)
        ax10.set_ylabel('Sales per Customer ($)', fontsize=12)
        ax10.tick_params(axis='x', rotation=45)
    except:
        # If RFM analysis not available, show something else
        ship_mode_data = df.groupby('Ship Mode').agg({
            'Sales': 'sum',
            'Fulfillment Days': 'mean'
        }).reset_index()
        
        ax9 = fig.add_subplot(gs[5, 0])
        sns.barplot(x='Ship Mode', y='Sales', data=ship_mode_data, ax=ax9, palette='viridis')
        ax9.set_title('Sales by Shipping Mode', fontsize=16)
        ax9.set_xlabel('Shipping Mode', fontsize=12)
        ax9.set_ylabel('Sales ($)', fontsize=12)
        
        ax10 = fig.add_subplot(gs[5, 1])
        sns.barplot(x='Ship Mode', y='Fulfillment Days', data=ship_mode_data, ax=ax10, palette='plasma')
        ax10.set_title('Average Fulfillment Days by Shipping Mode', fontsize=16)
        ax10.set_xlabel('Shipping Mode', fontsize=12)
        ax10.set_ylabel('Fulfillment Days', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ecommerce_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Dashboard created successfully.")

create_dashboard(df, category_perf, segment_perf, monthly_trends)

print("\nAll analyses completed successfully!")
print("Generated visualizations for all key analytics areas.")
print("You can now review the following output files:")
for img in ['rfm_customer_segmentation.png', 'product_category_analysis.png', 'discount_impact_analysis.png',
            'geographic_performance.png', 'time_series_analysis.png', 'segment_performance.png',
            'abc_inventory_analysis.png', 'subcategory_support.png', 'price_sensitivity.png', 'ecommerce_dashboard.png']:
    print(f"- {img}")