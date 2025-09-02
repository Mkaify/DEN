# Load and analyze the attached dataset, produce cleaned data, visualizations, forecasts, 
# profit/loss analysis (with clearly-documented assumptions), and generate a final PDF report.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from datetime import datetime
import os

# For forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_HW = True
except Exception:
    HAS_HW = False

# For PDF report creation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

DATA_PATH = "/mnt/data/Coffee Shop Sales.xlsx"

# ---------- 1) LOAD & CLEAN ----------
# Load sheet (auto-detect the first sheet if name unknown)
xls = pd.ExcelFile(DATA_PATH)
sheet_name = xls.sheet_names[0]
df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)

raw_shape = df.shape

# Standardize column names (lower snake_case)
df.columns = (
    df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
)

# Basic type fixes
# Ensure expected columns exist
expected_cols = [
    'transaction_id','transaction_date','transaction_time','transaction_qty',
    'store_id','store_location','product_id','unit_price',
    'product_category','product_type','product_detail'
]

# If some expected columns are missing, try to infer common variants
col_map = {}
for col in expected_cols:
    if col not in df.columns:
        # try variants
        for c in df.columns:
            if c.replace("__", "_") == col:
                col_map[c] = col
                break
df = df.rename(columns=col_map)

# Coerce date/time
if 'transaction_date' in df.columns:
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
if 'transaction_time' in df.columns:
    # Some datasets have time as string; coerce to timedelta
    try:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce').dt.time
    except Exception:
        pass

# Coerce numeric
for num_col in ['transaction_qty','unit_price','store_id','product_id']:
    if num_col in df.columns:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

# Trim strings
for s_col in ['store_location','product_category','product_type','product_detail']:
    if s_col in df.columns:
        df[s_col] = df[s_col].astype(str).str.strip()

# Remove exact duplicates (full-row)
dup_count = df.duplicated().sum()
df_clean = df.drop_duplicates().copy()

# Handle missing values (document if any remain)
missing_summary = df_clean.isna().sum()

# Create derived columns
# total_sales = qty * unit_price
if 'transaction_qty' in df_clean.columns and 'unit_price' in df_clean.columns:
    df_clean['total_sales'] = df_clean['transaction_qty'] * df_clean['unit_price']

# Remove rows with non-positive or invalid key fields
pre_filter_shape = df_clean.shape
df_clean = df_clean[
    (df_clean['transaction_qty'] > 0) &
    (df_clean['unit_price'] >= 0) &
    (df_clean['transaction_date'].notna())
].copy()
post_filter_shape = df_clean.shape

# ---------- 2) PROFIT / LOSS MODEL (ASSUMPTION-BASED) ----------
# There is no explicit unit cost in the data. We will model profit using:
#   profit = revenue - (COGS + overhead)
# where:
#   revenue = transaction_qty * unit_price
#   COGS% by product_category (industry-informed defaults):
#     - Coffee/Tea/Drinks: 35%
#     - Bakery/Food: 40%
#     - Beans (retail) & Packaged: 55%
#     - Merchandise: 65%
#   overhead_per_transaction: 0.75 currency units (labor, utilities, etc.)
#
# These assumptions are documented in the final report, and a sensitivity note is included.

category_cogs = {
    'coffee': 0.35, 'tea': 0.35, 'drinks': 0.35, 'beverage': 0.35,
    'bakery': 0.40, 'food': 0.40, 'pastry': 0.40, 'sandwich': 0.40,
    'beans': 0.55, 'packaged': 0.55, 'retail': 0.55,
    'merchandise': 0.65, 'merch': 0.65, 'mug': 0.65
}
def guess_cogs_pct(cat: str) -> float:
    if pd.isna(cat):
        return 0.45
    cat_l = str(cat).lower()
    for k, v in category_cogs.items():
        if k in cat_l:
            return v
    # fallback default
    return 0.45

df_clean['cogs_pct'] = df_clean['product_category'].apply(guess_cogs_pct)
overhead_per_txn = 0.75

# Compute per-row profit
df_clean['cogs'] = df_clean['total_sales'] * df_clean['cogs_pct']
df_clean['profit'] = df_clean['total_sales'] - df_clean['cogs'] - overhead_per_txn

# ---------- 3) AGGREGATIONS ----------
by_product = df_clean.groupby('product_detail', as_index=False).agg(
    transactions=('transaction_id', 'count'),
    qty=('transaction_qty', 'sum'),
    avg_price=('unit_price', 'mean'),
    revenue=('total_sales', 'sum'),
    cogs=('cogs', 'sum'),
    profit=('profit', 'sum')
)
by_product['profit_margin'] = np.where(by_product['revenue']>0, by_product['profit']/by_product['revenue'], np.nan)
by_product_sorted_rev = by_product.sort_values('revenue', ascending=False)

by_category = df_clean.groupby('product_category', as_index=False).agg(
    transactions=('transaction_id', 'count'),
    qty=('transaction_qty', 'sum'),
    revenue=('total_sales', 'sum'),
    profit=('profit', 'sum')
)
by_category['profit_margin'] = by_category['profit'] / by_category['revenue']

# Monthly trend
df_clean['date'] = df_clean['transaction_date'].dt.date
daily = df_clean.groupby('date', as_index=False)['total_sales'].sum().rename(columns={'total_sales':'daily_revenue'})
df_clean['year_month'] = df_clean['transaction_date'].dt.to_period('M').astype(str)
monthly = df_clean.groupby('year_month', as_index=False)['total_sales'].sum().rename(columns={'total_sales':'monthly_revenue'})

# ---------- 4) FORECAST (next 30 days using Holt-Winters if available, else simple moving average) ----------
forecast_series = None
try:
    daily_ts = daily.set_index('date')['daily_revenue'].astype(float)
    daily_ts = daily_ts.asfreq('D').fillna(0.0)
    if HAS_HW and len(daily_ts) >= 60:
        # Additive trend + weekly seasonality (period=7) as a reasonable default
        model = ExponentialSmoothing(daily_ts, trend='add', seasonal='add', seasonal_periods=7)
        hw_fit = model.fit(optimized=True)
        forecast_series = hw_fit.forecast(30)
    else:
        # Fallback: rolling mean as naive forecast
        window = min(14, max(3, len(daily_ts)//12))
        rolling = daily_ts.rolling(window=window, min_periods=1).mean()
        last_val = rolling.iloc[-1] if len(rolling) else 0.0
        forecast_series = pd.Series([float(last_val)]*30, index=pd.date_range(daily_ts.index[-1] + pd.Timedelta(days=1), periods=30, freq='D'))
except Exception as e:
    # As a last resort, flat forecast at overall mean
    overall_mean = float(df_clean['total_sales'].mean()) if not df_clean.empty else 0.0
    start_date = pd.to_datetime(df_clean['transaction_date'].max()).date() if 'transaction_date' in df_clean.columns else datetime.today().date()
    forecast_index = pd.date_range(start_date + pd.Timedelta(days=1), periods=30, freq='D')
    forecast_series = pd.Series([overall_mean]*30, index=forecast_index)

# ---------- 5) VISUALIZATIONS ----------
plot_dir = "/mnt/data/plots"
os.makedirs(plot_dir, exist_ok=True)
plots = {}

# Helper to save a plot with a title
def save_current_plot(name):
    out = os.path.join(plot_dir, name)
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    plots[name] = out

# Top 15 products by revenue
top15 = by_product_sorted_rev.head(15)
plt.figure(figsize=(10, 6))
plt.bar(top15['product_detail'], top15['revenue'])
plt.title("Top 15 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.xticks(rotation=75, ha='right')
save_current_plot("top15_products_revenue.png")

# Profit by product (top/bottom 10)
by_product_sorted_profit = by_product.sort_values('profit', ascending=False)
top10_profit = by_product_sorted_profit.head(10)
bottom10_profit = by_product_sorted_profit.tail(10)

plt.figure(figsize=(10, 6))
plt.bar(top10_profit['product_detail'], top10_profit['profit'])
plt.title("Top 10 Products by Profit (Assumption-Based)")
plt.xlabel("Product")
plt.ylabel("Profit")
plt.xticks(rotation=75, ha='right')
save_current_plot("top10_products_profit.png")

plt.figure(figsize=(10, 6))
plt.bar(bottom10_profit['product_detail'], bottom10_profit['profit'])
plt.title("Bottom 10 Products by Profit (Assumption-Based)")
plt.xlabel("Product")
plt.ylabel("Profit")
plt.xticks(rotation=75, ha='right')
save_current_plot("bottom10_products_profit.png")

# Category revenue & profit
plt.figure(figsize=(8, 5))
plt.bar(by_category['product_category'], by_category['revenue'])
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha='right')
save_current_plot("category_revenue.png")

plt.figure(figsize=(8, 5))
plt.bar(by_category['product_category'], by_category['profit'])
plt.title("Profit by Category (Assumption-Based)")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45, ha='right')
save_current_plot("category_profit.png")

# Monthly revenue trend
plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(monthly['year_month']), monthly['monthly_revenue'])
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
save_current_plot("monthly_revenue_trend.png")

# Daily revenue + forecast
if forecast_series is not None and len(daily) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(daily['date']), daily['daily_revenue'], label="Observed")
    # Plot forecast
    plt.plot(forecast_series.index, forecast_series.values, label="Forecast")
    plt.title("Daily Revenue and 30-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    save_current_plot("daily_forecast.png")

# ---------- 6) SAVE CLEANED DATA ----------
cleaned_path = "/mnt/data/Coffee_Shop_Sales_Cleaned.xlsx"
with pd.ExcelWriter(cleaned_path, engine='xlsxwriter') as writer:
    df_clean.to_excel(writer, index=False, sheet_name="cleaned_data")
    by_product.to_excel(writer, index=False, sheet_name="by_product")
    by_category.to_excel(writer, index=False, sheet_name="by_category")
    daily.to_excel(writer, index=False, sheet_name="daily")
    monthly.to_excel(writer, index=False, sheet_name="monthly")

# ---------- 7) BUILD RECOMMENDATIONS ----------
# Identify candidates to push (high margin & high revenue)
revenue_quantile = by_product['revenue'].quantile(0.75)
margin_quantile = by_product['profit_margin'].quantile(0.75)
push_products = by_product[(by_product['revenue'] >= revenue_quantile) & (by_product['profit_margin'] >= margin_quantile)].copy()

# Identify loss-making products (profit < 0 under assumptions)
loss_products = by_product[by_product['profit'] < 0].sort_values('profit').copy()

# Basic strategies
strategies_profit = [
    "Prioritize marketing for high-margin, high-revenue products (push list).",
    "Review pricing on medium-margin items; test small increases (1–3%) where elasticity allows.",
    "Bundle complementary items (e.g., pastry + latte) to lift average ticket.",
    "Protect stock availability for top-margin SKUs; enable low-stock alerts.",
    "Negotiate supplier terms for top-cost drivers to lower COGS% by 2–5 points.",
]
strategies_loss = [
    "For loss-makers: either raise price modestly (1–5%), reduce portion/cost, or lower overhead by streamlining prep.",
    "Use targeted promotions on low-demand SKUs to validate product–market fit; discontinue persistent underperformers.",
    "Redesign bundles to pair loss-makers with high-margin items.",
    "Collect customer feedback to identify taste/quality issues for bakery items with weak repeats.",
]

# ---------- 8) GENERATE PDF REPORT ----------
report_path = "/mnt/data/Coffee_Shop_Sales_Report.pdf"

doc = SimpleDocTemplate(report_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
styles = getSampleStyleSheet()
story = []

def add_heading(text, style='Heading1'):
    story.append(Paragraph(text, styles[style]))
    story.append(Spacer(1, 8))

def add_para(text):
    story.append(Paragraph(text, styles['BodyText']))
    story.append(Spacer(1, 6))

# Cover
add_heading("Coffee Shop Sales Analysis – Predictive Profitability & Strategy", 'Title')
add_para(f"Prepared for: Digital Empowerment Network (Week 04)")
add_para(f"Analyst: Senior Data Scientist (OpenAI)")
add_para(f"Date: {datetime.now().strftime('%B %d, %Y')}")
story.append(Spacer(1, 12))

# Project Overview
add_heading("Project Overview")
add_para("Objective: Clean and analyze the dataset to determine product-level profitability, forecast short-term revenue, and recommend actions to increase profit margins or mitigate losses.")

# Methodology
add_heading("Methodology")
add_para("• Data Cleaning: standardized schema, removed duplicates, coerced types, and constructed derived fields (total_sales, profit).")
add_para("• Profit Model: profit = revenue − (COGS + overhead). COGS% is category-informed; overhead is a fixed per-transaction allocation. All assumptions are clearly stated below and can be tuned for sensitivity.")
add_para("• Analysis: product/category aggregations, trend analysis, profitability ranking, and identification of loss-makers.")
add_para("• Forecasting: 30-day daily revenue forecast using Holt-Winters (or a robust fallback), to anticipate near-term demand.")
add_para("• Recommendations: prioritized actions for profit expansion and loss mitigation.")

# Data Quality
add_heading("Data Quality & Cleaning Summary")
add_para(f"Raw dataset shape: {raw_shape}. Duplicates removed: {dup_count}. Shape after basic filters (valid qty/price/date): {post_filter_shape}.")
ms_table_data = [["Column", "Missing"]]
for col, val in missing_summary.items():
    ms_table_data.append([col, int(val)])
ms_table = Table(ms_table_data, hAlign='LEFT')
ms_table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
story.append(ms_table)
story.append(Spacer(1, 12))

# Assumptions
add_heading("Profit Model Assumptions")
ass_text = (
    "• COGS% by category: Drinks=35%, Bakery/Food=40%, Beans/Packaged=55%, Merchandise=65% (fallback 45%). "
    "• Overhead allocation: 0.75 per transaction. "
    "These values approximate typical coffee shop economics; they can be tuned to your store's reality. "
    "This model may mark some items as loss-making if price/volume cannot cover COGS and overhead."
)
add_para(ass_text)

# Key Findings
add_heading("Key Findings")
# Top winners table (up to 10)
top_winners = by_product.sort_values('profit', ascending=False).head(10)[['product_detail','transactions','revenue','profit','profit_margin']]
top_winners['profit_margin'] = (top_winners['profit_margin']*100).round(1)
tw_data = [["Product","Txns","Revenue","Profit","Margin %"]] + top_winners.round(2).values.tolist()
tw_table = Table(tw_data, hAlign='LEFT')
tw_table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
story.append(Paragraph("Top 10 Products by Profit:", styles['BodyText']))
story.append(tw_table)
story.append(Spacer(1, 10))

# Loss-makers table (up to 10)
loss_tbl = loss_products.head(10)[['product_detail','transactions','revenue','profit','profit_margin']]
loss_tbl['profit_margin'] = (loss_tbl['profit_margin']*100).round(1)
lm_data = [["Product","Txns","Revenue","Profit","Margin %"]] + loss_tbl.round(2).values.tolist()
lm_table = Table(lm_data, hAlign='LEFT')
lm_table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
story.append(Paragraph("Top 10 Loss-Making Products (under assumptions):", styles['BodyText']))
story.append(lm_table)
story.append(Spacer(1, 12))

# Insert plots (as available)
for title, fname in [
    ("Top 15 Products by Revenue", plots.get("top15_products_revenue.png")),
    ("Top 10 Products by Profit", plots.get("top10_products_profit.png")),
    ("Bottom 10 Products by Profit", plots.get("bottom10_products_profit.png")),
    ("Revenue by Category", plots.get("category_revenue.png")),
    ("Profit by Category", plots.get("category_profit.png")),
    ("Monthly Revenue Trend", plots.get("monthly_revenue_trend.png")),
    ("Daily Revenue & 30-Day Forecast", plots.get("daily_forecast.png")),
]:
    if fname and os.path.exists(fname):
        story.append(Paragraph(title, styles['Heading3']))
        story.append(Image(fname, width=500, height=300))
        story.append(Spacer(1, 12))

# Recommendations
add_heading("Recommendations – Profit Increase")
for s in strategies_profit:
    add_para("• " + s)

add_heading("Recommendations – Loss Mitigation")
for s in strategies_loss:
    add_para("• " + s)

# Challenges
add_heading("Challenges & Notes")
add_para("The dataset did not include explicit unit cost; therefore, profits were modeled using industry COGS and a per-transaction overhead. This yields practical, explainable rankings but should be recalibrated once exact costs are available.")
add_para("Seasonality and promotions are approximated via short-range daily forecasting. For long-term planning, incorporate holiday flags, weather, and campaign calendars.")

# Conclusion
add_heading("Conclusion")
add_para("Using a transparent cost model and forecast, we identified high-leverage products to promote, items to fix or retire, and prioritized actions that can improve margins. Implementing the recommendations and refining cost parameters will sharpen profit accuracy and impact.")

# Build PDF
doc.build(story)

# Display some key outputs as tables for the user
from caas_jupyter_tools import display_dataframe_to_user

display_dataframe_to_user("By Product – Profitability (Assumption-Based)", by_product.sort_values('profit', ascending=False).round(2))
display_dataframe_to_user("By Category – Summary", by_category.sort_values('revenue', ascending=False).round(2))
display_dataframe_to_user("Daily Revenue", daily.tail(30))
display_dataframe_to_user("Monthly Revenue", monthly)

# Return file paths to user
{
    "cleaned_dataset": cleaned_path,
    "report_pdf": report_path,
    "plots_dir": plot_dir,
    "sheet_used": sheet_name,
    "raw_shape": raw_shape,
    "post_filter_shape": post_filter_shape,
    "duplicates_removed": int(dup_count)
}
