"""
=============================================================================
SmartSpend AI - Student Personal Finance & ML Forecaster (Stage 2)
=============================================================================
A beginner-friendly Streamlit web application designed for college students
to track income, monitor expenses, and use Machine Learning (Linear Regression)
to forecast future expenses and detect budget overrun risks.

Stage 2 Features:
1. Stage 1 Core Dashboard (Metrics, Forms, History, SQLite).
2. Machine Learning Expense Forecasting (Scikit-Learn Linear Regression).
3. Forecast Horizon Slider (1 - 6 months ahead).
4. Model evaluation metrics (R² score, Mean Absolute Error - MAE, trend slope).
5. Category-wise future expense distribution.
6. Budget risk scoring (Low, Moderate, High Risk).
7. 6-Month historical training dataset generator.
=============================================================================
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

# Import our custom Machine Learning & AI Coach modules
import ml_predictor as ml
import ai_coach as ac

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartSpend AI - Student Finance & ML Forecast",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM STYLING (Dark Navy, Cream, and Emerald Green Palette)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Main app background in soft cream */
    .stApp {
        background-color: #FAF9F6;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top banner / header styling */
    .header-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FAF9F6;
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 20px;
        border-left: 6px solid #10B981;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .header-card h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .header-card p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        color: #CBD5E1;
    }

    /* Custom financial & ML metric cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Risk Alert Box */
    .risk-box {
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0 24px 0;
        border-left: 6px solid;
    }

    /* AI Coach Advice Cards */
    .coach-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .coach-badge {
        background-color: #ECFDF5;
        color: #047857;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-block;
    }

    /* Student Persona Card */
    .persona-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FAF9F6;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        border-left: 6px solid #10B981;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }

    /* Spending Leakage Warning Card */
    .leak-card {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    /* Custom button styling */
    div.stButton > button:first-child {
        background-color: #10B981;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 9px 18px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #059669;
        color: #FFFFFF;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATABASE SETUP & SQLITE OPERATIONS
# -----------------------------------------------------------------------------
DB_FILE = "smartspend.db"

def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database by creating the transactions table if it doesn't exist.
    If the table is empty, realistic demo transactions are automatically seeded.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                type TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT
            )
        """)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]

        if count == 0:
            seed_initial_sample_data(conn)

def seed_initial_sample_data(conn):
    """
    Seeds basic 1-month student transactions.
    """
    sample_data = [
        ("2026-10-01", "Income", "Part-time Job", 850.00, "Campus Library Desk Shift"),
        ("2026-10-03", "Income", "Scholarship / Grant", 400.00, "Monthly Merit Stipend"),
        ("2026-10-04", "Expense", "Rent & Housing", 500.00, "Student Dorm / Apartment Share"),
        ("2026-10-07", "Expense", "Food & Groceries", 145.20, "Weekly Grocery Run"),
        ("2026-10-10", "Expense", "Books & Supplies", 85.50, "Python & AI Course Textbooks"),
        ("2026-10-12", "Expense", "Transportation", 45.00, "Monthly University Transit Pass"),
        ("2026-10-16", "Expense", "Entertainment", 28.00, "Weekend Movie with Friends"),
        ("2026-10-20", "Expense", "Food & Groceries", 18.50, "Study Session Coffee & Snacks")
    ]
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO transactions (date, type, category, amount, description)
        VALUES (?, ?, ?, ?, ?)
    """, sample_data)
    conn.commit()

def seed_extended_historical_data():
    """
    Seeds 6 months of historical student data (May - October 2026) for ML model training.
    """
    df_hist = ml.get_extended_historical_data()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS transactions")
        conn.commit()
        init_db()
        
        # Insert generated 6-month historical dataset
        for _, row in df_hist.iterrows():
            cursor.execute("""
                INSERT INTO transactions (date, type, category, amount, description)
                VALUES (?, ?, ?, ?, ?)
            """, (row["date"], row["type"], row["category"], float(row["amount"]), row["description"]))
        conn.commit()

def load_transactions():
    """
    Fetches all transactions from SQLite into a Pandas DataFrame.
    """
    with get_db_connection() as conn:
        query = "SELECT id, date, type, category, amount, description FROM transactions ORDER BY date DESC, id DESC"
        df = pd.read_sql_query(query, conn)
    return df

def add_transaction(date_str, trans_type, category, amount, description):
    """
    Safely inserts a new transaction record into SQLite.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (date, type, category, amount, description)
                VALUES (?, ?, ?, ?, ?)
            """, (date_str, trans_type, category, float(amount), description.strip()))
            conn.commit()
            return True, "Transaction added successfully!"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def delete_transaction(transaction_id):
    """
    Deletes a transaction by its primary key ID.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE id = ?", (int(transaction_id),))
            conn.commit()
            return True, "Transaction deleted successfully!"
    except Exception as e:
        return False, f"Database error: {str(e)}"

def reset_to_initial_demo():
    """
    Clears all records and restores the basic 8 demo transactions.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS transactions")
        conn.commit()
    init_db()

# Initialize DB on app startup
init_db()

# -----------------------------------------------------------------------------
# 4. CATEGORY DEFINITIONS FOR STUDENTS
# -----------------------------------------------------------------------------
EXPENSE_CATEGORIES = [
    "Food & Groceries",
    "Books & Supplies",
    "Rent & Housing",
    "Tuition & Fees",
    "Transportation",
    "Entertainment",
    "Health & Personal",
    "Other Expense"
]

INCOME_CATEGORIES = [
    "Part-time Job",
    "Allowance / Family",
    "Scholarship / Grant",
    "Freelance / Tutoring",
    "Other Income"
]

# -----------------------------------------------------------------------------
# 5. SIDEBAR: DATA ENTRY & ML DEMO TOOLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://api.iconify.design/solar:wallet-money-bold-duotone.svg?color=%2310b981", 
        width=48
    )
    st.title("SmartSpend AI")
    st.caption("Student Finance & ML Forecaster")

    st.subheader("➕ Add Transaction")
    
    with st.form("transaction_form", clear_on_submit=True):
        trans_type = st.radio("Type", options=["Expense", "Income"], horizontal=True)
        categories = EXPENSE_CATEGORIES if trans_type == "Expense" else INCOME_CATEGORIES
        category = st.selectbox("Category", options=categories)

        amount = st.number_input(
            "Amount ($)",
            min_value=0.00,
            step=1.00,
            format="%.2f",
            help="Enter a positive dollar amount."
        )

        trans_date = st.date_input("Date", value=date.today())
        description = st.text_input("Description (Optional)", placeholder="e.g. Textbook, Groceries", max_chars=80)

        submitted = st.form_submit_button("Record Transaction")

        if submitted:
            if amount <= 0:
                st.error("⚠️ Amount must be greater than $0.00.")
            elif not category:
                st.error("⚠️ Please select a category.")
            else:
                date_str = trans_date.strftime("%Y-%m-%d")
                success, msg = add_transaction(date_str, trans_type, category, amount, description)
                if success:
                    st.success(f"✅ {trans_type} of ${amount:.2f} recorded!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to save: {msg}")

    st.divider()

    # Developer & Hackathon Demonstration Controls
    with st.expander("⚙️ Datasets & Demo Tools"):
        st.write("**Switch Datasets for Testing:**")
        if st.button("Load 6-Month History (for ML Training)", width="stretch"):
            seed_extended_historical_data()
            st.success("Loaded 6 months of historical transactions into database!")
            st.rerun()

        if st.button("Reset to Basic Demo (1 Month)", width="stretch"):
            reset_to_initial_demo()
            st.success("Database restored to standard 1-month demo!")
            st.rerun()

    st.info(
        "🧠 **AIML Hackathon Status:**\n"
        "- ✅ Stage 1: Core Dashboard & DB\n"
        "- ✅ Stage 2: ML Expense Forecaster\n"
        "- ✅ Stage 3: AI Savings Coach (Personalized Insights)"
    )

# -----------------------------------------------------------------------------
# 6. MAIN APPLICATION: TOP HEADER & TABS
# -----------------------------------------------------------------------------

# Top branding banner
st.markdown("""
<div class="header-card">
    <h1>🎓 SmartSpend AI</h1>
    <p>Empowering students to track spending, build smart budgets, and forecast future expenses using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# Fetch latest transactions from SQLite
df_transactions = load_transactions()

# Triple-tab layout: Tab 1 (Dashboard), Tab 2 (ML Forecaster), Tab 3 (AI Savings Coach)
tab_overview, tab_forecast, tab_coach = st.tabs([
    "📊 Overview & Transactions",
    "📈 Machine Learning Expense Forecaster",
    "💡 AI Savings Coach"
])

# =============================================================================
# TAB 1: OVERVIEW & TRANSACTIONS (STAGE 1)
# =============================================================================
with tab_overview:
    # Calculate summary metrics
    if not df_transactions.empty:
        total_income = df_transactions[df_transactions["type"] == "Income"]["amount"].sum()
        total_expenses = df_transactions[df_transactions["type"] == "Expense"]["amount"].sum()
    else:
        total_income = 0.0
        total_expenses = 0.0

    remaining_balance = total_income - total_expenses
    savings_rate = ((remaining_balance / total_income) * 100) if total_income > 0 else 0.0

    # 4 Key Metrics Display
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Income</div>
            <div class="metric-value" style="color: #10B981;">+${total_income:,.2f}</div>
            <div class="metric-sub">Recorded Inflow</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Expenses</div>
            <div class="metric-value" style="color: #EF4444;">-${total_expenses:,.2f}</div>
            <div class="metric-sub">Recorded Outflow</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        bal_color = "#10B981" if remaining_balance >= 0 else "#EF4444"
        bal_sign = "+" if remaining_balance > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Balance</div>
            <div class="metric-value" style="color: {bal_color};">{bal_sign}${remaining_balance:,.2f}</div>
            <div class="metric-sub">Current Cash Position</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        rate_color = "#10B981" if savings_rate >= 20 else ("#F59E0B" if savings_rate > 0 else "#EF4444")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Savings Rate</div>
            <div class="metric-value" style="color: {rate_color};">{savings_rate:.1f}%</div>
            <div class="metric-sub">Target: ≥ 20%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visualizations
    st.subheader("📊 Spending Breakdown")
    ch1, ch2 = st.columns(2)
    expenses_df = df_transactions[df_transactions["type"] == "Expense"]

    CHART_COLORS = ["#10B981", "#0F172A", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#64748B"]

    with ch1:
        if not expenses_df.empty:
            cat_spending = expenses_df.groupby("category")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)
            fig_donut = px.pie(
                cat_spending,
                values="amount",
                names="category",
                title="Spending by Category",
                hole=0.55,
                color_discrete_sequence=CHART_COLORS
            )
            fig_donut.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>")
            fig_donut.update_layout(margin=dict(t=40, b=20, l=20, r=20), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("ℹ️ No expense records found.")

    with ch2:
        if not expenses_df.empty:
            fig_bar = px.bar(
                cat_spending,
                x="amount",
                y="category",
                orientation="h",
                title="Category Totals ($)",
                color="amount",
                color_continuous_scale=["#A7F3D0", "#10B981", "#047857"]
            )
            fig_bar.update_layout(margin=dict(t=40, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>")
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("ℹ️ Bar chart will appear once expenses are added.")

    st.divider()

    # Transaction History
    st.subheader("📋 Transaction History")
    if df_transactions.empty:
        st.warning("No transactions recorded yet in the database.")
    else:
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            type_filter = st.selectbox("Filter by Type", options=["All", "Income", "Expense"], key="t1_filter")
        with fc2:
            search_query = st.text_input("Search Description or Category", placeholder="e.g. Books, Job", key="t1_search")
        with fc3:
            delete_options = {
                f"#{row['id']} | {row['date']} | {row['category']} (${row['amount']:.2f})": row['id']
                for _, row in df_transactions.iterrows()
            }
            selected_to_delete = st.selectbox("Delete Transaction", options=["-- Select to Delete --"] + list(delete_options.keys()))

        filtered_df = df_transactions.copy()
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df["type"] == type_filter]
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["description"].str.lower().str.contains(q, na=False) |
                filtered_df["category"].str.lower().str.contains(q, na=False)
            ]

        if selected_to_delete != "-- Select to Delete --":
            target_id = delete_options[selected_to_delete]
            if st.button("🗑️ Confirm Delete Selected Record"):
                success, msg = delete_transaction(target_id)
                if success:
                    st.success("Record deleted successfully.")
                    st.rerun()
                else:
                    st.error(f"Error: {msg}")

        display_df = filtered_df.copy()
        display_df["amount"] = display_df.apply(
            lambda r: f"+${r['amount']:,.2f}" if r['type'] == 'Income' else f"-${r['amount']:,.2f}",
            axis=1
        )
        display_df = display_df.rename(columns={"id": "ID", "date": "Date", "type": "Type", "category": "Category", "amount": "Amount", "description": "Description"})
        st.dataframe(display_df, width="stretch", hide_index=True)


# =============================================================================
# TAB 2: MACHINE LEARNING EXPENSE FORECASTER (STAGE 2)
# =============================================================================
with tab_forecast:
    st.markdown("### 📈 Time-Series Machine Learning Forecaster")
    st.write(
        "Using **Supervised Linear Regression** (`scikit-learn`), SmartSpend AI discovers spending velocity "
        "and trends from historical transactions to project future monthly costs and alert you of potential deficits."
    )

    # 1. Prepare time-series dataset from SQLite
    # If the database currently has only 1 month of data, offer one-click loading of the 6-month historical training set
    monthly_df = ml.prepare_monthly_timeseries(df_transactions)

    if len(monthly_df) < 2:
        st.warning(
            "⚠️ **Notice**: Machine Learning requires at least 2 distinct months of historical data to discover trends. "
            "Your database currently contains transactions from 1 month."
        )
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            if st.button("🚀 Load 6-Month Training Dataset Now"):
                seed_extended_historical_data()
                st.success("Loaded 6 months of historical transactions into database!")
                st.rerun()
        with col_btn2:
            st.caption(
                "Loading the 6-month student dataset populates realistic groceries, rent, and semester supplies "
                "from May 2026 to October 2026, allowing the ML model to train immediately."
            )
        st.stop()

    # 2. Interactive Forecasting Controls
    st.write("")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
    with ctrl_col1:
        forecast_horizon = st.slider(
            "Forecast Horizon (Months)",
            min_value=1,
            max_value=6,
            value=3,
            help="Number of future months to project."
        )
    with ctrl_col2:
        # Default income to average historical income
        avg_hist_income = float(monthly_df["Income"].mean()) if monthly_df["Income"].mean() > 0 else 1250.0
        expected_monthly_income = st.number_input(
            "Expected Monthly Income ($)",
            min_value=100.0,
            value=round(avg_hist_income, 2),
            step=50.0,
            help="Your anticipated monthly student income to evaluate budget risk."
        )
    with ctrl_col3:
        st.metric(
            label="Historical Training Window",
            value=f"{len(monthly_df)} Months",
            delta=f"{len(df_transactions)} total transactions"
        )

    # 3. Train the Regression Model
    combined_forecast_df, metrics = ml.train_linear_forecast_model(monthly_df, forecast_months=forecast_horizon)

    if metrics:
        next_month_pred = metrics["next_month_forecast"]
        risk_info = ml.assess_budget_risk(next_month_pred, expected_monthly_income)

        # 4. Display ML Key Metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Next Month Spend</div>
                <div class="metric-value" style="color: #0F172A;">${next_month_pred:,.2f}</div>
                <div class="metric-sub">Trend: {metrics['trend_direction']} ({metrics['percent_change']:+.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            proj_diff = risk_info["buffer"]
            diff_color = "#10B981" if proj_diff >= 0 else "#EF4444"
            diff_sign = "+" if proj_diff > 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Projected Net Balance</div>
                <div class="metric-value" style="color: {diff_color};">{diff_sign}${proj_diff:,.2f}</div>
                <div class="metric-sub">Income - Predicted Spend</div>
            </div>
            """, unsafe_allow_html=True)

        with mc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy (R² Score)</div>
                <div class="metric-value" style="color: #10B981;">{metrics['r2_score']:.2f}</div>
                <div class="metric-sub">Goodness of Linear Fit</div>
            </div>
            """, unsafe_allow_html=True)

        with mc4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error (MAE)</div>
                <div class="metric-value" style="color: #64748B;">±${metrics['mae']:,.2f}</div>
                <div class="metric-sub">Avg Dollar Variance</div>
            </div>
            """, unsafe_allow_html=True)

        # 5. Risk Assessment Banner
        st.markdown(f"""
        <div class="risk-box" style="background-color: #FFFFFF; border-left-color: {risk_info['badge_color']};">
            <b style="color: {risk_info['badge_color']}; font-size: 1.1rem;">Risk Evaluation: {risk_info['status']}</b>
            <p style="margin: 4px 0 0 0; color: #334155;">{risk_info['message']}</p>
        </div>
        """, unsafe_allow_html=True)

        # 6. Interactive Forecast Chart (Plotly)
        st.subheader("📊 Time-Series Spending & ML Projection Curve")

        hist_data = combined_forecast_df[combined_forecast_df["is_forecast"] == False]
        future_data = combined_forecast_df[combined_forecast_df["is_forecast"] == True]

        fig = go.Figure()

        # Historical Actual Line
        fig.add_trace(go.Scatter(
            x=hist_data["month_label"],
            y=hist_data["Expense"],
            mode="lines+markers",
            name="Actual Historical Expenses",
            line=dict(color="#0F172A", width=3),
            marker=dict(size=8, color="#0F172A"),
            hovertemplate="<b>%{x}</b><br>Actual: $%{y:,.2f}<extra></extra>"
        ))

        # Connect the last actual month to the first forecast point
        bridge_df = pd.concat([hist_data.tail(1), future_data.head(1)])
        fig.add_trace(go.Scatter(
            x=bridge_df["month_label"],
            y=bridge_df["Expense"],
            mode="lines",
            showlegend=False,
            line=dict(color="#10B981", width=3, dash="dash")
        ))

        # Future ML Forecast Projection Line
        fig.add_trace(go.Scatter(
            x=future_data["month_label"],
            y=future_data["Expense"],
            mode="lines+markers",
            name="ML Projected Forecast",
            line=dict(color="#10B981", width=3, dash="dash"),
            marker=dict(size=8, color="#10B981", symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Forecast: $%{y:,.2f}<extra></extra>"
        ))

        # Expected Income Reference Line
        fig.add_hline(
            y=expected_monthly_income,
            line_dash="dot",
            line_color="#3B82F6",
            annotation_text=f"Expected Income: ${expected_monthly_income:,.0f}",
            annotation_position="top right"
        )

        fig.update_layout(
            title="Monthly Expense Trajectory & Forecast ($)",
            xaxis_title="Timeline (Historical → Future)",
            yaxis_title="Total Expense ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=40, l=40, r=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")

        # 7. Category-Level Breakdown Forecast
        st.subheader("🎯 Predicted Spending by Category for Next Month")
        cat_forecast_df = ml.forecast_category_breakdown(df_transactions, next_month_pred)

        if not cat_forecast_df.empty:
            cat_c1, cat_c2 = st.columns([3, 2])
            with cat_c1:
                fig_cat_bar = px.bar(
                    cat_forecast_df,
                    x="category",
                    y="forecast_amount",
                    title="Next Month's Projected Category Allocation ($)",
                    labels={"category": "Category", "forecast_amount": "Forecast Amount ($)"},
                    color="forecast_amount",
                    color_continuous_scale=["#A7F3D0", "#10B981", "#065F46"]
                )
                fig_cat_bar.update_layout(margin=dict(t=40, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                fig_cat_bar.update_traces(hovertemplate="<b>%{x}</b><br>Forecast: $%{y:,.2f}<extra></extra>")
                st.plotly_chart(fig_cat_bar, width="stretch")

            with cat_c2:
                st.write("**Forecast Data Table:**")
                table_display = cat_forecast_df.copy()
                table_display["share_percent"] = table_display["share_percent"].apply(lambda p: f"{p:.1f}%")
                table_display["forecast_amount"] = table_display["forecast_amount"].apply(lambda a: f"${a:,.2f}")
                table_display = table_display.rename(columns={
                    "category": "Category",
                    "amount": "Historical Sum",
                    "share_percent": "Historical Share",
                    "forecast_amount": "Next Month Forecast"
                })
                st.dataframe(table_display[["Category", "Historical Share", "Next Month Forecast"]], width="stretch", hide_index=True)

        # 8. Educational AIML Section
        with st.expander("🎓 How the ML Model Works: A Student's Guide to Linear Regression"):
            st.markdown("""
            ### 1. The Core Equation:
            $$y = mx + c$$
            - **$x$ (Independent Variable)**: Sequential time step representing each month (0, 1, 2, ...).
            - **$y$ (Dependent Variable)**: Total student expenses incurred in that month.
            - **$m$ (Slope)**: Spending acceleration. Positive slope means spending is increasing; negative means spending is decreasing.
            - **$c$ (Intercept)**: Baseline minimum living expenses when $x = 0$.

            ### 2. Evaluation Metrics:
            - **$R^2$ Score**: Tells us what percentage of expense variance is explained by time. Closer to $1.0$ indicates consistent historical patterns.
            - **Mean Absolute Error (MAE)**:
              $$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^n |y_i - \\hat{y}_i|$$
              Gives the average dollar error margin of our forecast.
            """)

# =============================================================================
# TAB 3: AI SAVINGS COACH (STAGE 3)
# =============================================================================
with tab_coach:
    st.markdown("### 💡 AI Student Savings Coach")
    st.write(
        "Your personalized financial advisor. SmartSpend AI analyzes your spending habits, "
        "classifies your **Student Persona**, flags **Hidden Money Leaks**, and generates actionable, personalized budgeting advice."
    )

    # 1. Monthly Income Setting for Coaching
    coach_c1, coach_c2 = st.columns([2, 3])
    with coach_c1:
        coach_income = st.number_input(
            "Your Estimated Monthly Income ($)",
            min_value=50.0,
            value=1250.0,
            step=50.0,
            key="coach_income_input",
            help="Enter your typical monthly income (part-time job, allowance, scholarship)."
        )
    with coach_c2:
        st.caption(
            "The AI Savings Coach uses this baseline to calculate your **Financial Health Score (0-100)**, "
            "evaluate the **Student 50/30/20 Rule**, and detect budget leaks."
        )

    # 2. Generate Base Coach Analytics & Persona
    coach_data = ml.generate_ai_savings_coach(df_transactions, user_monthly_income=coach_income)
    persona = ac.detect_student_persona(df_transactions, coach_income)

    # 3. Student Financial Persona Card
    st.markdown(f"""
    <div class="persona-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF;">
                {persona['emoji']} {persona['name']}
            </span>
            <span style="background-color: #10B981; color: #FFFFFF; font-size: 0.82rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                Identified Student Persona
            </span>
        </div>
        <p style="margin: 0 0 12px 0; color: #94A3B8; font-style: italic; font-size: 1.0rem;">
            "{persona['tagline']}"
        </p>
        <p style="margin: 0 0 12px 0; color: #E2E8F0; font-size: 0.96rem; line-height: 1.5;">
            {persona['description']}
        </p>
        <div style="display: flex; gap: 24px; font-size: 0.88rem;">
            <div><b style="color: #10B981;">💪 Strength:</b> <span style="color: #CBD5E1;">{persona['strength']}</span></div>
            <div><b style="color: #F59E0B;">⚠️ Watch Out:</b> <span style="color: #CBD5E1;">{persona['watch_out']}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Health Score & Key Overview Metrics
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {coach_data['score_color']};">
            <div class="metric-label">Financial Health Score</div>
            <div class="metric-value" style="color: {coach_data['score_color']};">{coach_data['health_score']}/100</div>
            <div class="metric-sub">Grade: {coach_data['grade']}</div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly Average Spend</div>
            <div class="metric-value" style="color: #0F172A;">${coach_data['total_expense']:,.2f}</div>
            <div class="metric-sub">Across {coach_data.get('num_months', 1)} active month(s)</div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        bal_col = '#10B981' if coach_data['net_savings'] >= 0 else '#EF4444'
        bal_sg = '+' if coach_data['net_savings'] >= 0 else '-'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly Net Savings</div>
            <div class="metric-value" style="color: {bal_col};">
                {bal_sg}${abs(coach_data['net_savings']):,.2f}
            </div>
            <div class="metric-sub">Monthly Retained Cash</div>
        </div>
        """, unsafe_allow_html=True)

    with sc4:
        sr_col = '#10B981' if coach_data['savings_rate'] >= 20 else ('#F59E0B' if coach_data['savings_rate'] > 0 else '#EF4444')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Savings Rate</div>
            <div class="metric-value" style="color: {sr_col};">
                {coach_data['savings_rate']:.1f}%
            </div>
            <div class="metric-sub">Benchmark: ≥ 20.0%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 5. Student 50/30/20 Budget Breakdown
    st.subheader("⚖️ Student 50/30/20 Budget Benchmark")
    bd = coach_data["breakdown"]

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        needs_color = "#10B981" if bd["needs_pct"] <= 55 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card" style="text-align: left;">
            <div class="metric-label">🏠 Needs (Target: ≤ 50%)</div>
            <div class="metric-value" style="color: {needs_color}; font-size: 1.4rem;">{bd['needs_pct']:.1f}% (${bd['needs_spent']:,.2f})</div>
            <div class="metric-sub">Rent, Groceries, Tuition, Transit</div>
        </div>
        """, unsafe_allow_html=True)

    with b_col2:
        wants_color = "#10B981" if bd["wants_pct"] <= 30 else "#F59E0B"
        st.markdown(f"""
        <div class="metric-card" style="text-align: left;">
            <div class="metric-label">🍕 Wants (Target: ≤ 30%)</div>
            <div class="metric-value" style="color: {wants_color}; font-size: 1.4rem;">{bd['wants_pct']:.1f}% (${bd['wants_spent']:,.2f})</div>
            <div class="metric-sub">Dining out, Entertainment, Coffee</div>
        </div>
        """, unsafe_allow_html=True)

    with b_col3:
        savings_color = "#10B981" if bd["savings_pct"] >= 20 else "#EF4444"
        st.markdown(f"""
        <div class="metric-card" style="text-align: left;">
            <div class="metric-label">💰 Savings (Target: ≥ 20%)</div>
            <div class="metric-value" style="color: {savings_color}; font-size: 1.4rem;">{bd['savings_pct']:.1f}% (${bd['savings_spent']:,.2f})</div>
            <div class="metric-sub">Emergency fund & personal goals</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 6. Spending Leakage Diagnostics
    st.subheader("🔍 AI Spending Leakage Detector")
    leaks = ac.diagnose_spending_leaks(df_transactions, coach_income)

    if leaks:
        for leak in leaks:
            badge_bg = "#FEF2F2" if leak["severity"] == "High" else "#FFFBEB"
            badge_col = "#DC2626" if leak["severity"] == "High" else "#D97706"
            st.markdown(f"""
            <div class="leak-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #0F172A;">
                        {leak['emoji']} {leak['title']}
                    </span>
                    <span style="background-color: {badge_bg}; color: {badge_col}; font-size: 0.78rem; font-weight: 700; padding: 3px 9px; border-radius: 9999px;">
                        {leak['severity']} Severity
                    </span>
                </div>
                <p style="margin: 0 0 6px 0; color: #475569; font-size: 0.92rem;">
                    {leak['detail']}
                </p>
                <div style="font-size: 0.88rem; color: #065F46; background-color: #ECFDF5; padding: 6px 12px; border-radius: 6px;">
                    <b>💡 Smart Fix:</b> {leak['fix']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎉 No severe spending leaks detected! Your daily spending is well-controlled.")

    st.write("")

    # 7. Personalized Recommendations
    st.subheader("🎯 Prioritized Action Cards")
    if coach_data["recommendations"]:
        for rec in coach_data["recommendations"]:
            st.markdown(f"""
            <div class="coach-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">
                        {rec['icon']} {rec['title']}
                    </span>
                    <span class="coach-badge">{rec['impact']}</span>
                </div>
                <p style="margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">
                    {rec['advice']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # 8. Interactive 'What-If' Rebalancing Simulator
    st.subheader("🎛️ Interactive 'What-If' Habit Simulator")
    st.write("Adjust the sliders below to see in real-time how simple behavioral changes will improve your monthly cash flow:")

    w1, w2, w3 = st.columns(3)
    with w1:
        cut_food = st.slider("Trim Food & Groceries by (%)", min_value=0, max_value=40, value=15, step=5, help="e.g. Cook lunch 2 days/week instead of buying campus food.")
    with w2:
        cut_ent = st.slider("Trim Entertainment by (%)", min_value=0, max_value=50, value=20, step=5, help="e.g. Share group streaming or look for student freebies.")
    with w3:
        side_gig = st.number_input("Add Side-Income ($/mo)", min_value=0.0, value=75.0, step=25.0, help="e.g. 3 hours/week of peer tutoring or campus library assistant.")

    sim_res = ac.simulate_rebalancing(df_transactions, coach_income, cut_food, cut_ent, side_gig)

    # Show Simulation Impact Cards
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly Expense Cuts</div>
            <div class="metric-value" style="color: #10B981;">+${sim_res['total_saved']:,.2f}</div>
            <div class="metric-sub">Food: ${sim_res['food_savings']:.2f} | Ent: ${sim_res['ent_savings']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">New Monthly Surplus</div>
            <div class="metric-value" style="color: #10B981;">${sim_res['new_surplus']:,.2f}</div>
            <div class="metric-sub">Was ${sim_res['old_surplus']:,.2f} (+${sim_res['improvement']:,.2f} gained!)</div>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">New Savings Rate</div>
            <div class="metric-value" style="color: #10B981;">{sim_res['new_savings_rate']:.1f}%</div>
            <div class="metric-sub">Up from {sim_res['old_savings_rate']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 9. Context-Aware Student Financial Q&A
    st.subheader("💬 Ask the AI Savings Coach")
    st.write("Get direct, plain-English guidance based on your actual database numbers:")

    faq_options = {
        "Where is my money leaking the most?": "where_leaking",
        "How can I realistically save $50 more this month?": "save_50_more",
        "Should I build an emergency fund first?": "emergency_fund",
        "What are practical hacks to cut textbook/academic costs?": "textbook_hack",
        "How do student credit cards work and should I get one?": "credit_cards"
    }

    selected_q = st.selectbox("Select a question for your AI Coach:", list(faq_options.keys()))

    # Build student financial context
    top_cat_name = "Food & Groceries"
    top_cat_amt = 150.0
    if not df_transactions.empty:
        exp_sub = df_transactions[df_transactions["type"] == "Expense"]
        if not exp_sub.empty:
            cg = exp_sub.groupby("category")["amount"].sum()
            top_cat_name = str(cg.idxmax())
            top_cat_amt = float(cg.max())

    fin_ctx = {
        "income": coach_income,
        "expense": coach_data["total_expense"],
        "surplus": coach_data["net_savings"],
        "savings_rate": coach_data["savings_rate"],
        "top_category": top_cat_name,
        "top_amount": top_cat_amt,
        "persona": persona["name"]
    }

    answer_text = ac.get_coaching_answer(faq_options[selected_q], fin_ctx)
    st.info(answer_text)

    # 10. 7-Day Student Financial Habit Challenge
    with st.expander("📅 View Your Personalized 7-Day Student Financial Challenge"):
        challenge_days = ac.generate_weekly_challenge(coach_data["health_score"])
        st.write("Complete one small daily task this week to build financial discipline without feeling restricted:")
        for day in challenge_days:
            st.markdown(f"**{day['day']} - {day['task']}**: {day['desc']}")

    # 11. Interactive Savings Goal Calculator
    st.divider()
    st.subheader("🎯 Student Savings Goal Planner")
    st.write("Plan a specific milestone (e.g. laptop, travel, emergency fund) and get an AI savings roadmap:")

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        goal_name = st.selectbox(
            "Goal Target",
            options=["Emergency Fund", "New Laptop / Tech", "Semester Textbooks & Lab Kits", "Spring Break Trip", "Custom Goal"]
        )
    with gc2:
        target_amount = st.number_input("Target Amount ($)", min_value=25.0, value=300.0, step=25.0)
    with gc3:
        months_to_save = st.slider("Timeframe (Months)", min_value=1, max_value=12, value=3)

    goal_plan = ml.calculate_savings_goal_plan(
        goal_name,
        target_amount,
        months_to_save,
        coach_data["net_savings"]
    )

    if goal_plan["is_on_track"]:
        st.success(
            f"🎉 **You are on track!** Saving **${goal_plan['required_monthly']:,.2f}/month** for {months_to_save} month(s) "
            f"will achieve your **{goal_name}** (${target_amount:,.2f}). Your current monthly surplus gives you an extra buffer of ${goal_plan['surplus_buffer']:,.2f}."
        )
    else:
        st.warning(
            f"⚠️ **Gap Detected**: Reaching **{goal_name}** requires **${goal_plan['required_monthly']:,.2f}/month**, "
            f"leaving a monthly gap of **${goal_plan['monthly_gap']:,.2f}**. Follow the AI Coach recommendations above to bridge this gap!"
        )

# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "💡 **SmartSpend AI (Stage 3)** | Built with Python, Streamlit, SQLite, and Scikit-learn. "
    "Featuring Time-Series ML Forecasting & Personalized AI Savings Coach for students."
)
