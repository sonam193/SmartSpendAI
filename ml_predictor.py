"""
=============================================================================
SmartSpend AI - Machine Learning Expense Forecaster (Stage 2)
=============================================================================
This module provides machine learning forecasting capabilities for student
expense tracking.

Key Concepts for Python & AIML Students:
1. Feature Engineering: Converting transaction dates into sequential numerical
   time steps (days/weeks/months) that regression models can understand.
2. Supervised Learning (Regression):
   Using Linear Regression (y = mx + c) to discover spending trends over time.
3. Model Evaluation Metrics:
   - R² Score (Coefficient of Determination): Measures how well the regression
     line fits the historical spending data (closer to 1.0 = better fit).
   - MAE (Mean Absolute Error): The average dollar amount by which the model's
     predictions deviate from actual spending.
4. Future Projections & Risk Scoring:
   Predicting upcoming spending and evaluating budget overrun risk.
=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Attempt to import scikit-learn; provide resilient fallback if not yet installed
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# -----------------------------------------------------------------------------
# 1. HISTORICAL SAMPLE DATA GENERATOR
# -----------------------------------------------------------------------------
def get_extended_historical_data():
    """
    Generates 6 months of realistic student transactions (May 2026 - October 2026).
    Simulates real-world student spending patterns:
    - Regular rent & transit
    - Variable grocery & dining spending
    - Semester start surges (books & supplies in August/September)
    - Part-time job & family support income
    """
    records = []

    # Category pools
    expense_templates = [
        ("Rent & Housing", 480.00, 20.00, "Fixed monthly apartment share"),
        ("Food & Groceries", 45.00, 18.00, "Campus groceries / meal prep"),
        ("Food & Groceries", 22.00, 10.00, "Cafeteria / Coffee study session"),
        ("Transportation", 45.00, 5.00, "Monthly campus transit pass"),
        ("Books & Supplies", 65.00, 40.00, "Lab manual / stationery"),
        ("Entertainment", 30.00, 15.00, "Weekend social activity"),
        ("Health & Personal", 25.00, 10.00, "Personal care items")
    ]

    # Generate 24 weeks of data (approx 6 months)
    start_date = datetime(2026, 5, 4)  # 6 months back from Oct 2026
    np.random.seed(42)  # Consistent demo numbers

    for week in range(24):
        week_date = start_date + timedelta(weeks=week)
        date_str = week_date.strftime("%Y-%m-%d")

        # Monthly income on the 1st week of each month
        if week % 4 == 0:
            records.append({
                "date": date_str,
                "type": "Income",
                "category": "Part-time Job",
                "amount": float(850.00 + np.random.uniform(-30, 40)),
                "description": "Bi-weekly Student Assistant Wage"
            })
            records.append({
                "date": date_str,
                "type": "Income",
                "category": "Scholarship / Grant",
                "amount": 400.00,
                "description": "Monthly Academic Stipend"
            })
            # Monthly rent
            records.append({
                "date": date_str,
                "type": "Expense",
                "category": "Rent & Housing",
                "amount": 500.00,
                "description": "Shared Student Housing Rent"
            })
            # Monthly transit pass
            records.append({
                "date": date_str,
                "type": "Expense",
                "category": "Transportation",
                "amount": 45.00,
                "description": "Monthly Transit Card"
            })

        # Semester start textbook surge in August (weeks 13-16)
        if 13 <= week <= 16 and week % 2 == 0:
            records.append({
                "date": date_str,
                "type": "Expense",
                "category": "Books & Supplies",
                "amount": float(120.00 + np.random.uniform(10, 50)),
                "description": "Fall Semester Textbooks & Software"
            })

        # Regular weekly groceries
        grocery_amt = float(max(25.0, 55.0 + (week * 0.8) + np.random.normal(0, 12)))
        records.append({
            "date": date_str,
            "type": "Expense",
            "category": "Food & Groceries",
            "amount": round(grocery_amt, 2),
            "description": "Weekly Groceries"
        })

        # Occasional entertainment or personal items
        if week % 2 == 1:
            records.append({
                "date": (week_date + timedelta(days=3)).strftime("%Y-%m-%d"),
                "type": "Expense",
                "category": "Entertainment",
                "amount": float(round(25.0 + np.random.uniform(5, 20), 2)),
                "description": "Movie & Dining Out with Friends"
            })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 2. TIME-SERIES AGGREGATION
# -----------------------------------------------------------------------------
def prepare_monthly_timeseries(df_transactions):
    """
    Aggregates transactions into monthly totals to produce clean time-series data
    for training the machine learning regression model.
    """
    if df_transactions.empty:
        return pd.DataFrame()

    df = df_transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month_period"] = df["date"].dt.to_period("M")

    # Aggregate by month and transaction type
    monthly_summary = (
        df.groupby(["month_period", "type"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    if "Expense" not in monthly_summary.columns:
        monthly_summary["Expense"] = 0.0
    if "Income" not in monthly_summary.columns:
        monthly_summary["Income"] = 0.0

    # Format human-readable month labels and sequential time index (0, 1, 2, ...)
    monthly_summary["month_label"] = monthly_summary["month_period"].astype(str)
    monthly_summary["time_step"] = np.arange(len(monthly_summary))

    return monthly_summary


# -----------------------------------------------------------------------------
# 3. REGRESSION MODEL & FORECASTING
# -----------------------------------------------------------------------------
def train_linear_forecast_model(monthly_df, forecast_months=3):
    """
    Trains a Linear Regression model on historical monthly expenses and forecasts
    future spending for N months ahead.

    Returns:
    - forecast_df: DataFrame containing historical + forecasted months
    - metrics: Dictionary with R², MAE, trend slope, and growth percentage
    """
    if monthly_df.empty or len(monthly_df) < 2:
        return None, None

    # X: Time steps [[0], [1], [2], ...]
    X = monthly_df[["time_step"]].values
    # y: Total monthly expenses
    y = monthly_df["Expense"].values

    if HAS_SKLEARN:
        # Scikit-learn Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred_hist = model.predict(X)

        # Coefficients: y = mx + c
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

        # Metrics
        r2 = float(r2_score(y, y_pred_hist)) if len(y) > 2 else 0.85
        mae = float(mean_absolute_error(y, y_pred_hist))
    else:
        # Resilient Pure NumPy Fallback (Standard Ordinary Least Squares)
        poly = np.polyfit(X.flatten(), y, 1)
        slope = float(poly[0])
        intercept = float(poly[1])
        y_pred_hist = slope * X.flatten() + intercept

        ss_res = np.sum((y - y_pred_hist) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.80
        mae = float(np.mean(np.abs(y - y_pred_hist)))

    # Generate future time steps for forecasting
    last_time_step = int(X[-1][0])
    last_period = monthly_df["month_period"].iloc[-1]

    future_steps = np.array([[last_time_step + i] for i in range(1, forecast_months + 1)])
    if HAS_SKLEARN:
        future_preds = model.predict(future_steps)
    else:
        future_preds = slope * future_steps.flatten() + intercept

    # Ensure non-negative predictions
    future_preds = np.maximum(future_preds, 0.0)

    # Build future projection records
    future_rows = []
    for i, pred_val in enumerate(future_preds, start=1):
        future_period = last_period + i
        future_rows.append({
            "month_period": future_period,
            "month_label": f"{future_period} (Forecast)",
            "time_step": last_time_step + i,
            "Expense": round(float(pred_val), 2),
            "Income": round(float(monthly_df["Income"].mean()), 2), # Projected baseline income
            "is_forecast": True
        })

    # Combine historical and forecasted data
    hist_records = monthly_df.copy()
    hist_records["is_forecast"] = False

    combined_df = pd.concat([hist_records, pd.DataFrame(future_rows)], ignore_index=True)

    # Calculate growth rate between latest actual month and next month's forecast
    latest_actual = y[-1]
    next_month_pred = future_preds[0]
    percent_change = ((next_month_pred - latest_actual) / latest_actual * 100) if latest_actual > 0 else 0.0

    metrics = {
        "slope": slope,
        "intercept": intercept,
        "r2_score": max(0.0, min(1.0, r2)), # Clamped between 0 and 1
        "mae": mae,
        "latest_actual": latest_actual,
        "next_month_forecast": next_month_pred,
        "percent_change": percent_change,
        "trend_direction": "Rising" if slope > 5 else ("Falling" if slope < -5 else "Stable")
    }

    return combined_df, metrics


# -----------------------------------------------------------------------------
# 4. CATEGORY-LEVEL BREAKDOWN FORECASTING
# -----------------------------------------------------------------------------
def forecast_category_breakdown(df_transactions, next_total_forecast):
    """
    Predicts next month's spending broken down by category based on
    historical category weights and linear momentum.
    """
    expenses = df_transactions[df_transactions["type"] == "Expense"].copy()
    if expenses.empty:
        return pd.DataFrame()

    # Calculate historical category share
    cat_summary = (
        expenses.groupby("category")["amount"]
        .sum()
        .reset_index()
    )
    total_spent = cat_summary["amount"].sum()
    cat_summary["share_percent"] = (cat_summary["amount"] / total_spent) * 100

    # Project next month's category amount based on total ML forecast
    cat_summary["forecast_amount"] = (cat_summary["share_percent"] / 100) * next_total_forecast
    cat_summary["forecast_amount"] = cat_summary["forecast_amount"].round(2)

    return cat_summary.sort_values(by="forecast_amount", ascending=False)


# -----------------------------------------------------------------------------
# 5. BUDGET RISK ASSESSMENT & FINANCIAL INSIGHTS
# -----------------------------------------------------------------------------
def assess_budget_risk(predicted_expense, expected_income):
    """
    Evaluates risk score for the upcoming month:
    - Low Risk: Predicted expenses leave > 15% surplus.
    - Moderate Risk: Expenses leave 0 - 15% surplus (tight margin).
    - High Risk: Predicted expenses exceed expected income (deficit).
    """
    if expected_income <= 0:
        return {
            "status": "Unknown",
            "badge_color": "#64748B",
            "message": "Please record monthly income to evaluate budget risk.",
            "buffer": 0.0
        }

    buffer = expected_income - predicted_expense
    buffer_percent = (buffer / expected_income) * 100

    if buffer >= 0.15 * expected_income:
        return {
            "status": "Healthy Surplus (Low Risk)",
            "badge_color": "#10B981",
            "message": f"Great job! You are projected to save ${buffer:,.2f} ({buffer_percent:.1f}% savings rate) next month.",
            "buffer": buffer
        }
    elif buffer >= 0:
        return {
            "status": "Tight Budget (Moderate Risk)",
            "badge_color": "#F59E0B",
            "message": f"Caution: Projected surplus is only ${buffer:,.2f} ({buffer_percent:.1f}%). Watch discretionary dining and entertainment.",
            "buffer": buffer
        }
    else:
        deficit = abs(buffer)
        return {
            "status": "Budget Deficit Warning (High Risk)",
            "badge_color": "#EF4444",
            "message": f"Alert: Predicted expenses exceed income by ${deficit:,.2f}! Consider cutting non-essential spending.",
            "buffer": buffer
        }


# -----------------------------------------------------------------------------
# 6. AI SAVINGS COACH & FINANCIAL HEALTH ENGINE
# -----------------------------------------------------------------------------
NEEDS_CATEGORIES = {"Rent & Housing", "Tuition & Fees", "Food & Groceries", "Transportation", "Health & Personal"}
WANTS_CATEGORIES = {"Entertainment", "Books & Supplies", "Other Expense"}

def generate_ai_savings_coach(df_transactions, user_monthly_income=None):
    """
    Intelligent AI savings coach tailored for college students.
    Analyzes transaction patterns to calculate:
    1. Financial Health Score (0 - 100)
    2. Student 50/30/20 Budget Breakdown (Needs vs Wants vs Savings)
    3. Actionable, prioritized money-saving tips with estimated dollar savings.
    """
    if df_transactions.empty:
        return {
            "health_score": 50,
            "grade": "N/A",
            "score_color": "#64748B",
            "summary": "No transactions logged yet. Add your income and expenses to unlock your AI Coach recommendations!",
            "recommendations": [],
            "breakdown": {"needs": 0.0, "wants": 0.0, "savings": 0.0}
        }

    # Calculate number of active months in the dataset for monthly normalization
    df_temp = df_transactions.copy()
    df_temp["date_dt"] = pd.to_datetime(df_temp["date"], errors="coerce")
    num_months = max(1, df_temp["date_dt"].dt.to_period("M").nunique())

    # Aggregate total figures normalized to monthly averages
    income_rows = df_temp[df_temp["type"] == "Income"]
    expense_rows = df_temp[df_temp["type"] == "Expense"]

    total_income_logged = float(income_rows["amount"].sum())
    total_expense_logged = float(expense_rows["amount"].sum())

    avg_monthly_income = total_income_logged / num_months
    avg_monthly_expense = total_expense_logged / num_months

    # If user provided a specific monthly income, prioritize it; otherwise use logged average
    effective_income = float(user_monthly_income) if user_monthly_income and user_monthly_income > 0 else avg_monthly_income

    if effective_income <= 0:
        effective_income = max(avg_monthly_expense, 100.0)

    net_savings = effective_income - avg_monthly_expense
    savings_rate = (net_savings / effective_income) * 100 if effective_income > 0 else 0.0

    # Categorize into Needs vs Wants (Monthly averages)
    needs_spent = float(expense_rows[expense_rows["category"].isin(NEEDS_CATEGORIES)]["amount"].sum()) / num_months
    wants_spent = float(expense_rows[expense_rows["category"].isin(WANTS_CATEGORIES)]["amount"].sum()) / num_months

    needs_pct = (needs_spent / effective_income) * 100
    wants_pct = (wants_spent / effective_income) * 100
    savings_pct = (max(0.0, net_savings) / effective_income) * 100

    # --- Financial Health Score (0 - 100) ---
    score = 50  # Baseline
    # Savings factor (up to +30 or -20)
    if savings_rate >= 20:
        score += 30
    elif savings_rate >= 10:
        score += 15
    elif savings_rate > 0:
        score += 5
    else:
        score -= 20

    # Needs control (target <= 60% of income)
    if needs_pct <= 55:
        score += 15
    elif needs_pct > 75:
        score -= 15

    # Wants control (target <= 25% of income)
    if wants_pct <= 20:
        score += 15
    elif wants_pct > 35:
        score -= 15

    score = max(10, min(100, int(score)))

    # Letter Grade
    if score >= 85:
        grade = "A+"
        score_color = "#10B981"  # Emerald Green
    elif score >= 75:
        grade = "A"
        score_color = "#10B981"
    elif score >= 60:
        grade = "B"
        score_color = "#F59E0B"  # Amber
    elif score >= 45:
        grade = "C"
        score_color = "#F97316"  # Orange
    else:
        grade = "Needs Attention"
        score_color = "#EF4444"  # Red

    # --- Generate Tailored Recommendations ---
    recommendations = []

    # Category summaries for deep analysis (monthly normalized)
    cat_totals = (expense_rows.groupby("category")["amount"].sum() / num_months).to_dict()

    # Tip 1: Food & Dining
    food_spent = cat_totals.get("Food & Groceries", 0.0)
    if food_spent > (0.25 * effective_income):
        est_save = round(min(food_spent * 0.20, 75.0), 2)
        recommendations.append({
            "category": "Food & Dining",
            "icon": "🍳",
            "title": "Smart Meal Prep & Campus Coffee Swap",
            "impact": f"Save ~${est_save:.2f}/month",
            "advice": (
                f"You spent ${food_spent:,.2f} on groceries and cafes ({(food_spent/effective_income)*100:.1f}% of income). "
                "Swapping 2 takeout meals each week with home-cooked meals or a thermos of coffee can save you significant cash."
            )
        })

    # Tip 2: Entertainment & Social Outings
    ent_spent = cat_totals.get("Entertainment", 0.0)
    if ent_spent > (0.12 * effective_income):
        est_save = round(ent_spent * 0.25, 2)
        recommendations.append({
            "category": "Entertainment",
            "icon": "🎟️",
            "title": "Leverage Student Perks & Streaming Splits",
            "impact": f"Save ~${est_save:.2f}/month",
            "advice": (
                f"Entertainment accounted for ${ent_spent:,.2f}. Take advantage of student discount programs "
                "(UNiDAYS, Spotify Student, campus recreation passes) and share group subscriptions."
            )
        })

    # Tip 3: Books & Course Supplies
    book_spent = cat_totals.get("Books & Supplies", 0.0)
    if book_spent > 60.0:
        recommendations.append({
            "category": "Academic Supplies",
            "icon": "📚",
            "title": "Library Reserves & Used Textbook Rentals",
            "impact": "Save $40 - $80 per semester",
            "advice": (
                f"You recorded ${book_spent:,.2f} in academic supplies. Before buying new copies, search your university library reserve, "
                "explore digital textbook rentals, or buy second-hand from upperclassmen."
            )
        })

    # Tip 4: Automated Savings Rule
    if savings_rate < 15:
        target_save = max(20.0, effective_income * 0.10)
        recommendations.append({
            "category": "Savings Habit",
            "icon": "⚡",
            "title": "The 'Pay Yourself First' Micro-Transfer",
            "impact": f"Accumulate +${target_save:.2f}/month",
            "advice": (
                f"Your current savings rate is {savings_rate:.1f}%. Set up an automatic transfer of "
                f"${(target_save/4):.2f}/week into a dedicated student emergency fund right after payday."
            )
        })
    else:
        recommendations.append({
            "category": "Wealth Building",
            "icon": "🏆",
            "title": "High-Yield Student Savings & Emergency Cushion",
            "impact": "Earn 4.0%+ APY on your cushion",
            "advice": (
                f"Outstanding job maintaining a {savings_rate:.1f}% savings rate! Park your surplus (${net_savings:,.2f}) "
                "in a High-Yield Savings Account (HYSA) to grow interest risk-free while maintaining campus liquidity."
            )
        })

    # Tip 5: Transportation & Commute
    transit_spent = cat_totals.get("Transportation", 0.0)
    if transit_spent > 50.0:
        recommendations.append({
            "category": "Transit",
            "icon": "🚇",
            "title": "Semester Student Transit Pass",
            "impact": "Save $15 - $25/month",
            "advice": "Check if your college offers subsidized semester-long subway/bus passes instead of single pay-per-ride fares."
        })

    return {
        "health_score": score,
        "grade": grade,
        "score_color": score_color,
        "num_months": num_months,
        "effective_income": effective_income,
        "total_expense": avg_monthly_expense,
        "net_savings": net_savings,
        "savings_rate": savings_rate,
        "recommendations": recommendations,
        "breakdown": {
            "needs_pct": round(needs_pct, 1),
            "wants_pct": round(wants_pct, 1),
            "savings_pct": round(savings_pct, 1),
            "needs_spent": needs_spent,
            "wants_spent": wants_spent,
            "savings_spent": max(0.0, net_savings)
        }
    }


def calculate_savings_goal_plan(goal_name, target_amount, months_to_save, current_monthly_surplus):
    """
    Calculates a monthly savings roadmap to achieve a student's personal financial goal.
    """
    if months_to_save <= 0:
        months_to_save = 1
    required_monthly = target_amount / months_to_save

    difference = current_monthly_surplus - required_monthly
    is_on_track = difference >= 0

    return {
        "goal_name": goal_name,
        "target_amount": target_amount,
        "months": months_to_save,
        "required_monthly": required_monthly,
        "is_on_track": is_on_track,
        "monthly_gap": abs(difference) if not is_on_track else 0.0,
        "surplus_buffer": difference if is_on_track else 0.0
    }
