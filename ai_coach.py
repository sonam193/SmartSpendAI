"""
=============================================================================
SmartSpend AI - AI Savings Coach Engine (Stage 3)
=============================================================================
This module provides personalized, understandable financial coaching for
college students.

Key Features:
1. Student Financial Persona Detection (e.g. 'Social Foodie', 'Campus Commuter').
2. Spending Leakage Diagnostics (weekend spikes, micro-purchases, subscription drain).
3. Interactive 'What-If' Rebalancing Simulator (calculates impact of habit shifts).
4. Context-Aware Student Q&A Engine (answers questions using the student's actual numbers).
5. 7-Day Student Financial Challenge Generator (practical daily micro-habits).
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
from ml_predictor import generate_ai_savings_coach


# -----------------------------------------------------------------------------
# 1. STUDENT PERSONA DETECTION
# -----------------------------------------------------------------------------
def detect_student_persona(df_transactions, effective_income):
    """
    Analyzes transaction patterns to assign a relatable student spending persona.
    Helps students understand their financial personality without shame or complex jargon.
    """
    if df_transactions.empty or effective_income <= 0:
        return {
            "name": "The Fresh Explorer",
            "emoji": "🌱",
            "tagline": "Starting your financial journey.",
            "description": "Log a few more campus transactions to unlock your full student financial personality profile!",
            "strength": "Clean slate with high potential.",
            "watch_out": "Unrecorded cash and card spending."
        }

    expenses = df_transactions[df_transactions["type"] == "Expense"]
    if expenses.empty:
        return {
            "name": "The Prudent Saver",
            "emoji": "🛡️",
            "tagline": "Zero outflow recorded so far.",
            "description": "You have recorded income with minimal to no expenses.",
            "strength": "High cash retention.",
            "watch_out": "Make sure to record all daily purchases."
        }

    total_expense = expenses["amount"].sum()
    cat_totals = expenses.groupby("category")["amount"].sum().to_dict()

    food_share = (cat_totals.get("Food & Groceries", 0.0) / total_expense) * 100
    ent_share = (cat_totals.get("Entertainment", 0.0) / total_expense) * 100
    transit_share = (cat_totals.get("Transportation", 0.0) / total_expense) * 100
    books_share = (cat_totals.get("Books & Supplies", 0.0) / total_expense) * 100
    savings_rate = ((effective_income - total_expense) / effective_income) * 100

    # Persona classification logic
    if savings_rate >= 25:
        return {
            "name": "The Master Budgeter",
            "emoji": "🎯",
            "tagline": "Disciplined, strategic, and forward-thinking.",
            "description": f"You maintain a stellar {savings_rate:.1f}% savings rate! You keep non-essential costs low and prioritize financial security.",
            "strength": "Consistently retains a cash surplus every month.",
            "watch_out": "Ensure you don't over-restrict yourself—budget a little for guilt-free fun!"
        }
    elif food_share >= 32:
        return {
            "name": "The Campus Foodie",
            "emoji": "🍕",
            "tagline": "Fueled by coffee runs, takeout, and campus snacks.",
            "description": f"Over {food_share:.1f}% of your spending goes toward dining and groceries. Quick meals and coffees are your primary financial leak.",
            "strength": "Enjoys social dining and campus life.",
            "watch_out": "Small $5-$12 food runs add up to hundreds by the end of the month."
        }
    elif ent_share >= 18:
        return {
            "name": "The Social Butterfly",
            "emoji": "🎉",
            "tagline": "Life of the party and weekend adventures.",
            "description": f"Entertainment and social events make up {ent_share:.1f}% of your expenses. You value experiences with friends.",
            "strength": "Strong campus network and active social life.",
            "watch_out": "Spontaneous weekend outings can derail your monthly budget."
        }
    elif transit_share >= 15:
        return {
            "name": "The Campus Commuter",
            "emoji": "🚇",
            "tagline": "Constantly on the move between campus and home.",
            "description": f"Transit accounts for {transit_share:.1f}% of your budget. Travel and transit cards take a noticeable bite from your wallet.",
            "strength": "Efficient time management on the go.",
            "watch_out": "Single-ride fares—look into university-subsidized semester transit passes."
        }
    elif books_share >= 18:
        return {
            "name": "The Academic Achiever",
            "emoji": "📚",
            "tagline": "Invested in coursework, materials, and study kits.",
            "description": f"A significant portion ({books_share:.1f}%) of your expenses went into textbooks, software, or lab gear.",
            "strength": "Prioritizes education and academic resources.",
            "watch_out": "Buying brand-new textbooks instead of renting or using library reserves."
        }
    else:
        return {
            "name": "The Balanced Scholar",
            "emoji": "⚖️",
            "tagline": "Steady, well-distributed student lifestyle.",
            "description": "Your spending is distributed evenly across essentials with no single category dominating your budget.",
            "strength": "Versatile and well-rounded spending habits.",
            "watch_out": "Creeping subscription fees and unmonitored miscellaneous items."
        }


# -----------------------------------------------------------------------------
# 2. SPENDING LEAKAGE DIAGNOSTICS
# -----------------------------------------------------------------------------
def diagnose_spending_leaks(df_transactions, effective_income):
    """
    Scans recent transactions to detect hidden financial 'leaks'—small or recurring
    habits that quietly drain student budgets.
    """
    leaks = []
    if df_transactions.empty:
        return leaks

    expenses = df_transactions[df_transactions["type"] == "Expense"].copy()
    if expenses.empty:
        return leaks

    # Calculate active months to normalize numbers
    expenses["date_dt"] = pd.to_datetime(expenses["date"], errors="coerce")
    num_months = max(1, expenses["date_dt"].dt.to_period("M").nunique())

    # 1. Weekend Leakage Check (Friday to Sunday spending vs Weekdays)
    expenses["day_of_week"] = expenses["date_dt"].dt.day_name()
    weekend_days = ["Friday", "Saturday", "Sunday"]

    weekend_spend = expenses[expenses["day_of_week"].isin(weekend_days)]["amount"].sum() / num_months
    total_spend = expenses["amount"].sum() / num_months

    if total_spend > 0:
        weekend_pct = (weekend_spend / total_spend) * 100
        if weekend_pct >= 45:
            leaks.append({
                "type": "Weekend Drain",
                "emoji": "🍻",
                "severity": "High",
                "title": f"Weekend Spike: {weekend_pct:.1f}% of spending occurs Friday–Sunday",
                "detail": f"You spend an average of ${weekend_spend:,.2f}/month on weekends alone. Most student financial leaks happen during spontaneous weekend dining and nightlife.",
                "fix": "Set a fixed 'Weekend Fun Envelope' of cash or a separate debit card to cap weekend spend."
            })

    # 2. Frequent Micro-Purchases (< $15 transactions)
    micro_trans = expenses[expenses["amount"] <= 15.0]
    micro_sum = (micro_trans["amount"].sum() / num_months)
    micro_count = int(round(len(micro_trans) / num_months))

    if micro_count >= 3 and micro_sum >= 25.0:
        leaks.append({
            "type": "Micro-Purchases",
            "emoji": "☕",
            "severity": "Medium",
            "title": f"Micro-Drain: ~{micro_count} small purchases/mo total ${micro_sum:,.2f}/mo",
            "detail": f"Small $3–$12 purchases (coffee, convenience snacks, fast food) silently add up to ${micro_sum:,.2f} each month.",
            "fix": "Brew coffee at dorm/home 3 days a week and carry a refillable water bottle and snack pack."
        })

    # 3. High Discretionary Ratio
    wants_cats = ["Entertainment", "Other Expense"]
    wants_spend = (expenses[expenses["category"].isin(wants_cats)]["amount"].sum() / num_months)
    if effective_income > 0:
        wants_ratio = (wants_spend / effective_income) * 100
        if wants_ratio > 25:
            leaks.append({
                "type": "Discretionary Overload",
                "emoji": "🛍️",
                "severity": "High",
                "title": f"Discretionary Spending at {wants_ratio:.1f}% of Income",
                "detail": f"You spend ~${wants_spend:,.2f}/month on entertainment and non-essential items, exceeding the recommended student ceiling.",
                "fix": "Apply the 48-Hour Rule: Wait two full days before making any non-essential purchase over $25."
            })

    # 4. Single-Category Dominance (Non-housing)
    cat_totals = (expenses[expenses["category"] != "Rent & Housing"].groupby("category")["amount"].sum() / num_months)
    if not cat_totals.empty:
        top_cat = cat_totals.idxmax()
        top_amt = cat_totals.max()
        if effective_income > 0 and (top_amt / effective_income) >= 0.28:
            leaks.append({
                "type": "Heavy Category Concentration",
                "emoji": "⚠️",
                "severity": "Medium",
                "title": f"Heavy Outflow in '{top_cat}': ~${(top_amt):,.2f}/mo",
                "detail": f"{top_cat} consumes {(top_amt/effective_income)*100:.1f}% of your monthly income.",
                "fix": f"Target a modest 10% reduction in {top_cat} next month to free up ~${(top_amt * 0.10):,.2f}/month."
            })

    return leaks


# -----------------------------------------------------------------------------
# 3. INTERACTIVE 'WHAT-IF' BUDGET REBALANCING SIMULATOR
# -----------------------------------------------------------------------------
def simulate_rebalancing(df_transactions, effective_income, food_cut_pct, ent_cut_pct, extra_income=0.0):
    """
    Computes real-time mathematical impact when a student cuts discretionary spending
    or takes on extra side-income (e.g. tutoring, lab assistant hours).
    Normalized to monthly figures.
    """
    if df_transactions.empty:
        return {
            "old_surplus": effective_income,
            "new_surplus": effective_income + extra_income,
            "total_saved": 0.0,
            "food_savings": 0.0,
            "ent_savings": 0.0,
            "extra_income": extra_income,
            "old_savings_rate": 100.0 if effective_income > 0 else 0.0,
            "new_savings_rate": 100.0 if (effective_income + extra_income) > 0 else 0.0,
            "improvement": extra_income
        }

    df_temp = df_transactions.copy()
    df_temp["date_dt"] = pd.to_datetime(df_temp["date"], errors="coerce")
    num_months = max(1, df_temp["date_dt"].dt.to_period("M").nunique())

    expenses = df_temp[df_temp["type"] == "Expense"]
    total_expense = (float(expenses["amount"].sum()) / num_months) if not expenses.empty else 0.0

    # Categorize monthly averages
    cat_totals = (expenses.groupby("category")["amount"].sum() / num_months).to_dict() if not expenses.empty else {}
    food_spend = cat_totals.get("Food & Groceries", 0.0)
    ent_spend = cat_totals.get("Entertainment", 0.0)

    # Dollar savings from cuts
    food_savings = food_spend * (food_cut_pct / 100.0)
    ent_savings = ent_spend * (ent_cut_pct / 100.0)
    total_saved = food_savings + ent_savings

    # New budget state
    new_expense = max(0.0, total_expense - total_saved)
    new_income = effective_income + extra_income
    new_surplus = new_income - new_expense
    new_savings_rate = (new_surplus / new_income * 100) if new_income > 0 else 0.0

    # Old baseline
    old_surplus = effective_income - total_expense
    old_savings_rate = (old_surplus / effective_income * 100) if effective_income > 0 else 0.0

    improvement = new_surplus - old_surplus

    return {
        "old_surplus": old_surplus,
        "new_surplus": new_surplus,
        "total_saved": total_saved,
        "food_savings": food_savings,
        "ent_savings": ent_savings,
        "extra_income": extra_income,
        "old_savings_rate": old_savings_rate,
        "new_savings_rate": new_savings_rate,
        "improvement": improvement
    }


# -----------------------------------------------------------------------------
# 4. CONTEXT-AWARE STUDENT FINANCIAL Q&A ENGINE
# -----------------------------------------------------------------------------
def get_coaching_answer(question_key, financial_context):
    """
    Answers common student financial questions using the user's REAL live figures
    from the database, translating raw data into personalized, empowering advice.
    """
    inc = financial_context.get("income", 1250.0)
    exp = financial_context.get("expense", 900.0)
    surplus = financial_context.get("surplus", 350.0)
    rate = financial_context.get("savings_rate", 28.0)
    top_cat = financial_context.get("top_category", "Food & Groceries")
    top_amt = financial_context.get("top_amount", 180.0)
    persona = financial_context.get("persona", "The Balanced Scholar")

    answers = {
        "where_leaking": (
            f"🔍 **Leakage Analysis for your profile ({persona}):**\n\n"
            f"Based on your transactions, your largest outflow is **{top_cat}** at **${top_amt:,.2f}** "
            f"({(top_amt/exp*100):.1f}% of all expenses). Additionally, micro-transactions under $15 "
            f"(campus snacks, drinks) tend to drain student accounts by 10-15% silently.\n\n"
            f"💡 **Action Step**: Trimming just 10% from {top_cat} will instantly put **${(top_amt*0.10):,.2f}** "
            f"back in your pocket every month."
        ),
        "save_50_more": (
            f"💵 **How to Free Up $50.00 This Month:**\n\n"
            f"Your current monthly surplus is **${surplus:,.2f}** ({rate:.1f}% savings rate). "
            f"To boost this by +$50 without hurting your academic performance:\n\n"
            f"1. **Meal Prep Twice a Week**: Saves ~$30/month compared to buying campus cafeteria lunches.\n"
            f"2. **Audit Subscriptions**: Cancel or split one unused media/cloud service (saves ~$10–$15/month).\n"
            f"3. **Use Student Transit Passes**: Avoid peak single-ride rideshare trips for short distances (saves ~$15/month).\n\n"
            f"🎯 Doing this pushes your net savings from **${surplus:,.2f}** to **${(surplus+50):,.2f}**!"
        ),
        "emergency_fund": (
            f"🛡️ **Should You Build an Emergency Fund First?**\n\n"
            f"**Yes!** For a college student with your expense level (~${exp:,.2f}/mo), we recommend a starter **Emergency Cushion of $500 to $1,000**.\n\n"
            f"• At your current savings rate, putting aside **$100/month** gets you a fully funded $500 safety cushion in **5 months**.\n"
            f"• Keep this in a High-Yield Student Savings Account so it earns interest while protecting you from sudden laptop repairs or travel emergencies."
        ),
        "textbook_hack": (
            f"📚 **Cutting Academic & Textbook Costs:**\n\n"
            f"Textbooks are one of college's steepest hidden expenses. Here is the student strategy:\n\n"
            f"1. **University Library Reserve**: Campus libraries often keep physical and digital reserve copies of course textbooks for 2-hour checkouts.\n"
            f"2. **OpenStax & Open Educational Resources (OER)**: Search for peer-reviewed open textbook alternatives.\n"
            f"3. **Digital Rentals / Semester Rentals**: Renting via VitalSource or Chegg typically saves 60-70% over buying new.\n"
            f"4. **Department Senior Buy-Back**: Connect with student clubs in your major to buy used copies at half price."
        ),
        "credit_cards": (
            f"💳 **Student Credit Cards & Credit Scores:**\n\n"
            f"Building credit in college is a smart move if done carefully:\n\n"
            f"• **Golden Rule**: Only charge what you can pay off in full every single month. Never carry a balance to avoid high interest (~24-29% APR).\n"
            f"• **Keep Utilization Low**: Use less than 20% of your credit limit (e.g. if limit is $500, keep balance under $100).\n"
            f"• Put a single small recurring bill (like Spotify Student) on auto-pay and set auto-debit to pay the card in full automatically."
        )
    }

    return answers.get(question_key, "I am your AI Coach! Select any question above to get tailored advice based on your transactions.")


# -----------------------------------------------------------------------------
# 5. 7-DAY STUDENT FINANCIAL CHALLENGE
# -----------------------------------------------------------------------------
def generate_weekly_challenge(health_score):
    """
    Generates a realistic, engaging 7-day challenge tailored to the student's
    current Financial Health Score.
    """
    if health_score >= 80:
        return [
            {"day": "Day 1", "task": "Automate Savings", "desc": "Set up an automatic $15 transfer to your savings cushion on payday."},
            {"day": "Day 2", "task": "Student Perk Audit", "desc": "Check your university student portal for free software licenses (Microsoft 365, Notion, GitHub Student Pack)."},
            {"day": "Day 3", "task": "Hydration Swap", "desc": "Carry your refillable water bottle and thermos instead of buying bottled drinks."},
            {"day": "Day 4", "task": "Cook Together", "desc": "Host a communal meal prep session with dorm-mates or friends."},
            {"day": "Day 5", "task": "Library Treasure Hunt", "desc": "Explore university library streaming databases for free movies and documentaries."},
            {"day": "Day 6", "task": "No-Spend Saturday", "desc": "Spend a full weekend day doing zero-cost activities (campus sports, gaming, park walk)."},
            {"day": "Day 7", "task": "Weekly Review", "desc": "Log all your receipts in SmartSpend AI and celebrate your progress!"}
        ]
    else:
        return [
            {"day": "Day 1", "task": "Track Every Penny", "desc": "Log every single transaction today in SmartSpend AI—even a $1.50 snack."},
            {"day": "Day 2", "task": "The 48-Hour Pause", "desc": "Add any impulse item you want to an online wishlist instead of buying it."},
            {"day": "Day 3", "task": "Subscription Cleanse", "desc": "Review active app store subscriptions and cancel at least one unused service."},
            {"day": "Day 4", "task": "Campus Food Hunt", "desc": "Pack a sandwich, fruit, and snacks from your dorm/apartment before leaving for class."},
            {"day": "Day 5", "task": "Transit Audit", "desc": "Walk or bike for any trip under 1.5 miles instead of paying rideshare fees."},
            {"day": "Day 6", "task": "Free Fun Saturday", "desc": "Find free student events happening on campus or in your local community."},
            {"day": "Day 7", "task": "Check Health Score", "desc": "Return to the AI Savings Coach tab to watch your health score climb!"}
        ]
