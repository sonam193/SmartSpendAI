# 🎓 SmartSpend AI - Student Personal Finance & ML Forecaster

SmartSpend AI is an interactive personal finance web application built with Python, Streamlit, SQLite, and Scikit-learn. Designed specifically for college students, it combines budgeting and expense tracking with **Machine Learning Forecasting (Stage 2)** and an **Intelligent AI Savings Coach (Stage 3)** to diagnose spending leaks, simulate habit changes, and promote lifelong financial discipline.

---

## 🎨 Color Palette & Design
- **Dark Navy (`#0F172A`, `#1E293B`)**: Headers, primary text, and high-contrast structural containers.
- **Warm Cream (`#FAF9F6`, `#FFFFFF`)**: Dashboard background and crisp card surfaces.
- **Emerald Green (`#10B981`, `#059669`)**: Forecast curves, positive balance metrics, and primary buttons.

---

## 🚀 Features Overview

### 📊 Stage 1: Core Dashboard & Database
1. **Live Monthly Overview Cards**: Income, Expenses, Remaining Balance, and Savings Rate.
2. **Transaction Entry Form**: Categorized student entries with real-time validation and error handling.
3. **Local SQLite Persistence (`smartspend.db`)**: Stores transactions locally across sessions.
4. **Interactive Charts**: Donut chart and horizontal bar chart for category spending.
5. **Search & Filter**: Search transactions by keyword, filter by type, or delete records.

### 🤖 Stage 2: Machine Learning Expense Forecasting
1. **Time-Series Linear Regression Model**:
   - Converts transaction dates into sequential time steps.
   - Learns historical spending velocity using Ordinary Least Squares ($y = mx + c$).
   - Forecasts total monthly expenses 1 to 6 months into the future.
2. **Model Evaluation Metrics Explained**:
   - **$R^2$ Score (Coefficient of Determination)**: Measures how well the trendline fits past spending ($0.0 \rightarrow 1.0$).
   - **Mean Absolute Error (MAE)**: Measures average dollar deviation from actual spending ($\pm \$XX.XX$).
   - **Trend Direction**: Flags whether spending is *Rising*, *Stable*, or *Decreasing*.
3. **Category-Level Expense Forecast**:
   - Predicts upcoming spending broken down by category (Food, Rent, Books, Transit, Entertainment).
4. **Budget Overrun Risk Engine**:
   - Compares predicted future expenses against expected income.
   - Categorizes risk as **Low Risk (Healthy Surplus)**, **Moderate Risk (Tight Margin)**, or **High Risk (Deficit Alert)**.

### 💡 Stage 3: AI Savings Coach (Personalized Budgeting)
1. **Student Financial Persona Detection**:
   - Analyzes category ratios to identify the student's profile (e.g. *The Campus Foodie*, *The Social Butterfly*, *The Campus Commuter*, *The Balanced Scholar*, *The Master Budgeter*).
   - Identifies strengths, spending habits, and specific risk areas without financial jargon.
2. **Student 50/30/20 Budget Benchmark**:
   - Calculates exact breakdown of **Needs** (Housing, Groceries, Transit: $\le 50\%$), **Wants** (Dining out, Entertainment: $\le 30\%$), and **Savings** ($\ge 20\%$).
3. **Financial Health Score (0–100)**:
   - Assigns a student credit-style score and letter grade (A+, A, B, C, Needs Attention).
4. **🔍 AI Spending Leakage Detector**:
   - Identifies hidden budget drains: **Weekend Spikes** (Friday–Sunday spending surges), **Micro-Purchase Clusters** ($<\$15$ convenience purchases), and category over-concentration.
   - Provides immediate, actionable campus-ready fixes.
5. **🎛️ Interactive 'What-If' Rebalancing Simulator**:
   - Sliders to test trimming food or entertainment by $5\%-50\%$ and adding part-time side income.
   - Calculates immediate real-time dollar impact, new surplus, and improved savings rate.
6. **💬 Context-Aware Student Financial Q&A**:
   - Answers common student questions (*Where is my money leaking?*, *How do I save \$50 more?*, *Emergency fund guide*, *Textbook savings hacks*) using the user's **actual live database numbers**!
7. **📅 7-Day Student Financial Challenge**:
   - Generates daily bite-sized micro-habits to build financial discipline without feeling restricted.
8. **🎯 Student Savings Goal Planner**:
   - Calculates monthly savings roadmaps for milestone goals (Laptop, Travel, Emergency Fund).

---

## 📁 File Structure

```text
SmartSpendAI/
│
├── .streamlit/
│   └── config.toml        # Dashboard theme configuration
├── app.py                 # Main Streamlit web application (Triple-Tab Dashboard)
├── ml_predictor.py        # Stage 2: Time-Series ML Regression & Projections
├── ai_coach.py            # Stage 3: Personalized Student AI Coach Engine
├── requirements.txt       # Dependencies (streamlit, pandas, plotly, scikit-learn)
├── smartspend.db          # Local SQLite database (persists transactions)
└── README.md              # Project documentation & beginner guide
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python -m streamlit run app.py
```
Open your browser at [http://localhost:8502](http://localhost:8502).

---

## 🧠 Machine Learning & AI Coach Concepts for Students

If you are presenting this project at an AIML hackathon, here is how the AI pipeline works:

1. **Supervised Regression ($y = mx + c$)**:
   - Maps sequential time steps ($x = 0, 1, 2, \dots$) to total monthly expenses ($y$).
   - Slope ($m$) quantifies spending acceleration or deceleration.
2. **Behavioral Pattern Recognition**:
   - Categorizes discretionary outflow by weekday vs. weekend distribution.
   - Flags small-ticket transaction density (micro-drains).
3. **Mathematical Budget Rebalancing**:
   - Simulates elasticity: $\Delta \text{Savings} = (\text{Food} \times r_f) + (\text{Entertainment} \times r_e) + \text{SideIncome}$.
   - Shows students exactly how small daily shifts compound into hundreds of dollars saved.
