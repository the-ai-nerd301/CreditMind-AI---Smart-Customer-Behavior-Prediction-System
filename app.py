import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="CreditMind AI: Smart Customer Behavior Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================
# CUSTOM CSS / DESIGN SYSTEM
# =============================

st.markdown(
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

:root{
    --neon-cyan:#00f5ff;
    --neon-purple:#a855f7;
    --neon-pink:#ec4899;
    --neon-green:#10b981;
    --neon-red:#ef4444;
    --neon-amber:#f59e0b;
    --glass:rgba(255,255,255,0.06);
    --glass-border:rgba(0,245,255,0.18);
}

html, body, [class*="css"]{
    font-family:'Rajdhani', sans-serif;
}

.stApp{
    background:radial-gradient(ellipse at top left, #0b1120 0%, #050505 55%),
               linear-gradient(135deg,#050505,#0a0e1a,#020617);
    color:#e2e8f0;
}

/* ambient background fx */
.bg-grid{
    position:fixed; inset:0; z-index:0; pointer-events:none;
    background-image:
        linear-gradient(rgba(0,245,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.05) 1px, transparent 1px);
    background-size:42px 42px;
    animation:gridDrift 24s linear infinite;
    mask-image:radial-gradient(ellipse at center, black 40%, transparent 85%);
}
@keyframes gridDrift{
    0%{background-position:0 0;}
    100%{background-position:42px 42px;}
}
.blob{position:fixed; border-radius:50%; filter:blur(100px); opacity:0.22; z-index:0; pointer-events:none;}
.blob1{width:520px; height:520px; background:#a855f7; top:-140px; left:-140px; animation:float1 19s ease-in-out infinite;}
.blob2{width:460px; height:460px; background:#00f5ff; bottom:-140px; right:-100px; animation:float2 23s ease-in-out infinite;}
@keyframes float1{0%,100%{transform:translate(0,0);} 50%{transform:translate(60px,40px);}}
@keyframes float2{0%,100%{transform:translate(0,0);} 50%{transform:translate(-50px,-30px);}}

section.main > div.block-container{ position:relative; z-index:1; }

/* title */
.title{
    font-family:'Orbitron', sans-serif;
    font-size:52px;
    font-weight:900;
    text-align:center;
    letter-spacing:2px;
    background:linear-gradient(90deg,#00f5ff,#a855f7,#ec4899,#00f5ff);
    background-size:300% auto;
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    animation:shimmer 6s linear infinite, fadeInUp 0.8s ease;
    margin-bottom:0;
}
@keyframes shimmer{ to{ background-position:300% center; } }
@keyframes fadeInUp{
    from{ opacity:0; transform:translateY(14px); }
    to{ opacity:1; transform:translateY(0); }
}
.subtitle{
    text-align:center;
    color:#94a3b8;
    font-size:17px;
    font-weight:500;
    letter-spacing:0.5px;
    margin-top:2px;
    animation:fadeInUp 1s ease;
}

/* HUD status chips */
.hud-row{ display:flex; justify-content:center; gap:14px; flex-wrap:wrap; margin:18px 0 6px 0; animation:fadeInUp 1.1s ease; }
.hud-chip{
    font-family:'Share Tech Mono', monospace;
    font-size:12.5px;
    letter-spacing:1px;
    color:#a5f3fc;
    background:rgba(0,245,255,0.06);
    border:1px solid rgba(0,245,255,0.25);
    border-radius:20px;
    padding:6px 16px;
    display:flex; align-items:center; gap:8px;
}
.dot{ width:8px; height:8px; border-radius:50%; background:var(--neon-green); box-shadow:0 0 8px var(--neon-green); animation:pulseDot 1.6s ease-in-out infinite; }
@keyframes pulseDot{ 0%,100%{ opacity:1; } 50%{ opacity:0.35; } }

/* glass cards */
.card{
    background:var(--glass);
    padding:22px;
    border-radius:18px;
    backdrop-filter:blur(16px);
    border:1px solid var(--glass-border);
    box-shadow:0 0 22px rgba(0,245,255,0.12);
    margin:8px 0;
    transition:transform 0.25s ease, box-shadow 0.25s ease;
    animation:fadeInUp 0.6s ease;
}
.card:hover{ transform:translateY(-4px); box-shadow:0 0 32px rgba(0,245,255,0.28); }
.card.purple{ border-color:rgba(168,85,247,0.35); box-shadow:0 0 22px rgba(168,85,247,0.15); }
.card.purple:hover{ box-shadow:0 0 34px rgba(168,85,247,0.32); }
.card.hero{
    border-color:rgba(236,72,153,0.4);
    box-shadow:0 0 30px rgba(236,72,153,0.2);
    text-align:center;
}
.card.hero:hover{ box-shadow:0 0 42px rgba(236,72,153,0.38); }

.metric-title{ font-family:'Rajdhani'; font-size:15px; font-weight:600; letter-spacing:1px; color:#38bdf8; text-transform:uppercase; margin-bottom:6px; }
.metric-value{ font-family:'Share Tech Mono', monospace; font-size:32px; font-weight:700; color:#f8fafc; text-shadow:0 0 14px rgba(0,245,255,0.5); }
.metric-value.big{ font-size:42px; }
.metric-sub{ font-size:12.5px; color:#94a3b8; margin-top:4px; }

/* risk badge */
.badge{
    display:inline-flex; align-items:center; gap:8px;
    padding:8px 18px; border-radius:20px; font-weight:700;
    font-family:'Orbitron'; font-size:13px; letter-spacing:1px;
    animation:fadeInUp 0.6s ease;
}
.badge.risk{ background:rgba(239,68,68,0.12); color:#fca5a5; border:1px solid rgba(239,68,68,0.4); box-shadow:0 0 18px rgba(239,68,68,0.25); }
.badge.safe{ background:rgba(16,185,129,0.12); color:#6ee7b7; border:1px solid rgba(16,185,129,0.4); box-shadow:0 0 18px rgba(16,185,129,0.25); }
.badge .dot{ width:9px; height:9px; }
.badge.risk .dot{ background:var(--neon-red); box-shadow:0 0 8px var(--neon-red); }
.badge.safe .dot{ background:var(--neon-green); box-shadow:0 0 8px var(--neon-green); }

/* empty state hud */
.empty-hud{
    text-align:center; padding:50px 20px; border-radius:18px;
    border:1px dashed rgba(0,245,255,0.28); background:rgba(255,255,255,0.02);
    color:#94a3b8; font-family:'Rajdhani'; font-size:16px;
}
.empty-hud .icon{ font-size:38px; margin-bottom:12px; }

/* buttons */
div.stButton > button{
    background:linear-gradient(90deg,#06b6d4,#9333ea);
    color:white; border-radius:14px; padding:14px 34px; border:none;
    font-family:'Orbitron'; font-weight:700; letter-spacing:1.5px; font-size:14px;
    box-shadow:0 0 18px rgba(147,51,234,0.45);
    position:relative; overflow:hidden;
    transition:transform 0.2s ease, box-shadow 0.2s ease;
}
div.stButton > button:hover{ transform:translateY(-2px) scale(1.015); box-shadow:0 0 34px rgba(6,182,212,0.6); }
div.stButton > button:active{ transform:translateY(0) scale(0.99); }
div.stButton > button::after{
    content:""; position:absolute; top:0; left:-60%; width:40%; height:100%;
    background:linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
    transform:skewX(-20deg);
    transition:left 0.6s ease;
}
div.stButton > button:hover::after{ left:130%; }

/* sidebar */
section[data-testid="stSidebar"]{
    background:rgba(8,10,20,0.75);
    backdrop-filter:blur(18px);
    border-right:1px solid rgba(0,245,255,0.15);
}
.sidebar-brand{ font-family:'Orbitron'; font-weight:700; font-size:20px; color:#67e8f9; text-shadow:0 0 12px rgba(0,245,255,0.5); letter-spacing:1px; }
.sidebar-caption{ color:#64748b; font-size:12.5px; margin-bottom:14px; }
.sidebar-section{
    font-family:'Orbitron'; font-size:12px; letter-spacing:1.5px; color:#c4b5fd;
    border-left:3px solid var(--neon-purple); padding:5px 10px; margin:18px 0 8px 0;
    background:rgba(168,85,247,0.06); border-radius:0 6px 6px 0;
}
section[data-testid="stSidebar"] label{ color:#94a3b8 !important; font-weight:600; font-size:13.5px; }

/* tabs */
button[data-baseweb="tab"]{ font-family:'Orbitron'; font-size:13px; letter-spacing:1px; color:#94a3b8; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#67e8f9 !important; }
div[data-baseweb="tab-highlight"]{ background-color:var(--neon-cyan) !important; box-shadow:0 0 10px var(--neon-cyan); }

/* neon table */
.neon-table{ width:100%; border-collapse:collapse; font-family:'Rajdhani'; font-size:15px; margin:10px 0; }
.neon-table th{
    font-family:'Orbitron'; font-size:11.5px; letter-spacing:1px; text-transform:uppercase;
    color:#67e8f9; background:rgba(0,245,255,0.07); padding:10px 14px; text-align:left;
    border-bottom:1px solid rgba(0,245,255,0.25);
}
.neon-table td{ padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.06); color:#e2e8f0; }
.neon-table tr:hover td{ background:rgba(0,245,255,0.04); }

/* divider */
.neon-hr{ border:none; height:1px; margin:22px 0; background:linear-gradient(90deg, transparent, rgba(0,245,255,0.4), transparent); }

/* footer */
.footer-note{ text-align:center; color:#475569; font-size:12px; margin-top:36px; letter-spacing:0.5px; }

/* scrollbar */
::-webkit-scrollbar{ width:8px; }
::-webkit-scrollbar-track{ background:#0a0a0a; }
::-webkit-scrollbar-thumb{ background:linear-gradient(180deg,#06b6d4,#9333ea); border-radius:8px; }

footer{ visibility:hidden; }

</style>

<div class="bg-grid"></div>
<div class="blob blob1"></div>
<div class="blob blob2"></div>
""",
unsafe_allow_html=True
)


# =============================
# LOAD MODELS
# =============================

linear_model = joblib.load("credit_limit_prediction_LR_model.pkl")
ridge_model = joblib.load("credit_limit_prediction_Ridge_model.pkl")
lasso_model = joblib.load("credit_limit_prediction_lasso_model.pkl")
churn_model = joblib.load("customer_churn_LogReg_model.pkl")


# =============================
# HERO
# =============================

st.markdown("<h1 class='title'>💳 CreditMind AI: Smart Customer Behavior Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-powered banking analytics · credit limit &amp; churn inference engine</div>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="hud-row">
        <div class="hud-chip"><span class="dot"></span>ENGINE ONLINE</div>
        <div class="hud-chip">🧠 4 MODELS LOADED</div>
        <div class="hud-chip">⚡ REAL-TIME INFERENCE</div>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================
# SIDEBAR INPUT
# =============================

st.sidebar.markdown("<div class='sidebar-brand'>⚙ CUSTOMER CONSOLE</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-caption'>Configure the customer profile, then run inference.</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-section'>👤 DEMOGRAPHICS</div>", unsafe_allow_html=True)

Customer_Age = st.sidebar.slider("Age", 18, 90, 40)
Gender = st.sidebar.selectbox("Gender", ["M", "F"])
Dependent_count = st.sidebar.number_input("Dependents", 0, 10, 2)
Education_Level = st.sidebar.selectbox(
    "Education",
    ["Graduate", "High School", "Unknown", "Uneducated", "College", "Post-Graduate", "Doctorate"]
)
Marital_Status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unknown"])
Income_Category = st.sidebar.selectbox(
    "Income Category",
    ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
)

st.sidebar.markdown("<div class='sidebar-section'>💳 ACCOUNT PROFILE</div>", unsafe_allow_html=True)

Card_Category = st.sidebar.selectbox("Card Type", ["Blue", "Silver", "Gold", "Platinum"])
Months_on_book = st.sidebar.slider("Months With Bank", 1, 70, 36)
Total_Relationship_Count = st.sidebar.slider("Products Count", 1, 10, 3)
Months_Inactive_12_mon = st.sidebar.slider("Inactive Months", 0, 12, 2)
Contacts_Count_12_mon = st.sidebar.slider("Bank Contacts", 0, 10, 2)

st.sidebar.markdown("<div class='sidebar-section'>💰 FINANCIAL BEHAVIOR</div>", unsafe_allow_html=True)

Credit_Limit = st.sidebar.number_input("Current Credit Limit ($)", min_value=0.0, value=5000.0, step=100.0)
Total_Revolving_Bal = st.sidebar.number_input("Revolving Balance ($)", min_value=0, value=1000)
Avg_Open_To_Buy = st.sidebar.number_input("Open To Buy ($)", min_value=0, value=4000)
Total_Amt_Chng_Q4_Q1 = st.sidebar.number_input("Amount Change Ratio", min_value=0.0, value=1.0, step=0.05)
Total_Trans_Amt = st.sidebar.number_input("Transaction Amount ($)", min_value=0, value=4000)
Total_Trans_Ct = st.sidebar.number_input("Transaction Count", min_value=0, value=60)
Total_Ct_Chng_Q4_Q1 = st.sidebar.number_input("Transaction Change Ratio", min_value=0.0, value=0.7, step=0.05)
Avg_Utilization_Ratio = st.sidebar.slider("Utilization Ratio", 0.0, 1.0, 0.3)


# =============================
# CREATE DATAFRAME
# =============================

input_data = pd.DataFrame({
    "Customer_Age": [Customer_Age],
    "Gender": [Gender],
    "Dependent_count": [Dependent_count],
    "Education_Level": [Education_Level],
    "Marital_Status": [Marital_Status],
    "Income_Category": [Income_Category],
    "Card_Category": [Card_Category],
    "Months_on_book": [Months_on_book],
    "Total_Relationship_Count": [Total_Relationship_Count],
    "Months_Inactive_12_mon": [Months_Inactive_12_mon],
    "Contacts_Count_12_mon": [Contacts_Count_12_mon],
    "Credit_Limit": [Credit_Limit],
    "Total_Revolving_Bal": [Total_Revolving_Bal],
    "Avg_Open_To_Buy": [Avg_Open_To_Buy],
    "Total_Amt_Chng_Q4_Q1": [Total_Amt_Chng_Q4_Q1],
    "Total_Trans_Amt": [Total_Trans_Amt],
    "Total_Trans_Ct": [Total_Trans_Ct],
    "Total_Ct_Chng_Q4_Q1": [Total_Ct_Chng_Q4_Q1],
    "Avg_Utilization_Ratio": [Avg_Utilization_Ratio]
})


# =============================
# GAUGE HELPER
# =============================

def churn_gauge(probability):
    pct = probability * 100
    if pct < 30:
        bar_color = "#10b981"
    elif pct < 60:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"color": "#f8fafc", "family": "Share Tech Mono", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": "rgba(0,245,255,0.25)",
            "steps": [
                {"range": [0, 30], "color": "rgba(16,185,129,0.15)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.15)"}
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": pct}
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=260,
        font={"color": "#e2e8f0"}
    )
    return fig


# =============================
# MAIN TABS
# =============================

tab_predict, tab_performance = st.tabs(["🎯 PREDICTION ENGINE", "📊 MODEL PERFORMANCE"])

with tab_predict:

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        run = st.button("🚀 RUN AI PREDICTION", use_container_width=True)

    if run:

        # --- credit limit predictions ---
        drop_cols = ["Credit_Limit", "Avg_Open_To_Buy", "Avg_Utilization_Ratio"]
        lr_pred = linear_model.predict(input_data.drop(columns=drop_cols))[0]
        ridge_pred = ridge_model.predict(input_data.drop(columns=drop_cols))[0]
        lasso_pred = lasso_model.predict(input_data.drop(columns=drop_cols))[0]
        ensemble = (lr_pred + ridge_pred + lasso_pred) / 3
        spread = max(lr_pred, ridge_pred, lasso_pred) - min(lr_pred, ridge_pred, lasso_pred)

        st.markdown("### 💰 Credit Limit Prediction")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Linear Regression</div>
                <div class="metric-value">${lr_pred:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Ridge Regression</div>
                <div class="metric-value">${ridge_pred:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="metric-title">Lasso Regression</div>
                <div class="metric-value">${lasso_pred:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="card hero">
                <div class="metric-title">Ensemble Average</div>
                <div class="metric-value big">${ensemble:,.0f}</div>
                <div class="metric-sub">Model spread: ${spread:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='neon-hr'>", unsafe_allow_html=True)

        # --- churn prediction ---
        churn = churn_model.predict(input_data)[0]
        probability = churn_model.predict_proba(input_data)[0][1]

        st.markdown("### 🔮 Customer Churn Prediction")
        g1, g2 = st.columns([1, 1])

        with g1:
            st.plotly_chart(churn_gauge(probability), use_container_width=True, config={"displayModeBar": False})

        with g2:
            if churn == 1:
                badge_html = "<div class='badge risk'><span class='dot'></span>HIGH RISK CUSTOMER</div>"
                if probability >= 0.6:
                    advice = "Elevated churn signal — recommend prioritizing retention outreach."
                else:
                    advice = "Moderate churn signal — monitor engagement and consider a check-in."
            else:
                badge_html = "<div class='badge safe'><span class='dot'></span>LOYAL CUSTOMER</div>"
                advice = "Low churn signal — standard engagement is sufficient."

            st.markdown(f"""
            <div class="card purple" style="margin-top:40px;">
                {badge_html}
                <div class="metric-sub" style="margin-top:14px; font-size:14px;">
                    Churn probability: <b style="color:#f8fafc;">{probability:.1%}</b>
                </div>
                <div class="metric-sub" style="margin-top:10px;">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="empty-hud">
                <div class="icon">🛰️</div>
                Configure the customer profile in the sidebar, then hit
                <b>RUN AI PREDICTION</b> to generate credit limit and churn forecasts.
            </div>
            """,
            unsafe_allow_html=True
        )


with tab_performance:

    st.markdown("### 📈 Regression Models")
    st.markdown(
        """
        <table class="neon-table">
            <tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th></tr>
            <tr><td>Linear Regression</td><td>0.56</td><td>6029</td><td>4121</td></tr>
            <tr><td>Ridge Regression</td><td>0.56</td><td>6029</td><td>4121</td></tr>
            <tr><td>Lasso Regression</td><td>0.56</td><td>6029</td><td>4121</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🧮 Classification — Churn Model")
    st.markdown(
        """
        <table class="neon-table">
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>90.0%</td></tr>
            <tr><td>Precision</td><td>76.8%</td></tr>
            <tr><td>Recall</td><td>53.8%</td></tr>
            <tr><td>F1 Score</td><td>63.3%</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    perf_fig = go.Figure(go.Bar(
        x=[90.0, 76.8, 53.8, 63.3],
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        orientation="h",
        marker=dict(color=["#00f5ff", "#a855f7", "#ec4899", "#10b981"]),
        text=["90.0%", "76.8%", "53.8%", "63.3%"],
        textposition="outside"
    ))
    perf_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0", "family": "Rajdhani"},
        xaxis=dict(range=[0, 100], showgrid=False, title="Score (%)"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=30, t=10, b=10),
        height=260
    )
    st.plotly_chart(perf_fig, use_container_width=True, config={"displayModeBar": False})


st.markdown(
    "<div class='footer-note'>CreditAI Intelligence · scikit-learn models served via Streamlit</div>",
    unsafe_allow_html=True
)