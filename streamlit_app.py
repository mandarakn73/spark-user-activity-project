import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="User Activity Pattern Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a1128; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00B0FF; }
    .metric-label { font-size: 0.8rem; color: #90CAF9; text-transform: uppercase; letter-spacing: 1px; }
    .anomaly-card {
        background: #1a0a0a;
        border: 1px solid #E53935;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
    }
    .normal-card {
        background: #0a1a0a;
        border: 1px solid #00897B;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.4rem 0;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00B0FF;
        border-left: 4px solid #00B0FF;
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    div[data-testid="stSidebarContent"] { background: #0D1B40; }
    .sidebar-info {
        background: #1565C0;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.8rem;
        color: #E3F2FD;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load & process data ──────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ecommerce_user_activity.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

# Pre-compute all analytics
action_counts   = df["action"].value_counts().reset_index()
action_counts.columns = ["action", "count"]

hourly          = df.groupby("hour").size().reset_index(name="count")
day_activity    = df.groupby("day_of_week").size().reset_index(name="count")
category_counts = df.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
device_counts   = df.groupby("device").size().reset_index(name="count")

user_stats = df.groupby("user_id").agg(
    total_actions=("action", "count"),
    views=("action", lambda x: (x == "view").sum()),
    carts=("action", lambda x: (x == "cart").sum()),
    purchases=("action", lambda x: (x == "purchase").sum()),
    categories=("category", "nunique"),
    sessions=("session_id", "nunique"),
).reset_index()

user_stats["ratio"] = user_stats.apply(
    lambda r: r["views"] if r["purchases"] == 0 else round(r["views"] / r["purchases"], 1), axis=1
)
user_stats["flag"] = user_stats.apply(
    lambda r: "HIGH RISK" if (r["views"] >= 5 and r["purchases"] == 0)
    else ("MODERATE" if r["ratio"] > 8 and r["purchases"] > 0 else "NORMAL"), axis=1
)

funnel_data = pd.DataFrame({
    "stage": ["Viewed", "Added to Cart", "Purchased"],
    "users": [
        (user_stats["views"] > 0).sum(),
        (user_stats["carts"] > 0).sum(),
        (user_stats["purchases"] > 0).sum(),
    ]
})

total_users    = df["user_id"].nunique()
total_events   = len(df)
total_sessions = df["session_id"].nunique()
anomaly_count  = (user_stats["flag"] != "NORMAL").sum()
peak_hour      = hourly.loc[hourly["count"].idxmax(), "hour"]
top_action     = action_counts.iloc[0]["action"]


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Spark Dashboard")
    st.markdown("**User Activity Pattern Detection**")
    st.markdown("---")

    st.markdown("### Filters")
    selected_actions = st.multiselect(
        "Filter by Action",
        options=df["action"].unique().tolist(),
        default=df["action"].unique().tolist()
    )
    selected_categories = st.multiselect(
        "Filter by Category",
        options=df["category"].unique().tolist(),
        default=df["category"].unique().tolist()
    )
    selected_devices = st.multiselect(
        "Filter by Device",
        options=df["device"].unique().tolist(),
        default=df["device"].unique().tolist()
    )

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <b>Project Info</b><br>
    Course: B.Tech Big Data Analytics<br>
    Technology: Apache Spark + PySpark<br>
    Dataset: 103 records, 20 users<br>
    Academic Year: 2024–25
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Raw Data Preview")
    filtered_df = df[
        df["action"].isin(selected_actions) &
        df["category"].isin(selected_categories) &
        df["device"].isin(selected_devices)
    ]
    st.dataframe(filtered_df[["user_id","action","category","hour","device"]].head(10), use_container_width=True)


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#00B0FF;margin-bottom:0;font-size:1.8rem'>
⚡ User Activity Pattern Detection
</h1>
<p style='color:#90CAF9;margin-top:0.2rem;font-size:0.95rem'>
Apache Spark · PySpark · E-Commerce Behavioral Analytics · B.Tech Big Data Project
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI Metric Cards ─────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
metrics = [
    (c1, str(total_events),    "Total Events"),
    (c2, str(total_users),     "Unique Users"),
    (c3, str(total_sessions),  "Sessions"),
    (c4, f"{peak_hour}:00",    "Peak Hour"),
    (c5, str(anomaly_count),   "Anomalies Found"),
    (c6, f"{round(anomaly_count/total_users*100)}%", "Anomaly Rate"),
]
for col, val, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analysis", "🔍 Pattern Detection", "🚨 Anomaly Detection", "📋 Raw Data"
])


# ══════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ══════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Most Common User Actions</div>', unsafe_allow_html=True)
        fig = px.bar(
            action_counts, x="action", y="count",
            color="action",
            color_discrete_map={"view": "#1565C0", "cart": "#FF6F00", "purchase": "#00897B"},
            text="count"
        )
        fig.update_traces(textposition="outside", textfont_size=14)
        fig.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", showlegend=False,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Action Distribution</div>', unsafe_allow_html=True)
        fig2 = px.pie(
            action_counts, names="action", values="count",
            color="action",
            color_discrete_map={"view": "#1565C0", "cart": "#FF6F00", "purchase": "#00897B"},
            hole=0.5
        )
        fig2.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Peak Activity by Hour</div>', unsafe_allow_html=True)
        fig3 = px.area(
            hourly, x="hour", y="count",
            color_discrete_sequence=["#00B0FF"]
        )
        fig3.update_traces(fill="tozeroy", line_color="#00B0FF")
        fig3.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white",
            xaxis=dict(showgrid=False, title="Hour of Day"),
            yaxis=dict(showgrid=False, title="Events"),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Category Popularity</div>', unsafe_allow_html=True)
        fig4 = px.bar(
            category_counts, x="count", y="category",
            orientation="h",
            color="count",
            color_continuous_scale=["#0D47A1", "#00B0FF"]
        )
        fig4.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", showlegend=False,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            coloraxis_showscale=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="section-title">Activity by Day of Week</div>', unsafe_allow_html=True)
        fig5 = px.bar(
            day_activity, x="day_of_week", y="count",
            color_discrete_sequence=["#7C4DFF"]
        )
        fig5.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown('<div class="section-title">Device Usage</div>', unsafe_allow_html=True)
        fig6 = px.pie(
            device_counts, names="device", values="count",
            color_discrete_sequence=["#00B0FF", "#FF6F00", "#00897B"],
            hole=0.4
        )
        fig6.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════
# TAB 2 — PATTERN DETECTION
# ══════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Purchase Funnel</div>', unsafe_allow_html=True)
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data["stage"],
            x=funnel_data["users"],
            textinfo="value+percent initial",
            marker=dict(color=["#1565C0", "#FF6F00", "#00897B"]),
        ))
        fig_funnel.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Conversion metrics
        view_users = int(funnel_data[funnel_data["stage"] == "Viewed"]["users"])
        cart_users = int(funnel_data[funnel_data["stage"] == "Added to Cart"]["users"])
        purch_users = int(funnel_data[funnel_data["stage"] == "Purchased"]["users"])

        m1, m2 = st.columns(2)
        m1.metric("View → Cart Rate",     f"{round(cart_users/view_users*100)}%")
        m2.metric("Cart → Purchase Rate", f"{round(purch_users/cart_users*100)}%")

    with col2:
        st.markdown('<div class="section-title">User Funnel Stage Breakdown</div>', unsafe_allow_html=True)

        funnel_classify = user_stats.copy()
        funnel_classify["stage"] = funnel_classify.apply(
            lambda r: "Complete (View→Cart→Purchase)" if r["purchases"] > 0
            else ("Partial (View→Cart)" if r["carts"] > 0 else "Browse Only"), axis=1
        )
        stage_counts = funnel_classify["stage"].value_counts().reset_index()
        stage_counts.columns = ["stage", "count"]

        fig_stage = px.bar(
            stage_counts, x="stage", y="count",
            color="stage",
            color_discrete_map={
                "Complete (View→Cart→Purchase)": "#00897B",
                "Partial (View→Cart)": "#FF6F00",
                "Browse Only": "#E53935"
            },
            text="count"
        )
        fig_stage.update_traces(textposition="outside")
        fig_stage.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", showlegend=False,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_stage, use_container_width=True)

        st.markdown('<div class="section-title">Session Action Sequences</div>', unsafe_allow_html=True)
        sequences = df.sort_values("time").groupby(["session_id","user_id"])["action"].apply(
            lambda x: " → ".join(x)
        ).reset_index()
        sequences.columns = ["Session", "User", "Action Sequence"]
        st.dataframe(sequences, use_container_width=True, height=220)

    st.markdown('<div class="section-title">Per-User Behaviour Breakdown</div>', unsafe_allow_html=True)
    fig_user = px.bar(
        user_stats.sort_values("total_actions", ascending=False),
        x="user_id", y=["views", "carts", "purchases"],
        barmode="stack",
        color_discrete_map={"views": "#1565C0", "carts": "#FF6F00", "purchases": "#00897B"},
        labels={"value": "Count", "variable": "Action"}
    )
    fig_user.update_layout(
        plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
        font_color="white",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_user, use_container_width=True)


# ══════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Anomaly Flag Distribution</div>', unsafe_allow_html=True)
        flag_counts = user_stats["flag"].value_counts().reset_index()
        flag_counts.columns = ["flag", "count"]
        fig_flag = px.pie(
            flag_counts, names="flag", values="count",
            color="flag",
            color_discrete_map={"NORMAL": "#00897B", "HIGH RISK": "#E53935", "MODERATE": "#FF6F00"},
            hole=0.5
        )
        fig_flag.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white", margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_flag, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Views vs Purchases (Scatter)</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            user_stats, x="views", y="purchases",
            color="flag", size="total_actions",
            color_discrete_map={"NORMAL": "#00897B", "HIGH RISK": "#E53935", "MODERATE": "#FF6F00"},
            hover_data=["user_id"],
            labels={"views": "Total Views", "purchases": "Total Purchases"}
        )
        fig_scatter.update_layout(
            plot_bgcolor="#0D1B40", paper_bgcolor="#0D1B40",
            font_color="white",
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-title">All Users — Anomaly Report</div>', unsafe_allow_html=True)

    for _, row in user_stats.sort_values("flag").iterrows():
        card_class = "anomaly-card" if row["flag"] != "NORMAL" else "normal-card"
        flag_color = "#E53935" if row["flag"] == "HIGH RISK" else ("#FF6F00" if row["flag"] == "MODERATE" else "#00897B")
        icon = "🔴" if row["flag"] == "HIGH RISK" else ("🟡" if row["flag"] == "MODERATE" else "🟢")
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:600;color:white;font-size:0.95rem">{icon} {row['user_id']}</span>
                <span style="background:{flag_color};color:white;padding:2px 12px;border-radius:20px;font-size:0.75rem;font-weight:600">{row['flag']}</span>
            </div>
            <div style="display:flex;gap:2rem;margin-top:0.5rem;font-size:0.82rem;color:#90CAF9">
                <span>👁 Views: <b style="color:white">{int(row['views'])}</b></span>
                <span>🛒 Cart: <b style="color:white">{int(row['carts'])}</b></span>
                <span>✅ Purchases: <b style="color:white">{int(row['purchases'])}</b></span>
                <span>📊 Ratio: <b style="color:white">{row['ratio']}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Full Dataset</div>', unsafe_allow_html=True)

    search = st.text_input("Search by User ID (e.g. U003)", "")
    display_df = filtered_df.copy()
    if search:
        display_df = display_df[display_df["user_id"].str.contains(search.upper())]

    st.dataframe(display_df, use_container_width=True, height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Download Filtered Data as CSV",
            data=display_df.to_csv(index=False),
            file_name="filtered_activity.csv",
            mime="text/csv"
        )
    with col2:
        st.markdown(f"**Showing:** {len(display_df)} of {len(df)} records")

    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").fillna("—"), use_container_width=True)