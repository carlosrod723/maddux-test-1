# dashboard.py

"""
Interactive Dashboard

Streamlit + Plotly dashboard for exploring MADDUX hitter breakout predictions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

load_dotenv()

# Get API key from Streamlit secrets (cloud) or .env (local)
if 'ANTHROPIC_API_KEY' in st.secrets:
    os.environ['ANTHROPIC_API_KEY'] = st.secrets['ANTHROPIC_API_KEY']

# Configuration
DB_PATH = "maddux_db.db"
st.set_page_config(
    page_title="MADDUX Hitter Analytics",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Premium Dark Theme - Custom CSS
# ==============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Sans+Pro:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0e14;
        --bg-secondary: #12181f;
        --bg-tertiary: #1a2028;
        --bg-card: #151c24;
        --accent-green: #6ee7b7;
        --accent-coral: #ff6b6b;
        --accent-gold: #ffd93d;
        --accent-blue: #4dabf7;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a6;
        --text-muted: #5f6368;
        --border-subtle: rgba(255,255,255,0.08);
        --glow-green: 0 0 20px rgba(110,231,183,0.3);
        --glow-gold: 0 0 20px rgba(255,217,61,0.3);
    }

    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0d1117 50%, var(--bg-primary) 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }

    /* Sidebar header */
    .sidebar-header {
        padding: 0 0 1.5rem 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1.5rem;
    }

    .sidebar-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.5rem;
        letter-spacing: 3px;
        color: var(--text-primary);
        margin: 0;
    }

    .sidebar-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0.25rem 0 0 0;
    }

    /* Sidebar section labels */
    .sidebar-section {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* Sidebar inputs */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
    }


    /* Typography */
    h1, .stTitle {
        font-family: 'Bebas Neue', sans-serif !important;
        letter-spacing: 3px !important;
        color: var(--text-primary) !important;
        text-transform: uppercase;
    }

    h2, h3, .stSubheader {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: 0.5px;
    }

    p, .stMarkdown, .stText {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: var(--text-secondary) !important;
    }

    /* Hero header styling */
    .hero-container {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }


    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        letter-spacing: 6px;
        color: var(--text-primary);
        margin: 0;
        text-transform: uppercase;
    }

    .hero-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-weight: 300;
        letter-spacing: 1px;
    }

    .formula-badge {
        display: inline-block;
        background: var(--bg-primary);
        border: 1px solid var(--accent-green);
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        margin-top: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: var(--accent-green);
        box-shadow: var(--glow-green);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }

    /* Hide Streamlit's default tab indicator line */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Bebas Neue', sans-serif !important;
        color: var(--accent-green) !important;
        font-size: 2rem !important;
        letter-spacing: 2px;
    }

    /* DataFrames - Dark Theme */
    .stDataFrame,
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
    }

    .stDataFrame [data-testid="stDataFrameResizable"],
    [data-testid="stDataFrame"] > div {
        background: var(--bg-card) !important;
    }

    /* Glide Data Grid (Streamlit's default dataframe renderer) */
    [data-testid="stDataFrame"] canvas {
        background: var(--bg-card) !important;
    }

    [data-testid="stDataFrame"] [class*="glideDataEditor"] {
        background: var(--bg-card) !important;
    }

    /* Force dark theme on all dataframe containers */
    .stDataFrame > div,
    .stDataFrame > div > div,
    [data-testid="stDataFrame"] > div > div {
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
    }

    /* Table elements */
    .stDataFrame table,
    [data-testid="stDataFrame"] table {
        background: var(--bg-card) !important;
    }

    .stDataFrame thead tr th,
    [data-testid="stDataFrame"] thead tr th {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    .stDataFrame tbody tr td,
    [data-testid="stDataFrame"] tbody tr td {
        background: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    .stDataFrame tbody tr:hover td,
    [data-testid="stDataFrame"] tbody tr:hover td {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }

    .stDataFrame tbody tr th,
    [data-testid="stDataFrame"] tbody tr th {
        background: var(--bg-tertiary) !important;
        color: var(--text-muted) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    /* Scrollbar styling for tables */
    .stDataFrame ::-webkit-scrollbar,
    [data-testid="stDataFrame"] ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    .stDataFrame ::-webkit-scrollbar-track,
    [data-testid="stDataFrame"] ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    .stDataFrame ::-webkit-scrollbar-thumb,
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        border: 1px solid var(--accent-green);
        border-radius: 8px;
        padding: 0.75rem 2rem;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--accent-green);
        color: var(--bg-primary);
        border-color: var(--accent-green);
    }

    /* Select boxes */
    .stSelectbox [data-baseweb="select"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background: var(--bg-tertiary);
    }

    /* Section dividers */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 2rem 0;
    }

    /* Stat highlight cards */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .stat-card.positive {
        border-color: var(--accent-green);
        box-shadow: inset 0 0 20px rgba(110,231,183,0.1);
    }

    .stat-card.negative {
        border-color: var(--accent-coral);
        box-shadow: inset 0 0 20px rgba(255,107,107,0.1);
    }

    .stat-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.5rem;
        letter-spacing: 2px;
    }

    .stat-value.positive { color: var(--accent-green); }
    .stat-value.negative { color: var(--accent-coral); }
    .stat-value.gold { color: var(--accent-gold); }

    .stat-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Quadrant guide styling */
    .quadrant-guide {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.5rem;
    }

    .quadrant-item {
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--border-subtle);
    }

    .quadrant-item:last-child {
        border-bottom: none;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Plotly Theme Configuration
# ==============================================================================

PLOTLY_THEME = {
    'paper_bgcolor': 'rgba(21, 28, 36, 0)',
    'plot_bgcolor': 'rgba(21, 28, 36, 0.8)',
    'font': {
        'family': 'Source Sans Pro, sans-serif',
        'color': '#e8eaed',
        'size': 12
    },
    'title': {
        'font': {
            'family': 'Bebas Neue, sans-serif',
            'size': 24,
            'color': '#e8eaed'
        },
        'x': 0,
        'xanchor': 'left'
    },
    'xaxis': {
        'gridcolor': 'rgba(255,255,255,0.05)',
        'linecolor': 'rgba(255,255,255,0.1)',
        'tickfont': {'color': '#9aa0a6'},
        'title_font': {'color': '#9aa0a6', 'size': 11}
    },
    'yaxis': {
        'gridcolor': 'rgba(255,255,255,0.05)',
        'linecolor': 'rgba(255,255,255,0.1)',
        'tickfont': {'color': '#9aa0a6'},
        'title_font': {'color': '#9aa0a6', 'size': 11}
    },
    'colorway': ['#6ee7b7', '#ff6b6b', '#ffd93d', '#4dabf7', '#a855f7', '#f97316'],
    'hoverlabel': {
        'bgcolor': '#1a2028',
        'bordercolor': '#6ee7b7',
        'font': {'family': 'Source Sans Pro', 'color': '#e8eaed'}
    }
}

# Custom color scales
DIVERGING_SCALE = [
    [0, '#ff6b6b'],
    [0.25, '#ff8787'],
    [0.5, '#2a3441'],
    [0.75, '#86efac'],
    [1, '#6ee7b7']
]

SEQUENTIAL_GREEN = [
    [0, '#0d1117'],
    [0.5, '#134e3a'],
    [1, '#6ee7b7']
]

# Matching colormap for pandas styling
MINT_CMAP = LinearSegmentedColormap.from_list('mint', ['#0d1117', '#134e3a', '#6ee7b7'])


def apply_plotly_theme(fig):
    """Apply consistent dark theme to Plotly figures."""
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME['paper_bgcolor'],
        plot_bgcolor=PLOTLY_THEME['plot_bgcolor'],
        font=PLOTLY_THEME['font'],
        title=PLOTLY_THEME['title'],
        xaxis=PLOTLY_THEME['xaxis'],
        yaxis=PLOTLY_THEME['yaxis'],
        hoverlabel=PLOTLY_THEME['hoverlabel'],
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            bgcolor='rgba(21,28,36,0.8)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1,
            font={'color': '#e8eaed'}
        )
    )
    return fig


# ==============================================================================
# Database Functions
# ==============================================================================

@st.cache_data
def load_deltas():
    """Load player deltas from database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            p.player_name,
            pd.player_id,
            pd.year,
            pd.prev_year,
            pd.pa,
            pd.max_ev,
            pd.prev_max_ev,
            pd.delta_max_ev,
            pd.hard_hit_pct,
            pd.prev_hard_hit_pct,
            pd.delta_hard_hit_pct,
            pd.ops,
            pd.prev_ops,
            pd.delta_ops,
            pd.maddux_score,
            pd.team
        FROM player_deltas pd
        JOIN players p ON pd.player_id = p.player_id
    """, conn)
    conn.close()
    return df


@st.cache_data
def load_seasons():
    """Load all player seasons."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            p.player_name,
            ps.player_id,
            ps.year,
            ps.pa,
            ps.max_ev,
            ps.hard_hit_pct,
            ps.barrel_pct,
            ps.ops,
            ps.obp,
            ps.slg,
            ps.wrc_plus,
            ps.team
        FROM player_seasons ps
        JOIN players p ON ps.player_id = p.player_id
    """, conn)
    conn.close()
    return df


def get_player_history(player_name: str):
    """Get full history for a specific player."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            p.player_name,
            pd.player_id,
            pd.year,
            pd.prev_year,
            pd.pa,
            pd.max_ev,
            pd.prev_max_ev,
            pd.delta_max_ev,
            pd.hard_hit_pct,
            pd.prev_hard_hit_pct,
            pd.delta_hard_hit_pct,
            pd.ops,
            pd.prev_ops,
            pd.delta_ops,
            pd.maddux_score,
            pd.team
        FROM player_deltas pd
        JOIN players p ON pd.player_id = p.player_id
        WHERE p.player_name = ?
        ORDER BY pd.year
    """, conn, params=(player_name,))
    conn.close()
    return df


# ==============================================================================
# Claude AI Analysis
# ==============================================================================

def get_claude_analysis(player_data: dict) -> str:
    """Get Claude's analysis of a player's breakout potential."""
    try:
        client = Anthropic()
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

        prompt = f"""Analyze this MLB hitter's breakout potential based on their MADDUX metrics:

Player: {player_data['player_name']}
Year: {player_data['year']}
Team: {player_data['team']}

MADDUX Score: {player_data['maddux_score']:.2f}
- Δ Max Exit Velocity: {player_data['delta_max_ev']:+.1f} mph
- Δ Hard Hit %: {player_data['delta_hard_hit_pct']:+.1f}%

Current Stats:
- Max EV: {player_data['max_ev']:.1f} mph
- Hard Hit %: {player_data['hard_hit_pct']:.1f}%
- OPS: {player_data['ops']:.3f}
- Previous OPS: {player_data['prev_ops']:.3f}
- Δ OPS: {player_data['delta_ops']:+.3f}

The MADDUX model formula is: Score = Δ Max EV + (2.1 × Δ Hard Hit%)
Higher scores indicate improved physical metrics that often precede OPS gains.

Provide a brief (3-4 sentence) analysis of:
1. What the metrics suggest about this player's trajectory
2. Whether the MADDUX score aligns with their actual OPS change
3. Breakout/regression outlook"""

        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"


# ==============================================================================
# Dashboard Layout
# ==============================================================================

def main():
    # Hero Header
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">MADDUX Analytics</h1>
        <p class="hero-subtitle">Predicting MLB Breakouts Through Physical Metrics</p>
        <div class="formula-badge">MADDUX = Δ Max EV + (2.1 × Δ Hard Hit%)</div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df_deltas = load_deltas()
    df_seasons = load_seasons()

    # Sidebar filters
    with st.sidebar:
        # Sidebar header
        st.markdown("""
        <div class="sidebar-header">
            <h2 class="sidebar-title">MADDUX</h2>
            <p class="sidebar-subtitle">Hitter Analytics</p>
        </div>
        """, unsafe_allow_html=True)

        # Filter section
        st.markdown('<p class="sidebar-section">Filters</p>', unsafe_allow_html=True)

        # Year filter
        years = sorted(df_deltas['year'].unique(), reverse=True)
        selected_year = st.selectbox("Season", years, index=0)

        # Min PA filter
        min_pa = st.slider("Minimum PA", 50, 400, 200, step=50)

        # Score filter
        score_range = st.slider(
            "MADDUX Score Range",
            float(df_deltas['maddux_score'].min()),
            float(df_deltas['maddux_score'].max()),
            (float(df_deltas['maddux_score'].min()), float(df_deltas['maddux_score'].max()))
        )

        # Results section
        st.markdown('<p class="sidebar-section">Results</p>', unsafe_allow_html=True)

        # Filter data
        df_filtered = df_deltas[
            (df_deltas['year'] == selected_year) &
            (df_deltas['pa'] >= min_pa) &
            (df_deltas['maddux_score'] >= score_range[0]) &
            (df_deltas['maddux_score'] <= score_range[1])
        ].copy()

        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem; letter-spacing: 2px; color: var(--accent-green);">{len(df_filtered)}</div>
            <div style="font-family: 'Source Sans Pro', sans-serif; font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px;">Players Match</div>
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="position: fixed; bottom: 1rem; padding-right: 1rem;">
            <p style="font-size: 0.7rem; color: var(--text-muted); margin: 0;">
                Data: 2015-2025<br>
                Powered by Claude AI
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # Main Content - Tabs
    # ==============================================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "● Scatter Analysis",
        "● Leaderboard",
        "● Player Deep Dive",
        "● Model Validation",
        "● Ask Claude"
    ])

    # --------------------------------------------------------------------------
    # Tab 1: Scatter Plot
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown(f"### MADDUX Score vs OPS Change — {selected_year}")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Create enhanced scatter plot
            fig = go.Figure()

            # Add scatter points with custom styling
            fig.add_trace(go.Scatter(
                x=df_filtered['maddux_score'],
                y=df_filtered['delta_ops'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=df_filtered['delta_ops'],
                    colorscale=DIVERGING_SCALE,
                    cmin=-0.15,
                    cmax=0.15,
                    line=dict(width=1, color='rgba(255,255,255,0.3)'),
                    opacity=0.85,
                    colorbar=dict(
                        title=dict(text='Δ OPS', font=dict(color='#9aa0a6')),
                        tickfont=dict(color='#9aa0a6'),
                        bgcolor='rgba(21,28,36,0.8)',
                        bordercolor='rgba(255,255,255,0.1)',
                        borderwidth=1
                    )
                ),
                text=df_filtered['player_name'],
                customdata=df_filtered[['team', 'delta_max_ev', 'delta_hard_hit_pct']].values,
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Team: %{customdata[0]}<br>'
                    'MADDUX: %{x:.2f}<br>'
                    'Δ OPS: %{y:+.3f}<br>'
                    'Δ Max EV: %{customdata[1]:+.1f}<br>'
                    'Δ Hard Hit%: %{customdata[2]:+.1f}'
                    '<extra></extra>'
                ),
                name='Players'
            ))

            # Add quadrant reference lines
            fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))
            fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))

            # Add trend line
            if len(df_filtered) > 2:
                import numpy as np
                z = np.polyfit(df_filtered['maddux_score'], df_filtered['delta_ops'], 1)
                x_line = np.linspace(df_filtered['maddux_score'].min(), df_filtered['maddux_score'].max(), 100)
                y_line = z[0] * x_line + z[1]

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#ffd93d', width=2, dash='dot'),
                    opacity=0.7
                ))

            # Quadrant annotations with better styling
            annotations = [
                dict(x=0.85, y=0.9, xref='paper', yref='paper',
                     text='<b>✓ MODEL HIT</b><br><span style="font-size:10px">High Score → OPS Up</span>',
                     showarrow=False, font=dict(size=11, color='#6ee7b7'),
                     bgcolor='rgba(110,231,183,0.1)', bordercolor='#6ee7b7', borderwidth=1, borderpad=6),
                dict(x=0.85, y=0.1, xref='paper', yref='paper',
                     text='<b>? UNLUCKY</b><br><span style="font-size:10px">High Score → OPS Down</span>',
                     showarrow=False, font=dict(size=11, color='#ffd93d'),
                     bgcolor='rgba(255,217,61,0.1)', bordercolor='#ffd93d', borderwidth=1, borderpad=6),
                dict(x=0.15, y=0.9, xref='paper', yref='paper',
                     text='<b>? LUCKY</b><br><span style="font-size:10px">Low Score → OPS Up</span>',
                     showarrow=False, font=dict(size=11, color='#ffd93d'),
                     bgcolor='rgba(255,217,61,0.1)', bordercolor='#ffd93d', borderwidth=1, borderpad=6),
                dict(x=0.15, y=0.1, xref='paper', yref='paper',
                     text='<b>✓ MODEL HIT</b><br><span style="font-size:10px">Low Score → OPS Down</span>',
                     showarrow=False, font=dict(size=11, color='#6ee7b7'),
                     bgcolor='rgba(110,231,183,0.1)', bordercolor='#6ee7b7', borderwidth=1, borderpad=6),
            ]

            fig.update_layout(
                title=dict(text=f'Does MADDUX Predict OPS Change?', font=dict(size=20)),
                xaxis_title='MADDUX Score',
                yaxis_title='Δ OPS',
                height=550,
                showlegend=True,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(21,28,36,0.8)'),
                annotations=annotations
            )

            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Quadrant Guide")
            st.markdown("""
            <div class="quadrant-guide">
                <div class="quadrant-item">
                    <span style="color:#6ee7b7">▸ TOP-RIGHT</span><br>
                    <span style="font-size:0.85rem;color:#9aa0a6">High MADDUX → OPS improved<br>Model predicted correctly</span>
                </div>
                <div class="quadrant-item">
                    <span style="color:#6ee7b7">▸ BOTTOM-LEFT</span><br>
                    <span style="font-size:0.85rem;color:#9aa0a6">Low MADDUX → OPS declined<br>Model predicted correctly</span>
                </div>
                <div class="quadrant-item">
                    <span style="color:#ffd93d">▸ TOP-LEFT</span><br>
                    <span style="font-size:0.85rem;color:#9aa0a6">Low MADDUX → OPS improved<br>Outperformed expectations</span>
                </div>
                <div class="quadrant-item">
                    <span style="color:#ffd93d">▸ BOTTOM-RIGHT</span><br>
                    <span style="font-size:0.85rem;color:#9aa0a6">High MADDUX → OPS declined<br>Underperformed expectations</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Key Metrics")

            correlation = df_filtered['maddux_score'].corr(df_filtered['delta_ops'])
            st.metric("Correlation (r)", f"{correlation:.3f}")

            high_score_players = df_filtered[df_filtered['maddux_score'] > 10]
            if len(high_score_players) > 0:
                high_score_success = (high_score_players['delta_ops'] > 0).sum() / len(high_score_players)
            else:
                high_score_success = 0
            st.metric("High Score Hit Rate", f"{high_score_success:.1%}")

    # --------------------------------------------------------------------------
    # Tab 2: Leaderboard
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown(f"""
        <h3 style="font-family: 'Bebas Neue', sans-serif; font-size: 1.75rem; letter-spacing: 2px;
                   color: var(--text-primary); margin-bottom: 0.25rem;">MADDUX Leaderboard — {selected_year}</h3>
        <p style="font-family: 'Source Sans Pro', sans-serif; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem;">
            Top breakout and regression candidates based on physical metric changes.
        </p>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ↑ Breakout Candidates")
            top_15 = df_filtered.nlargest(15, 'maddux_score')[
                ['player_name', 'team', 'maddux_score', 'delta_max_ev',
                 'delta_hard_hit_pct', 'delta_ops', 'ops']
            ].reset_index(drop=True)
            top_15.index = top_15.index + 1
            top_15.columns = ['Player', 'Team', 'MADDUX', 'Δ MaxEV', 'Δ HH%', 'Δ OPS', 'OPS']

            st.dataframe(
                top_15.style.format({
                    'MADDUX': '{:.1f}',
                    'Δ MaxEV': '{:+.1f}',
                    'Δ HH%': '{:+.1f}',
                    'Δ OPS': '{:+.3f}',
                    'OPS': '{:.3f}'
                }).background_gradient(subset=['MADDUX'], cmap='Greens'),
                use_container_width=True,
                height=400
            )

        with col2:
            st.markdown("#### ↓ Regression Candidates")
            bottom_15 = df_filtered.nsmallest(15, 'maddux_score')[
                ['player_name', 'team', 'maddux_score', 'delta_max_ev',
                 'delta_hard_hit_pct', 'delta_ops', 'ops']
            ].reset_index(drop=True)
            bottom_15.index = bottom_15.index + 1
            bottom_15.columns = ['Player', 'Team', 'MADDUX', 'Δ MaxEV', 'Δ HH%', 'Δ OPS', 'OPS']

            st.dataframe(
                bottom_15.style.format({
                    'MADDUX': '{:.1f}',
                    'Δ MaxEV': '{:+.1f}',
                    'Δ HH%': '{:+.1f}',
                    'Δ OPS': '{:+.3f}',
                    'OPS': '{:.3f}'
                }).background_gradient(subset=['MADDUX'], cmap='Reds_r'),
                use_container_width=True,
                height=400
            )

        # Enhanced bar chart
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Score Distribution")

        top_10 = df_filtered.nlargest(10, 'maddux_score')

        fig_bar = go.Figure()

        # Create gradient colors based on score
        colors = ['#6ee7b7' if s > 0 else '#ff6b6b' for s in top_10['maddux_score']]

        max_score = top_10['maddux_score'].max()

        fig_bar.add_trace(go.Bar(
            x=top_10['player_name'],
            y=top_10['maddux_score'],
            marker=dict(
                color=top_10['maddux_score'],
                colorscale=SEQUENTIAL_GREEN,
                line=dict(width=0)
            ),
            text=top_10['maddux_score'].round(1),
            textposition='outside',
            textfont=dict(color='#6ee7b7', size=12, family='JetBrains Mono'),
            hovertemplate='<b>%{x}</b><br>MADDUX: %{y:.1f}<extra></extra>',
            cliponaxis=False
        ))

        fig_bar.update_layout(
            title=dict(text=f'Top 10 MADDUX Scores', font=dict(size=18)),
            xaxis_title='',
            yaxis_title='MADDUX Score',
            height=450,
            showlegend=False,
            xaxis_tickangle=-45,
            yaxis_range=[0, max_score * 1.15]
        )

        apply_plotly_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------------------------------
    # Tab 3: Player Deep Dive
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("""
        <h3 style="font-family: 'Bebas Neue', sans-serif; font-size: 1.75rem; letter-spacing: 2px;
                   color: var(--text-primary); margin-bottom: 0.25rem;">Player Deep Dive</h3>
        <p style="font-family: 'Source Sans Pro', sans-serif; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem;">
            Explore individual player trajectories and MADDUX metrics over time.
        </p>
        """, unsafe_allow_html=True)

        # Player selector
        all_players = sorted(df_deltas['player_name'].unique())
        selected_player = st.selectbox("Select Player", all_players, index=0)

        # Get player history
        player_history = get_player_history(selected_player)

        if len(player_history) > 0:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Enhanced MADDUX score over time chart
                fig_history = go.Figure()

                # MADDUX Score line
                fig_history.add_trace(go.Scatter(
                    x=player_history['year'],
                    y=player_history['maddux_score'],
                    mode='lines+markers',
                    name='MADDUX Score',
                    line=dict(color='#6ee7b7', width=3),
                    marker=dict(size=12, color='#6ee7b7', line=dict(width=2, color='#0a0e14')),
                    fill='tozeroy',
                    fillcolor='rgba(110,231,183,0.1)'
                ))

                # OPS change bars
                colors = ['#6ee7b7' if x > 0 else '#ff6b6b' for x in player_history['delta_ops']]
                fig_history.add_trace(go.Bar(
                    x=player_history['year'],
                    y=player_history['delta_ops'] * 100,
                    name='Δ OPS (×100)',
                    marker=dict(color=colors, opacity=0.6),
                    yaxis='y2'
                ))

                fig_history.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash'))

                fig_history.update_layout(
                    title=dict(text=f'{selected_player} — Career Trajectory', font=dict(size=20)),
                    xaxis_title='Season',
                    yaxis=dict(
                        title='MADDUX Score',
                        side='left',
                        color='#6ee7b7',
                        showgrid=False
                    ),
                    yaxis2=dict(
                        title='Δ OPS (×100)',
                        side='right',
                        overlaying='y',
                        color='#9aa0a6',
                        showgrid=False
                    ),
                    height=420,
                    showlegend=True,
                    legend=dict(x=0.02, y=0.98),
                    barmode='overlay'
                )

                apply_plotly_theme(fig_history)
                # Remove grid lines for cleaner look
                fig_history.update_yaxes(showgrid=False)
                fig_history.update_xaxes(showgrid=False)
                st.plotly_chart(fig_history, use_container_width=True)

            with col2:
                # Enhanced radar chart
                latest = player_history.iloc[-1]

                fig_radar = go.Figure()

                categories = ['Max EV', 'Hard Hit %', 'OPS', 'MADDUX']

                # Normalize values
                values = [
                    min(max((latest['max_ev'] - 100) * 5, 0), 100),
                    min(latest['hard_hit_pct'], 100),
                    min(latest['ops'] * 100, 100),
                    min(max((latest['maddux_score'] + 30) * 1.67, 0), 100)
                ]

                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(110,231,183,0.2)',
                    line=dict(color='#6ee7b7', width=2),
                    marker=dict(size=8, color='#6ee7b7'),
                    name=selected_player
                ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            gridcolor='rgba(255,255,255,0.1)',
                            linecolor='rgba(255,255,255,0.1)'
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            linecolor='rgba(255,255,255,0.1)'
                        ),
                        bgcolor='rgba(21,28,36,0.5)'
                    ),
                    showlegend=False,
                    title=dict(text=f"Profile ({int(latest['year'])})", font=dict(size=16)),
                    height=420,
                    paper_bgcolor='rgba(21,28,36,0)',
                    font=dict(color='#e8eaed')
                )

                st.plotly_chart(fig_radar, use_container_width=True)

            # Stats table
            st.markdown("#### Season-by-Season Stats")
            display_cols = ['year', 'team', 'pa', 'max_ev', 'hard_hit_pct',
                          'delta_max_ev', 'delta_hard_hit_pct', 'maddux_score',
                          'ops', 'delta_ops']

            styled_df = player_history[display_cols].copy()
            styled_df.columns = ['Year', 'Team', 'PA', 'Max EV', 'HH%', 'Δ Max EV', 'Δ HH%', 'MADDUX', 'OPS', 'Δ OPS']

            st.dataframe(
                styled_df.style.format({
                    'Max EV': '{:.1f}',
                    'HH%': '{:.1f}',
                    'Δ Max EV': '{:+.1f}',
                    'Δ HH%': '{:+.1f}',
                    'MADDUX': '{:.1f}',
                    'OPS': '{:.3f}',
                    'Δ OPS': '{:+.3f}'
                }).background_gradient(subset=['MADDUX'], cmap='RdYlGn', vmin=-30, vmax=30),
                use_container_width=True
            )

            # AI Analysis
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### AI-Powered Analysis")

            st.markdown("""
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
                Get Claude's assessment of this player's breakout potential based on MADDUX metrics.
            </p>
            """, unsafe_allow_html=True)

            if st.button("Generate Analysis", type="primary"):
                with st.spinner("Analyzing player metrics..."):
                    latest_data = player_history.iloc[-1].to_dict()
                    analysis = get_claude_analysis(latest_data)
                    st.markdown(f"""
                    <div style="background: var(--bg-tertiary); border: 1px solid var(--border-subtle);
                                border-radius: 8px; padding: 1.5rem; margin-top: 1rem;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                            <span style="color: var(--accent-green); font-weight: 600; font-size: 0.75rem;
                                         text-transform: uppercase; letter-spacing: 1px;">Claude Analysis</span>
                        </div>
                        <p style="color: var(--text-primary); font-size: 0.95rem; line-height: 1.6; margin: 0;">
                            {analysis}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No delta data available for this player.")

    # --------------------------------------------------------------------------
    # Tab 4: Model Validation
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### Model Validation")
        st.markdown("""
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem;">
            Evaluating MADDUX predictive accuracy across historical seasons.
        </p>
        """, unsafe_allow_html=True)

        # Overall correlation by year
        st.markdown("#### Year-over-Year Correlation")

        correlations = []
        for year in sorted(df_deltas['year'].unique()):
            year_data = df_deltas[df_deltas['year'] == year]
            corr = year_data['maddux_score'].corr(year_data['delta_ops'])
            n = len(year_data)
            correlations.append({'Year': year, 'Correlation': corr, 'N': n})

        corr_df = pd.DataFrame(correlations)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_corr = go.Figure()

            colors = ['#6ee7b7' if c > 0 else '#ff6b6b' for c in corr_df['Correlation']]

            fig_corr.add_trace(go.Bar(
                x=corr_df['Year'],
                y=corr_df['Correlation'],
                marker=dict(
                    color=corr_df['Correlation'],
                    colorscale=SEQUENTIAL_GREEN,
                    cmin=corr_df['Correlation'].min(),
                    cmax=corr_df['Correlation'].max(),
                    line=dict(width=0)
                ),
                text=corr_df['Correlation'].round(3),
                textposition='outside',
                textfont=dict(size=11, family='JetBrains Mono', color='#e8eaed'),
                hovertemplate='<b>%{x}</b><br>r = %{y:.3f}<extra></extra>',
                cliponaxis=False
            ))

            fig_corr.add_hline(y=0, line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'))

            max_corr = corr_df['Correlation'].max()

            fig_corr.update_layout(
                title=dict(text='MADDUX ↔ Δ OPS Correlation by Year', font=dict(size=18)),
                xaxis_title='',
                yaxis_title='Correlation (r)',
                height=450,
                yaxis_range=[-0.2, max_corr + 0.15]
            )

            apply_plotly_theme(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.dataframe(
                corr_df.style.format({'Correlation': '{:.3f}'})
                .background_gradient(subset=['Correlation'], cmap=MINT_CMAP, vmin=corr_df['Correlation'].min(), vmax=corr_df['Correlation'].max()),
                use_container_width=True
            )

            avg_corr = corr_df['Correlation'].mean()
            corr_class = "positive" if avg_corr > 0 else "negative"
            st.markdown(f"""
            <div class="stat-card {corr_class}">
                <div class="stat-value {corr_class}">{avg_corr:.3f}</div>
                <div class="stat-label">Average Correlation</div>
            </div>
            """, unsafe_allow_html=True)

        # Hit rate analysis
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Prediction Accuracy by Threshold")

        thresholds = [5, 10, 15, 20]
        hit_rates = []

        for thresh in thresholds:
            high_score = df_deltas[df_deltas['maddux_score'] > thresh]
            if len(high_score) > 0:
                hits = (high_score['delta_ops'] > 0).sum()
                rate = hits / len(high_score)
                hit_rates.append({
                    'Threshold': f'>{thresh}',
                    'Hit Rate': rate,
                    'N': len(high_score),
                    'Hits': hits
                })

        hr_df = pd.DataFrame(hit_rates)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_hr = go.Figure()

            fig_hr.add_trace(go.Bar(
                x=hr_df['Threshold'],
                y=hr_df['Hit Rate'],
                marker=dict(
                    color=hr_df['Hit Rate'],
                    colorscale=[[0, '#ff6b6b'], [0.5, '#ffd93d'], [1, '#6ee7b7']],
                    cmin=0.4,
                    cmax=0.8,
                    line=dict(width=2, color='rgba(255,255,255,0.2)')
                ),
                text=[f'{r:.0%}' for r in hr_df['Hit Rate']],
                textposition='outside',
                textfont=dict(size=14, family='Bebas Neue', color='#e8eaed'),
                hovertemplate='<b>MADDUX %{x}</b><br>Hit Rate: %{y:.1%}<br>N = %{customdata}<extra></extra>',
                customdata=hr_df['N']
            ))

            fig_hr.add_hline(
                y=0.5,
                line=dict(color='#dc2626', width=2, dash='dash'),
                annotation=dict(text='Random (50%)', font=dict(color='#dc2626', size=12, weight=600))
            )

            fig_hr.update_layout(
                title=dict(text='OPS Improvement Rate by MADDUX Threshold', font=dict(size=18)),
                xaxis_title='MADDUX Score Threshold',
                yaxis_title='Hit Rate',
                height=400,
                yaxis_range=[0, 1],
                yaxis_tickformat='.0%'
            )

            apply_plotly_theme(fig_hr)
            st.plotly_chart(fig_hr, use_container_width=True)

        with col2:
            st.dataframe(
                hr_df.style.format({'Hit Rate': '{:.1%}'})
                .background_gradient(subset=['Hit Rate'], cmap='RdYlGn', vmin=0.4, vmax=0.8),
                use_container_width=True
            )

            st.markdown("#### Target Benchmarks")
            st.markdown("""
            <div class="quadrant-guide">
                <div class="quadrant-item">
                    <span style="color:#9aa0a6">Correlation</span><br>
                    <span style="color:#ffd93d;font-family:'JetBrains Mono'">Target: r > 0.15</span>
                </div>
                <div class="quadrant-item">
                    <span style="color:#9aa0a6">Hit Rate (>20)</span><br>
                    <span style="color:#ffd93d;font-family:'JetBrains Mono'">Target: > 65%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Tab 5: Ask Claude
    # --------------------------------------------------------------------------
    with tab5:
        st.markdown("### Ask Claude")
        st.markdown("""
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem;">
            Ask natural language questions about the MADDUX database. Claude will generate SQL queries and analyze the results.
        </p>
        """, unsafe_allow_html=True)

        # Example questions
        with st.expander("Example questions"):
            st.markdown("""
            - Who are the top 10 breakout candidates for 2025?
            - Show me players who improved their MADDUX score by 20+ points and what happened to their OPS
            - What's Aaron Judge's MADDUX history?
            - Which teams have the most high-MADDUX players in 2024?
            - Find players with negative MADDUX scores who still improved their OPS
            - What's the average OPS change for players with MADDUX scores above 15?
            """)

        # Query input
        user_question = st.text_area(
            "Your question",
            placeholder="e.g., Who are the top breakout candidates for 2025 based on MADDUX score?",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("Query", type="primary", use_container_width=True)

        if submit_button and user_question:
            with st.spinner("Claude is analyzing..."):
                try:
                    import re
                    client = Anthropic()
                    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

                    # Database schema for context
                    schema_context = """
DATABASE SCHEMA:

TABLE: players
- player_id (INTEGER PRIMARY KEY) - MLBAM ID
- player_name (TEXT) - Full name

TABLE: player_seasons
- player_id, year, pa, max_ev, hard_hit_pct, barrel_pct, ops, obp, slg, wrc_plus, team

TABLE: player_deltas
- player_id, year, prev_year, pa
- max_ev, prev_max_ev, delta_max_ev
- hard_hit_pct, prev_hard_hit_pct, delta_hard_hit_pct
- ops, prev_ops, delta_ops
- maddux_score (= delta_max_ev + 2.1 * delta_hard_hit_pct)
- team

DATA: 2015-2025, 1341 players, 3606 delta records
"""

                    # Step 1: Generate SQL
                    sql_response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        system=f"""You are a baseball analytics SQL expert. Generate SQLite queries for the MADDUX database.

{schema_context}

Return ONLY the SQL query wrapped in ```sql``` tags. Use JOINs to get player names. Limit to 20 rows unless asked for more.""",
                        messages=[{"role": "user", "content": user_question}]
                    )

                    sql_text = sql_response.content[0].text

                    # Extract SQL
                    sql_match = re.search(r'```sql\s*(.*?)\s*```', sql_text, re.DOTALL)

                    if sql_match:
                        sql = sql_match.group(1).strip()

                        # Display SQL
                        st.markdown("#### Generated SQL")
                        st.code(sql, language="sql")

                        # Execute query
                        conn = sqlite3.connect(DB_PATH)
                        try:
                            results_df = pd.read_sql_query(sql, conn)
                            conn.close()

                            # Display results
                            st.markdown(f"#### Results ({len(results_df)} rows)")
                            st.dataframe(results_df, use_container_width=True, height=300)

                            # Step 2: Get analysis
                            if len(results_df) > 0:
                                analysis_response = client.messages.create(
                                    model=model,
                                    max_tokens=1000,
                                    messages=[{
                                        "role": "user",
                                        "content": f"""The user asked: "{user_question}"

SQL Results ({len(results_df)} rows):
{results_df.head(20).to_string()}

Provide a concise analysis:
1. Direct answer to the question
2. Key insights from the data
3. Any caveats or limitations

The MADDUX score formula is: delta_max_ev + (2.1 × delta_hard_hit_pct)
Higher scores indicate physical improvement that often precedes OPS gains."""
                                    }]
                                )

                                analysis = analysis_response.content[0].text

                                st.markdown("#### Analysis")
                                st.markdown(f"""
                                <div style="background: var(--bg-tertiary); border: 1px solid var(--border-subtle);
                                            border-radius: 8px; padding: 1.5rem; margin-top: 1rem;">
                                    <p style="color: var(--text-primary); font-size: 0.95rem; line-height: 1.6; margin: 0;">
                                        {analysis}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

                        except Exception as e:
                            conn.close()
                            st.error(f"SQL Error: {str(e)}")
                            st.markdown("The generated SQL had an error. Try rephrasing your question.")

                    else:
                        st.warning("Could not generate a valid SQL query. Try rephrasing your question.")
                        st.markdown(f"Claude's response: {sql_text}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        elif submit_button:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
