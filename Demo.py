import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Configure the page
st.set_page_config(page_title="Game Theory Apps", layout="wide")
st.title("Game Theory Applications")

# Create tabs
tab1, tab2 = st.tabs(["Penalty Kick Analyzer", "Take vs. Share Dilemma"])

# ------------------------------------------------------------------------------------
# Tab 1: Penalty Kick Analyzer
# ------------------------------------------------------------------------------------
with tab1:
    st.header("Penalty Kick Strategy Analysis")

    # Slider for X (percentage)
    X_percent = st.slider(
        'Kicker’s effectiveness when kicking right (X%)',
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key="penalty_slider"
    )
    X = X_percent / 100.0  # Convert to decimal

    # Calculate probabilities
    p = 1 / (1 + X)
    q = X / (1 + X)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kicker's optimal probability to choose Left", f"{p * 100:.1f}%")
    with col2:
        st.metric("Goalie's optimal probability to dive Left", f"{q * 100:.1f}%")

    # Plot
    X_percent_values = np.linspace(0, 100, 100)
    X_values = X_percent_values / 100.0
    p_values = 1 / (1 + X_values)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X_percent_values, y=p_values * 100, mode='lines',
                              line=dict(color='purple', width=3)))
    fig1.update_layout(
        title="Optimal Kick Strategy vs Effectiveness",
        xaxis_title="X (% Effectiveness When Kicking Right)",
        yaxis_title="Probability to Choose Left (%)",
        height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Explanation
    with st.expander("Why this counterintuitive result?"):
        st.markdown("""
        - **Higher right-side skill (X↑)**: Goalie anticipates this and dives right more often.  
        - **Paradoxical solution**: To exploit the goalie's bias, you must kick left **more frequently** than intuition suggests!
        """)

# ------------------------------------------------------------------------------------
# Tab 2: Take vs. Share Dilemma
# ------------------------------------------------------------------------------------
with tab2:
    st.header("Take vs. Share Strategic Analysis")

    # Slider for q
    q = st.slider(
        "Player 2's probability of Taking (q)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="take_share_slider"
    )

    # Calculate payoffs
    E_Take = 8000 * (1 - q)
    E_Share = 4000 * (1 - q)

    # Dominance visualization
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=8000 * (1 - np.linspace(0, 1, 100)),
                              name='Take', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=np.linspace(0, 1, 100), y=4000 * (1 - np.linspace(0, 1, 100)),
                              name='Share', line=dict(color='blue')))

    # Highlight dominance region
    fig2.add_shape(type="rect", x0=0, x1=1, y0=4000, y1=8000,
                   fillcolor="rgba(255,182,193,0.3)", line_width=0)

    fig2.update_layout(
        title="Expected Payoffs vs Opponent's Strategy",
        xaxis_title="Probability Player 2 Takes (q)",
        yaxis_title="Expected Payoff",
        height=500
    )

    # Layout
    col3, col4 = st.columns([1, 2])
    with col3:
        st.metric("Expected Payoff for Taking", f"${E_Take:,.0f}")
        st.metric("Expected Payoff for Sharing", f"${E_Share:,.0f}")
        st.error("**Take always dominates!**")

    with col4:
        st.plotly_chart(fig2, use_container_width=True)

    # Game matrix
    st.markdown("""
    #### Payoff Matrix
    |                | Player 2: Take | Player 2: Share |
    |----------------|----------------|-----------------|
    | **Player 1: Take**  | (0, 0)        | (8000, 0)       |
    | **Player 1: Share** | (0, 8000)     | (4000, 4000)    |
    """)

# ------------------------------------------------------------------------------------
# Run with: streamlit run Demo.py