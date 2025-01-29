import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Function to calculate and normalize the probabilities
def calculate_normalized_probabilities(player_strategy, accuracies):
    effective_probabilities = [p * acc for p, acc in zip(player_strategy, accuracies)]
    total = sum(effective_probabilities)
    normalized_probabilities = [p / total for p in effective_probabilities]
    return normalized_probabilities

# Function to draw the goal with color-coded shot probabilities
def draw_color_coded_goal(normalized_probabilities):
    goal_width, goal_height = 7.32, 2.44
    num_sections = len(normalized_probabilities)
    section_width = goal_width / num_sections

    sorted_indices = np.argsort(normalized_probabilities)
    color_map = ['#FF4D4D', '#FFFF66', '#66FF66']  # Red, Yellow, Green
    section_colors = [None] * num_sections
    for rank, idx in enumerate(sorted_indices):
        section_colors[idx] = color_map[rank]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-1, goal_width + 1)
    ax.set_ylim(-1, goal_height + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw grey goalposts
    post_color = 'grey'
    ax.plot([0, 0], [0, goal_height], color=post_color, linewidth=3)  # Left post
    ax.plot([goal_width, goal_width], [0, goal_height], color=post_color, linewidth=3)  # Right post
    ax.plot([0, goal_width], [goal_height, goal_height], color=post_color, linewidth=3)  # Crossbar

    # Draw net inside the goal
    net_lines = 8  # Number of vertical and horizontal lines
    for i in range(1, net_lines):
        # Vertical net lines
        x = i * goal_width / net_lines
        ax.plot([x, x], [0, goal_height], color='gray', linestyle='dotted', alpha=0.7)
        # Horizontal net lines
        y = i * goal_height / net_lines
        ax.plot([0, goal_width], [y, y], color='gray', linestyle='dotted', alpha=0.7)

    # Draw color-coded probability sections
    for i, (norm_prob, color) in enumerate(zip(normalized_probabilities, section_colors)):
        x_start = i * section_width
        ax.add_patch(plt.Rectangle((x_start, 0), section_width, goal_height, color=color, alpha=0.5))  # More transparency
        ax.text(x_start + section_width / 2, goal_height / 2, f"{norm_prob:.1%}", color="black", ha="center", va="center", fontsize=10, weight="bold")

    return fig  # Explicitly return the figure

# Configure the page
st.set_page_config(page_title="Game Theory Apps", layout="wide")
st.title("Game Theory Applications")

# Create tabs
tab1, tab2 = st.tabs(["Penalty Kick Analyzer", "Take vs. Share Dilemma"])

# ------------------------------------------------------------------------------------
# Tab 1: Penalty Kick Analyzer
# ------------------------------------------------------------------------------------

with tab1:
    st.title("Penalty Kick Analyzer")
    X_percent = st.slider('Kicker’s effectiveness when kicking right (X%)', 0, 100, 50, 1, key="penalty_slider")
    X = X_percent / 100.0
    p = 1 / (1 + X)
    q = X / (1 + X)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kicker's optimal probability to choose Left", f"{p * 100:.1f}%")
    with col2:
        st.metric("Goalie's optimal probability to dive Left", f"{q * 100:.1f}%")
    X_percent_values = np.linspace(0, 100, 100)
    X_values = X_percent_values / 100.0
    p_values = 1 / (1 + X_values)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X_percent_values, y=p_values * 100, mode='lines', line=dict(color='purple', width=3)))
    fig1.update_layout(title="Optimal Kick Strategy vs Effectiveness", xaxis_title="X (% Effectiveness When Kicking Right)", yaxis_title="Probability to Choose Left (%)", height=500)
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Why this counterintuitive result?"):
        st.markdown("""
        - **Higher right-side skill (X↑)**: Goalie anticipates this and dives right more often.  
        - **Paradoxical solution**: To exploit the goalie's bias, you must kick left **more frequently** than intuition suggests!
        """)

    st.markdown("""
    #### Payoff Matrix
    |                | Keeper: Left | Keeper: Middle | Keeper: Right |
    |----------------|----------------|-----------------|------------------------|
    | **Kicker: Right**  | (0, 0)        | (8000, 0)       | (X1, Y1)              |
    | **Kicker: Middle** | (0, 8000)     | (0, 0)    | (X2, Y2)              |
    | **Kicker: Left** | (X3, Y3)     | (X4, Y4)      | (0, 0)              |
    """)

    # Input sliders for strategy and accuracy
    player_strategy_left = st.slider("Probability for Left", 0.0, 1.0, 0.9)
    player_strategy_middle = st.slider("Probability for Middle", 0.0, 1.0, 0.02)
    player_strategy_right = st.slider("Probability for Right", 0.0, 1.0, 0.08)
    accuracy_left = st.slider("Accuracy for Left", 0.0, 1.0, 0.5)
    accuracy_middle = st.slider("Accuracy for Middle", 0.0, 1.0, 1.0)
    accuracy_right = st.slider("Accuracy for Right", 0.0, 1.0, 1.0)

    # Compute normalized probabilities
    player_strategy = [player_strategy_left, player_strategy_middle, player_strategy_right]
    accuracies = [accuracy_left, accuracy_middle, accuracy_right]
    normalized_probabilities = calculate_normalized_probabilities(player_strategy, accuracies)

    # Generate and display goal visualization
    fig2 = draw_color_coded_goal(normalized_probabilities)
    st.pyplot(fig2)  # Only call once

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
