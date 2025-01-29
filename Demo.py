import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configure the page
st.set_page_config(page_title="Game Theory Suite", layout="wide")
st.title("Game Theory Application Suite")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Penalty Kick Analyzer", "Take vs. Share Dilemma", "RPS Analyzer"])

# ====================================================================================
# Tab 1: Penalty Kick Analyzer
# ====================================================================================
with tab1:
    st.header("Penalty Kick Strategy Analysis")

    # Slider for X (percentage)
    X_percent = st.slider(
        'Kicker’s effectiveness when kicking left (X%)',
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

# ====================================================================================
# Tab 2: Take vs. Share Dilemma
# ====================================================================================
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

# ====================================================================================
# Tab 3: RPS Analyzer
# ====================================================================================
with tab3:
    st.header("Advanced Rock-Paper-Scissors Analyzer")

    # ================== Payoff Matrix Section ==================
    st.sidebar.header("⚖️ RPS Configuration")
    payoff_mode = st.sidebar.radio(
        "Preset Modes",
        ('Standard RPS', 'Modified RPS', 'Custom'),
        help="Standard: Classic RPS rules\nModified: Adjusted payoffs\nCustom: Full control"
    )

    if payoff_mode == 'Standard RPS':
        payoff_matrix = {
            'Rock': {'Rock': (0, 0), 'Paper': (-1, 1), 'Scissors': (1, -1)},
            'Paper': {'Rock': (1, -1), 'Paper': (0, 0), 'Scissors': (-1, 1)},
            'Scissors': {'Rock': (-1, 1), 'Paper': (1, -1), 'Scissors': (0, 0)}
        }
    elif payoff_mode == 'Modified RPS':
        payoff_matrix = {
            'Rock': {'Rock': (0, 0), 'Paper': (-2, 2), 'Scissors': (1, -1)},
            'Paper': {'Rock': (2, -2), 'Paper': (0, 0), 'Scissors': (-1, 1)},
            'Scissors': {'Rock': (-1, 1), 'Paper': (1, -1), 'Scissors': (0, 0)}
        }
    else:
        st.sidebar.subheader("Custom Payoffs")
        payoff_matrix = {}
        for move in ['Rock', 'Paper', 'Scissors']:
            payoff_matrix[move] = {}
            for opp_move in ['Rock', 'Paper', 'Scissors']:
                bot_payoff = st.sidebar.number_input(
                    f'{move} vs {opp_move} (Bot payoff)',
                    min_value=-10,
                    max_value=10,
                    value=0
                )
                opp_payoff = -bot_payoff  # Zero-sum assumption
                payoff_matrix[move][opp_move] = (bot_payoff, opp_payoff)

    # Display payoff matrix
    st.subheader("Current Payoff Matrix (Bot, Opponent)")
    display_matrix = pd.DataFrame({
        move: {opp_move: f"({payoff_matrix[move][opp_move][0]}, {payoff_matrix[move][opp_move][1]})"
               for opp_move in payoff_matrix[move]}
        for move in payoff_matrix
    }).T
    st.dataframe(display_matrix)

    # ================== Bot Strategy Section ==================
    st.sidebar.header("🤖 Bot Strategy")
    bot_strategy_mode = st.sidebar.radio(
        "Strategy Mode",
        ('Optimal', 'Fixed R:25% P:25% S:50%'),
        help="Select the bot's strategy"
    )

    if bot_strategy_mode == 'Optimal':
        # Calculate expected utilities
        opponent_strategy = {move: 1 / 3 for move in ['Rock', 'Paper', 'Scissors']}  # Default uniform
        expected_utilities = {
            move: sum(payoff_matrix[move][opp_move][0] * prob
                      for opp_move, prob in opponent_strategy.items())
            for move in ['Rock', 'Paper', 'Scissors']
        }

        # Calculate optimal strategy
        positive_eu = {move: max(eu, 0) for move, eu in expected_utilities.items()}
        total_eu = sum(positive_eu.values())
        bot_strategy = {move: eu / total_eu for move, eu in positive_eu.items()} if total_eu > 0 else {move: 1 / 3 for
                                                                                                       move in
                                                                                                       ['Rock', 'Paper',
                                                                                                        'Scissors']}
    else:
        bot_strategy = {'Rock': 0.25, 'Paper': 0.25, 'Scissors': 0.50}

    # Display strategy
    st.subheader("Bot's Strategy Distribution")
    strategy_df = pd.DataFrame.from_dict(bot_strategy, orient='index', columns=['Probability'])
    st.table(strategy_df.style.format("{:.2%}"))

    # ================== Simulation Section ==================
    st.sidebar.header("🎲 Simulation Settings")
    sim_mode = st.sidebar.radio(
        "Simulation Mode",
        ('Manual Input', 'Random Trials'),
        help="Choose audience input method"
    )

    if sim_mode == 'Manual Input':
        st.sidebar.subheader("Audience Distribution")
        rock = st.sidebar.number_input("Rock Choices", 0, 10000, 0)
        paper = st.sidebar.number_input("Paper Choices", 0, 10000, 0)
        scissors = st.sidebar.number_input("Scissors Choices", 0, 10000, 0)
        total = rock + paper + scissors
        audience_dist = {
            'Rock': rock / total if total > 0 else 0.33,
            'Paper': paper / total if total > 0 else 0.33,
            'Scissors': scissors / total if total > 0 else 0.33
        }
    else:
        n_trials = st.sidebar.selectbox("Number of Trials", [100, 1000, 10000], index=2)
        if st.sidebar.button("Generate Random Trials"):
            audience_dist = dict(zip(
                ['Rock', 'Paper', 'Scissors'],
                np.random.dirichlet(np.ones(3), size=1)[0]
            ))
        else:
            audience_dist = {'Rock': 0.33, 'Paper': 0.33, 'Scissors': 0.34}

    if st.sidebar.button("Run Simulation"):
        st.subheader("Simulation Results")

        # Generate choices
        n = 10000  # Fixed to 10,000 trials
        bot_choices = np.random.choice(
            list(bot_strategy.keys()),
            size=n,
            p=list(bot_strategy.values())
        )
        audience_choices = np.random.choice(
            list(audience_dist.keys()),
            size=n,
            p=list(audience_dist.values())
        )

        # Calculate scores
        scores = [payoff_matrix[bot][audience][0] for bot, audience in zip(bot_choices, audience_choices)]
        cumulative_scores = np.cumsum(scores)

        # Results analysis
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Score (Bot)", f"{sum(scores):+,}")
            fig_cum = px.line(x=range(n), y=cumulative_scores,
                              labels={'x': 'Trial', 'y': 'Cumulative Score'},
                              title="Score Progression")
            st.plotly_chart(fig_cum, use_container_width=True)

        with col2:
            dist_df = pd.DataFrame({
                'Bot': pd.Series(bot_choices).value_counts(normalize=True),
                'Audience': pd.Series(audience_choices).value_counts(normalize=True)
            })
            fig_dist = px.bar(dist_df, barmode='group',
                              labels={'value': 'Frequency', 'variable': 'Player'},
                              title="Choice Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)
