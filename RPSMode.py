import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Cache heavy computations
@st.cache_data
def calculate_expected_utilities(payoff_matrix, opponent_strategy):
    """Calculate expected utilities for each move"""
    return {
        move: sum(payoff_matrix[move][opp_move][0] * prob
                  for opp_move, prob in opponent_strategy.items())
        for move in ['Rock', 'Paper', 'Scissors']
    }

def main():
    st.title("Advanced Rock-Paper-Scissors Game Theory Analyzer")

    # ================== Payoff Matrix Section ==================
    st.sidebar.header("⚖️ Payoff Matrix Configuration")
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
                opp_payoff = -bot_payoff  # Assuming zero-sum
                payoff_matrix[move][opp_move] = (bot_payoff, opp_payoff)

    # Display payoff matrix
    st.subheader("Current Payoff Matrix (Bot, Opponent)")
    display_matrix = pd.DataFrame({
        move: {opp_move: f"({payoff_matrix[move][opp_move][0]}, {payoff_matrix[move][opp_move][1]})"
                for opp_move in payoff_matrix[move]}
        for move in payoff_matrix
    }).T
    st.dataframe(display_matrix)

    # ================== Opponent Strategy Section ==================
    st.sidebar.header("🎮 Opponent Strategy")
    strategy_type = st.sidebar.radio(
        "Strategy Type",
        ('Uniform', 'Custom'),
        help="Define how you expect your opponent to play"
    )

    if strategy_type == 'Uniform':
        opponent_strategy = {move: 1 / 3 for move in ['Rock', 'Paper', 'Scissors']}
    else:
        st.sidebar.subheader("Custom Probabilities")
        rock = st.sidebar.slider("Rock Probability", 0.0, 1.0, 0.33)
        paper = st.sidebar.slider("Paper Probability", 0.0, 1.0, 0.33)
        scissors = st.sidebar.slider("Scissors Probability", 0.0, 1.0, 1.0 - rock - paper)
        total = rock + paper + scissors
        if total != 1.0:
            st.sidebar.error("Probabilities must sum to 1. They have been normalized.")
            rock /= total
            paper /= total
            scissors /= total

        opponent_strategy = {
            'Rock': rock,
            'Paper': paper,
            'Scissors': scissors
        }

    # ================== Game Theory Calculations ==================
    st.subheader("📈 Game Theory Analysis")

    # Expected utilities
    expected_utilities = calculate_expected_utilities(payoff_matrix, opponent_strategy)
    eu_df = pd.DataFrame.from_dict(expected_utilities, orient='index', columns=['Expected Utility'])
    st.subheader("Expected Utilities")
    st.table(eu_df.style.format("{:.4f}"))

    # Bot strategy
    st.sidebar.header("🧠 Bot Strategy Configuration")
    bot_strategy_mode = st.sidebar.radio(
        "Bot Strategy Mode",
        ('Optimal', 'Fixed R:25% P:25% S:50%', 'Always Scissors'),
        help="Select the bot's strategy"
    )

    if bot_strategy_mode == 'Optimal':
        # Calculate bot strategy based on expected utilities
        positive_eu = {move: max(eu, 0) for move, eu in expected_utilities.items()}
        total_eu = sum(positive_eu.values())

        if total_eu > 0:
            bot_strategy = {move: eu / total_eu for move, eu in positive_eu.items()}
        else:
            st.warning("All expected utilities ≤ 0. Using uniform strategy.")
            bot_strategy = {move: 1 / 3 for move in ['Rock', 'Paper', 'Scissors']}
    elif bot_strategy_mode == 'Fixed R:25% P:25% S:50%':
        bot_strategy = {
            'Rock': 0.25,
            'Paper': 0.25,
            'Scissors': 0.50
        }
    elif bot_strategy_mode == 'Always Scissors':
        bot_strategy = {
            'Rock': 0.0,
            'Paper': 0.0,
            'Scissors': 1.0
        }

    # Display Bot's Strategy
    strategy_df = pd.DataFrame.from_dict(bot_strategy, orient='index', columns=['Probability'])
    st.subheader("Bot's Strategy")
    st.table(strategy_df.style.format("{:.4f}"))

    # ================== Simulation Section ==================
    st.sidebar.header("🎲 Simulation Settings")
    sim_mode = st.sidebar.radio(
        "Simulation Mode",
        ('Manual Input', 'Random Trials'),
        help="Choose audience input method"
    )

    # Audience distribution handling
    if sim_mode == 'Manual Input':
        st.sidebar.subheader("Audience Distribution")
        rock = st.sidebar.number_input("Rock Choices", 0, 10000, 3333)
        paper = st.sidebar.number_input("Paper Choices", 0, 10000, 3333)
        scissors = st.sidebar.number_input("Scissors Choices", 0, 10000, 3334)
        total = rock + paper + scissors
        audience_dist = {
            'Rock': rock / total if total > 0 else 1 / 3,
            'Paper': paper / total if total > 0 else 1 / 3,
            'Scissors': scissors / total if total > 0 else 1 / 3
        }
    else:
        if st.sidebar.button("Generate 10,000 Random Trials"):
            audience_choices = np.random.choice(
                ['Rock', 'Paper', 'Scissors'],
                size=10000,
                p=list(opponent_strategy.values())
            )
            counts = pd.Series(audience_choices).value_counts()
            audience_dist = (counts / 10000).to_dict()
        else:
            audience_dist = opponent_strategy  # Fallback

    # Run simulation
    if st.sidebar.button("Run Simulation"):
        st.subheader("🎮 Simulation Results")

        # Generate choices
        bot_choices = np.random.choice(
            list(bot_strategy.keys()),
            size=10000,
            p=list(bot_strategy.values())
        )

        audience_choices = np.random.choice(
            list(audience_dist.keys()),
            size=10000,
            p=list(audience_dist.values())
        )

        # Calculate scores
        scores = [
            payoff_matrix[bot][audience][0]  # Bot's payoff
            for bot, audience in zip(bot_choices, audience_choices)
        ]
        cumulative_scores = np.cumsum(scores)

        # Results dataframe
        results = pd.DataFrame({
            'Bot Choice': bot_choices,
            'Audience Choice': audience_choices,
            'Score': scores,
            'Cumulative Score': cumulative_scores
        })

        # Visualizations
        st.write(f"**Total Score (Bot's Perspective):** {sum(scores)}")

        fig_cum = px.line(results, y='Cumulative Score',
                          title="Cumulative Score Over Time")
        st.plotly_chart(fig_cum, use_container_width=True)

        # Choice distribution comparison
        dist_df = pd.DataFrame({
            'Bot': pd.Series(bot_choices).value_counts(normalize=True),
            'Audience': pd.Series(audience_choices).value_counts(normalize=True)
        }).reset_index().melt(id_vars='index')

        fig_dist = px.bar(dist_df, x='index', y='value', color='variable', barmode='group',
                          labels={'index': 'Move', 'value': 'Frequency', 'variable': 'Player'},
                          title="Choice Distribution Comparison")
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()
