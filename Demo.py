import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import random


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
        ax.add_patch(
            plt.Rectangle((x_start, 0), section_width, goal_height, color=color, alpha=0.5))  # More transparency
        ax.text(x_start + section_width / 2, goal_height / 2, f"{norm_prob:.1%}", color="black", ha="center",
                va="center", fontsize=10, weight="bold")

    return fig  # Explicitly return the figure


# Configure the page
st.set_page_config(page_title="Game Theory Suite", layout="wide")
st.title("Game Theory Application Suite")

# Create tabs
introduction, prison, iesds, stop_light, sexes, penalty, take_or_share, rps, credits = st.tabs(
    ["Introduction", "Prisoner's dilema", "IESDS", "Stop light", "Battle of sexes", "Penalty Kick Analyzer",
     "Take vs. Share Dilemma", "Rock Paper Scissors", "Credits"])

# ====================================================================================
# Tab 1: Penalty Kick Analyzer
# ====================================================================================
with penalty:
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
    fig1.update_layout(title="Optimal Kick Strategy vs Effectiveness",
                       xaxis_title="X (% Effectiveness When Kicking Right)",
                       yaxis_title="Probability to Choose Left (%)", height=500)
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

# ====================================================================================
# Tab 2: Take vs. Share Dilemma
# ====================================================================================

with take_or_share:
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

with iesds:
    st.title("Game Theory Concepts: Strict Dominance and IESDS")

    st.header("Strict Dominance")
    st.write("""
        - Strategy 'x' strictly dominates strategy 'y' for a player if 'x' generates a greater payoff than 'y' regardless of what the other players do.
         - Rational players never play strictly dominated strategies. \n
         -- Why play 'y' when you can play 'x' instead?
    """)
    st.image(r"C:\Users\Gebruiker\PycharmProjects\learningchallange\GameTheoryApp\IESDS1.png")
    st.write(
        "Regardless of what strategy P2 chooses, it is always in the best interest of P1 to confess, as the payout is bigger in any case, therefore the 'Confess' strategy strictly dominates 'Keep Quiet'")

    st.header("Iteration Elimination of Strictly Dominated Strategies (IESDS)")
    st.write("""
        IESDS is a method where we iteratively remove strategies that are strictly dominated by other strategies,
        simplifying the game to find the optimal strategies for the players.
    """)
    st.image("IESDS2.png")
    st.write(" - If you ever see a strictly dominated strategy eliminate it immediately. \n - Order does not matter.")

# ------------------------------------------------------------------------------------
# Run with: streamlit run Demo.py
# ====================================================================================
# Tab 3: RPS Analyzer
# ====================================================================================
with rps:
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

# ------------------------------------------------------------------------------------
# Tab stop_light: Stop light game
# This game explains what are Nash equilibria using the stop game.
# ------------------------------------------------------------------------------------

with stop_light:
    st.title("Nash Equilibrium: Stoplight Game")
    st.write("""
             Picture a situation. Two cars are approaching an intersection.
             If they both crash into each other they will get significantly delayed.

             If they both stop, they will both wait, but most likely not for so long.


             If one can go and the only wait it's the best scenario for both of them because they are not wasting their time.

             This is represented below:
             """)
    st.image("images/stop_light.jpg")

    st.write(
        "Nash equilibrium is a law that everyone would want to follow even in the absence of an effective police force.")
    st.write("The payoffs represent the consequences of their decisions.")

    st.write("### Nash Equilibria")
    st.write("- **(Go, Stop)**: If Player 1 goes and Player 2 stops, neither has an incentive to change.")
    st.image("images/stop_light_go_stop.jpg")
    st.write(
        """
        If player 1 stops instead of going, player 1 gets **-1** < **1**

        If player 2 goes instead of stopping, player 2 gets **-5** < **0**
        """)

    st.write("- **(Stop, Go)**: If Player 1 stops and Player 2 goes, neither has an incentive to change.")
    st.image("images/stop_light_stop_go.jpg")
    st.write(
        """
        If player 1 goes instead of stopping, player 1 gets **-5** < **0**

        If player 2 stops instead of going, player 2 gets **-1** < **1**
        """)
    st.write("These are the two Nash equilibria of the game:")
    st.image("images/stop_light_equilibria.jpg")

# ------------------------------------------------------------------------------------
# Tab sexes: Battle of sexes
# This game explains what is a mixed strategy.
# ------------------------------------------------------------------------------------

with sexes:
    st.title("Battle of sexes")
    st.write("So far the strategy has only been pure. This means that the players will always stick to one strategy.")


def prisoners_dilemma():
    """
    Simulate the Prisoner's Dilemma game using Streamlit where the user plays against different AI strategies.
    """
    st.header("Prisoner's Dilemma Simulation")

    opponents = {
        "Always Keep Quiet": always_keep_quiet,
        "Always Confess": always_confess,
        "Tit for Tat": tit_for_tat,
        "Random": random_choice
    }

    col1, col2 = st.columns([2, 1])

    if "user_score" not in st.session_state:
        st.session_state.user_score = 0
        st.session_state.opponent_score = 0
        st.session_state.history = []

    with col1:
        opponent_name = st.selectbox("Choose an opponent:", list(opponents.keys()))
        opponent_func = opponents[opponent_name]

        col_btn1, col_btn2 = st.columns(2)

        if col_btn1.button("Keep Quiet", key="keep_quiet", use_container_width=True):
            user_move = "keep_quiet"
            opponent_move = opponent_func(st.session_state.history)
            user_points, opp_points = payoff(user_move, opponent_move)
            st.session_state.user_score += user_points
            st.session_state.opponent_score += opp_points
            st.session_state.history.append((user_move, opponent_move))

        if col_btn2.button("Confess", key="confess", use_container_width=True):
            user_move = "confess"
            opponent_move = opponent_func(st.session_state.history)
            user_points, opp_points = payoff(user_move, opponent_move)
            st.session_state.user_score += user_points
            st.session_state.opponent_score += opp_points
            st.session_state.history.append((user_move, opponent_move))

    with col2:
        st.subheader("Score & Payoff Matrix")
        st.metric(label="Your Score", value=st.session_state.user_score)
        st.metric(label="Opponent Score", value=st.session_state.opponent_score)

        payoff_matrix = pd.DataFrame({
            "You Keep Quiet": ["(-1,-1)", "(-12,0)"],
            "You Confess": ["(0,12)", "(-8,-8)"]
        }, index=["Opp. Keep Quiet", "Opp. Confess"])
        st.table(payoff_matrix)


def always_keep_quiet(history):
    return "keep_quiet"


def always_confess(history):
    return "confess"


def tit_for_tat(history):
    if not history:
        return "keep_quiet"
    return history[-1][0]


def random_choice(history):
    return random.choice(["keep_quiet", "confess"])


def payoff(player1, player2):
    if player1 == "keep_quiet" and player2 == "keep_quiet":
        return (-1, -1)
    elif player1 == "keep_quiet" and player2 == "confess":
        return (-12, 0)
    elif player1 == "confess" and player2 == "keep_quiet":
        return (0, 12)
    else:
        return (-8, -8)


with prison:
    prisoners_dilemma()

with introduction:
    st.title("Introduction")

    st.write(
        """
        ##### Game Theory is all around us—whether we realize it or not. From negotiations and traffic decisions to sports strategies and even everyday choices, the way we interact with others follows strategic patterns. Our app is designed to bring these concepts to life through interactive experiences that let you explore, play, and learn at your own pace.

        ##### With our app, you won’t just read about Game Theory—you’ll experience it. Dive into classic strategic dilemmas and test your decision-making skills with our interactive modules:
        
        ##### The Prisoner’s Dilemma – A classic example of why cooperation is hard, even when it's beneficial.
        
        ##### IESDS (Iterated Elimination of Strictly Dominated Strategies) – A method to predict rational choices in strategic games.
        
        ##### The Stoplight Game – A real-world application of Nash Equilibrium in traffic decisions.
        
        ##### The Battle of the Sexes – Exploring mixed strategies and payoffs in coordination problems.
        
        ##### Penalty Kick Analyzer – How professional athletes use mixed strategies in real-life competition.
        
        ##### Take vs. Share Dilemma – Examining the tension between selfishness and cooperation.
        
        ##### Rock, Paper, Scissors – A simple game with deeper strategic implications.
        """
    )