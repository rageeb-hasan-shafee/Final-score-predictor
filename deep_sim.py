import pandas as pd
import numpy as np
import random

# =====================================================================
# LAYER 1: PREDICTION LAYER
# =====================================================================
class PredictionLayer:
    def __init__(self, df):
        self.df = df.copy()
        # The exact outcomes requested: 0, 1, 2, 3, 4, 5, 6, and wicket
        self.outcomes = [0, 1, 2, 3, 4, 5, 6, 'W']
        self._prepare_data()

    def _prepare_data(self):
        # Create an 'outcome' column representing the exact event on the ball
        self.df['outcome'] = self.df.apply(
            lambda x: 'W' if x['wicket_fallen'] == 1 else x['runs_batter'], axis=1
        )
        
        # Calculate historical empirical probabilities
        self.global_probs = self._calc_probs([])
        self.bat_probs = self._calc_probs(['batter'])
        self.bowl_probs = self._calc_probs(['bowler'])
        self.matchup_probs = self._calc_probs(['batter', 'bowler'])

    def _calc_probs(self, keys):
        """Calculates probabilities from frequency. Returns dict or dataframe of dicts."""
        if not keys:
            counts = self.df['outcome'].value_counts()
            total = counts.sum()
            return {o: counts.get(o, 0) / total for o in self.outcomes}
            
        grouped = self.df.groupby(keys + ['outcome']).size().unstack(fill_value=0)
        # Convert absolute counts to probabilities
        probs_df = grouped.div(grouped.sum(axis=1), axis=0)
        
        # Ensure all outcome columns exist
        for o in self.outcomes:
            if o not in probs_df.columns:
                probs_df[o] = 0.0
                
        return probs_df

    def get_probabilities(self, batter, bowler):
        """
        Returns a probability distribution array aligned with self.outcomes.
        Uses a fallback system: Matchup -> Batter -> Bowler -> Global.
        """
        if (batter, bowler) in self.matchup_probs.index:
            probs_dict = self.matchup_probs.loc[(batter, bowler)].to_dict()
        elif batter in self.bat_probs.index:
            probs_dict = self.bat_probs.loc[batter].to_dict()
        elif bowler in self.bowl_probs.index:
            probs_dict = self.bowl_probs.loc[bowler].to_dict()
        else:
            probs_dict = self.global_probs

        return [probs_dict.get(o, 0.0) for o in self.outcomes]

# =====================================================================
# LAYER 2: SIMULATION LAYER
# =====================================================================
class SimulationLayer:
    def __init__(self, prediction_layer):
        self.pred_layer = prediction_layer
        self.outcomes = prediction_layer.outcomes

    def simulate(self, initial_state, lineup, available_bowlers, num_sims=1000):
        final_scores = []

        for _ in range(num_sims):
            # 1. Create a fresh copy of state for this specific simulation iteration
            state = {
                'score': initial_state['score'],
                'wickets': initial_state['wickets'],
                'over': initial_state['over'],
                'ball': initial_state['ball'],
                'striker': initial_state['striker'],
                'non_striker': initial_state['non_striker'],
                'last_bowler': initial_state['last_bowler'],
                'bowler_overs': initial_state['bowler_overs'].copy()
            }

            # 2. Setup Batting order index tracking
            try:
                next_bat_idx = max(lineup.index(state['striker']), lineup.index(state['non_striker'])) + 1
            except ValueError:
                next_bat_idx = 2

            curr_over = state['over']
            curr_ball = state['ball']
            
            # 3. Simulate ball by ball until 20 overs or 10 wickets
            while curr_over < 20 and state['wickets'] < 10:
                
                # --- Bowler Selection (Start of a new over) ---
                if curr_ball == 1:
                    # Bowler cannot bowl more than 4 overs and cannot bowl consecutive overs
                    valid_bowlers = [
                        b for b in available_bowlers 
                        if state['bowler_overs'].get(b, 0) < 4 and b != state['last_bowler']
                    ]
                    
                    # Fallback rules in case all valid bowlers are exhausted
                    if not valid_bowlers:
                        valid_bowlers = [b for b in available_bowlers if state['bowler_overs'].get(b, 0) < 4]
                    if not valid_bowlers:
                        valid_bowlers = available_bowlers

                    current_bowler = np.random.choice(valid_bowlers)
                    state['last_bowler'] = current_bowler
                    state['bowler_overs'][current_bowler] = state['bowler_overs'].get(current_bowler, 0) + 1
                else:
                    current_bowler = state['last_bowler']

                # --- Ball Prediction ---
                p_values = self.pred_layer.get_probabilities(state['striker'], current_bowler)
                
                # Normalize probabilities just in case (handles sum > 0 rounding errors)
                sum_p = sum(p_values)
                if sum_p > 0:
                    p_values = [p / sum_p for p in p_values]
                else:
                    p_values = [1/len(self.outcomes)] * len(self.outcomes)

                # Randomly pick an outcome based on historical probability distribution
                outcome = np.random.choice(self.outcomes, p=p_values)

                # --- Update Match State based on outcome ---
                if outcome == 'W':
                    state['wickets'] += 1
                    # New batsman comes in if wickets < 10
                    if state['wickets'] < 10 and next_bat_idx < len(lineup):
                        state['striker'] = lineup[next_bat_idx]
                        next_bat_idx += 1
                else:
                    runs = int(outcome)
                    state['score'] += runs
                    # Rule: Swap strike on 1s, 3s, 5s
                    if runs in [1, 3, 5]:
                        state['striker'], state['non_striker'] = state['non_striker'], state['striker']

                # --- Advance Ball & Over State ---
                curr_ball += 1
                if curr_ball > 6:
                    curr_over += 1
                    curr_ball = 1
                    # Rule: Swap strike at the end of the over
                    state['striker'], state['non_striker'] = state['non_striker'], state['striker']

            final_scores.append(state['score'])

        # Return the average prediction after 1000 match simulations
        return np.mean(final_scores)

# =====================================================================
# MAIN EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    print("Loading data...")
    # 1. Load your dataset
    df = pd.read_csv('final_ball_by_ball (1).csv')

    # 2. Build Required Lists
    # Create unified player list (union of batters, bowlers, non_strikers)
    player_list = list(
        set(df['batter'].unique()) | 
        set(df['bowler'].unique()) | 
        set(df['non_striker'].unique())
    )
    # Create Match list
    match_list = list(df['match_id'].unique())
    
    print(f"Dataset summary: {len(player_list)} unique players, {len(match_list)} matches.")

    # 3. Initialize Model Layers
    prediction_layer = PredictionLayer(df)
    simulator = SimulationLayer(prediction_layer)

    # 4. Define Current Match Input
    # Example state: Assume it is England's Innings in Match 211028 after exactly 10 overs
    initial_match_state = {
        'score': 93, 
        'wickets': 2,
        'over': 10,   # 10 overs completed. The simulation will start from Over 10, Ball 1 (11th over)
        'ball': 1,
        'striker': 'KP Pietersen',
        'non_striker': 'A Flintoff',
        'last_bowler': 'A Symonds', 
        # Number of overs each bowler has completed so far:
        'bowler_overs': {'B Lee': 2, 'GD McGrath': 2, 'MS Kasprowicz': 2, 'SR Watson': 2, 'A Symonds': 2}
    }

    lineup = ['ME Trescothick', 'GO Jones', 'A Flintoff', 'KP Pietersen', 'PD Collingwood', 'A Shah', 'SJ Harmison']
    available_bowlers = ['B Lee', 'GD McGrath', 'MS Kasprowicz', 'SR Watson', 'A Symonds']

    # 5. Run 1000 Simulations
    print(f"\nCurrent Score: {initial_match_state['score']}/{initial_match_state['wickets']} in {initial_match_state['over']} overs.")
    print("Running 1000 Monte Carlo Simulations. Please wait...")
    
    predicted_average_score = simulator.simulate(
        initial_match_state, 
        lineup, 
        available_bowlers, 
        num_sims=1000
    )

    print(f"\n--- PREDICTION RESULTS ---")
    print(f"Predicted Final Average Score: {predicted_average_score:.2f} runs")