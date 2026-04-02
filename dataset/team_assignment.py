import pandas as pd

# Load the main dataset
df = pd.read_csv('final_ball_by_ball_first_innings.csv')

# 1. True Team Mapping: Identify the team each player actually plays for (when they bat)
# We find the most frequent team a player appeared for as a BATTER.
player_to_team = df.groupby('batter')['team'].agg(lambda x: x.value_counts().index[0]).to_dict()

# 2. Advanced Inference for Bowlers who never bat (Optional but helpful)
# If a bowler never bats, we look at their teammates. 
# If their teammate bats for India, they are also India.
match_bowlers = df.groupby('match_id')['bowler'].unique().to_dict()
for _ in range(3): # Propagate teammates' teams
    for bowlers in match_bowlers.values():
        known_team = next((player_to_team[b] for b in bowlers if b in player_to_team), None)
        if known_team:
            for b in bowlers:
                if b not in player_to_team: player_to_team[b] = known_team

# 3. Apply to Stats Files
bat_stats = pd.read_csv('batter_stats.csv')
bowl_stats = pd.read_csv('bowler_stats.csv')

bat_stats['team'] = bat_stats['batter'].map(player_to_team).fillna("Other")
bowl_stats['team'] = bowl_stats['bowler'].map(player_to_team).fillna("Other")

# Save as V3 (The Clean Version)
bat_stats.to_csv('batter_stats_v2.csv', index=False)
bowl_stats.to_csv('bowler_stats_v2.csv', index=False)
print("CSVs Corrected! Bowlers are now assigned to their own teams.")