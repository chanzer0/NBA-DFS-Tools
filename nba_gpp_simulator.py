import json
import csv
import numpy as np

class NBA_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = []
    winning_lineups = []
    roster_construction = ['PG', 'SG', 'SF', 'PF', 'C', 'F', 'G']

    def __init__(self):
        self.load_config()
        self.load_projections(self.config['projection_path'])
        self.load_ownership(self.config['ownership_path'])
        self.load_boom_bust(self.config['boombust_path'])


    # Load config from file
    def load_config(self):
        with open('config.json') as json_file: 
            self.config = json.load(json_file)
         
    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name']
                self.player_dict[player_name] = {'Fpts': 0, 'Position': [], 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0, 'In Lineup': False}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])

                #some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                self.player_dict[player_name]['Position'] = [pos for pos in row['Position'].split('/')]

                if 'PG' in self.player_dict[player_name]['Position'] or 'SG' in self.player_dict[player_name]['Position']:
                    self.player_dict[player_name]['Position'].append('G')

                if 'SF' in self.player_dict[player_name]['Position'] or 'PF' in self.player_dict[player_name]['Position']:
                    self.player_dict[player_name]['Position'].append('F')

                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))
                # need to pre-emptively set ownership to 0 as some players will not have ownership
                # if a player does have ownership, it will be updated later on in load_ownership()
                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))
                
    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name']
                self.player_dict[player_name]['Ownership'] = float(row['Ownership %'])
    
    # Load standard deviations
    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name']
                if player_name in self.player_dict:
                    self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])

    def select_random_player(self, position):
        plyr_list = []
        prob_list = []
        for player in self.player_dict:
            if self.player_dict[player]['In Lineup'] == False:
                if position in self.player_dict[player]['Position']:
                    plyr_list.append(player)
                    prob_list.append(self.player_dict[player]['Ownership'])
                    
        prob_list = [float(i)/sum(prob_list) for i in prob_list]
        return np.random.choice(a=plyr_list, p=prob_list)

    def generate_field_lineups(self, num_lineups):
        for i in range(int(num_lineups)):
            reject = True
            while reject:
                salary = 0
                lineup = []
                for player in self.player_dict:
                    self.player_dict[player]['In Lineup'] = False
                for pos in self.roster_construction:
                    x = self.select_random_player(pos)
                    self.player_dict[x]['In Lineup'] = True
                    lineup.append(x)
                    salary += self.player_dict[x]['Salary']
                if (47000 <= salary <= 50000):
                    reject= False
            self.field_lineups.append(lineup)
        print(num_lineups + ' field lineups successfully generated')

    def run_tournament_simulation(self, num_iterations):
        for i in range(int(num_iterations)):
            temp_fpts_dict = {p: round((np.random.normal(stats['Fpts'], stats['StdDev'])), 2) for p,stats in self.player_dict.items()}
            field_score = {}

            for lineup in self.field_lineups:
                fpts_sim = sum(temp_fpts_dict[player] for player in lineup)
                field_score[fpts_sim] = lineup

            winning_lineup = max(field_score, key=float)
            self.winning_lineups.append(field_score[winning_lineup])

        print(num_iterations + ' tournament simulations finished')
        print(self.winning_lineups)

            
