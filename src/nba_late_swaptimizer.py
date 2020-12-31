import json, csv, os
import numpy as np
from pulp import *

class NBA_Late_Swaptimizer:
    site = None
    config = None
    player_dict = {}
    live_lineups = []

    def __init__(self, site):
        self.site = site
        self.load_config()
        self.problem = LpProblem('NBA', LpMaximize)

        projection_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        player_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

        boom_bust_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        self.load_boom_bust(boom_bust_path)

        lineup_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['late_swap_path']))
        self.load_lineups(lineup_path)

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file: 
            self.config = json.load(json_file) 

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                if player_name in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[player_name]['ID'] = int(row['ID'])
                    else:
                        self.player_dict[player_name]['ID'] = row['Id']


    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                self.player_dict[player_name] = {'Fpts': 0, 'Position': None, 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Game Time': None}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))

                # Need to handle MPE on draftkings
                if self.site == 'dk':
                    self.player_dict[player_name]['Position'] = [pos for pos in row['Position'].split('/')]
                else:
                    self.player_dict[player_name]['Position'] = row['Position']

    # Load the late-swap lineups we want to alter
    def load_lineups(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.live_lineups.append([player for player in row])

        print(self.live_lineups)


    def swaptimize(self):
        swaptimized_lineups = []
        for lineup in self.live_lineups:
            lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}
            for player in lp_variables:
                # If player is in our lineup, and it has already locked, we cannot change him
                if player in lineup:

                    player.setInitialValue(1)
                    player.fixValue()
            # Need 2 PG
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' == self.player_dict[player]['Position']) == 2
            # Need 2 SG
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' == self.player_dict[player]['Position']) == 2
            # Need 2 SF
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' == self.player_dict[player]['Position']) == 2
            # Need 2 PF
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' == self.player_dict[player]['Position']) == 2
            # Need 1 center
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' == self.player_dict[player]['Position']) == 1
            # Can only roster 9 total players
            self.problem += lpSum(lp_variables[player] for player in self.player_dict) == 9
