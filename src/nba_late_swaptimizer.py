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

        if self.site == 'dk':
            self.load_live_dk_lineups(lineup_path)
        else:
            self.load_live_fd_lineups(lineup_path)

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
    def load_live_dk_lineups(self, path):
        # Read live lineups into a dictionary
        lineups = {}
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Lineups are in here
                if row['Entry ID'] != '':
                    lineups[row['Entry ID']] = {
                        'PG': (row['PG'][:-11],0), 
                        'SG': (row['SG'][:-11],0), 
                        'SF': (row['SF'][:-11],0), 
                        'PF': (row['PF'][:-11],0), 
                        'C': (row['C'][:-11],0), 
                        'G': (row['G'][:-11],0), 
                        'F': (row['F'][:-11],0), 
                        'UTIL': (row['UTIL'][:-11],0)
                    }

                # Player start times are in here
                elif None in row:
                    player_name = row[None][1]
                    game_info = row[None][5]
                    if player_name in self.player_dict:
                        start_time = game_info.split(' ', 1)[-1][:-5]
                        date_obj = datetime.datetime.strptime(start_time, '%m/%d/%Y %H:%M')
                        # times are parsed as AM, but are actually PM - i.e. 0700 (AM) needs to be 1900 (700 PM), thus add 12 hours
                        self.player_dict[player_name]['Start Time'] = date_obj + datetime.timedelta(hours=12) 
        return lineups

    def load_live_fd_lineups(self, path):
        # Read live lineups into a dictionary
        lineups = {}
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                pass


    def swaptimize(self):
        lineup_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(self.site, self.config['late_swap_path']))
        live_lineups = self.load_live_lineups(lineup_path)
        print(live_lineups)

        # Need to indicate whether or not a player is "locked" or "swappable" (1 and 0, respectively)
        # Current time in EST, since thats what DK uses in their files
        curr_time = datetime.datetime.now(pytz.timezone('EST'))
        for entry_id,players in live_lineups.items():
            print(players['PG'])
            for pos in players:
                player_name = players[pos][0]
                if self.player_dict[player_name]['Start Time'] is not None:
                    player_start_time = pytz.timezone('EST').localize(self.player_dict[player_name]['Start Time'])
                    # If their game has already started, we need to "lock" them
                    if curr_time > player_start_time:
                        players[pos] = (players[pos][0], 1)

            
