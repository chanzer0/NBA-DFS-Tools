import json
import csv
import os
import datetime
import pytz
import numpy as np
import pulp as plp
from dateutil import parser
from dateutil.tz import UTC
import random


class NBA_Late_Swaptimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    team_list = []
    live_lineups = {}
    swapped_lineups = {}
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    global_team_limit = None
    projection_minimum = 0
    randomness_amount = 0
    dk_positionality = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    def __init__(self, site=None):
        self.site = site

        if (self.site == 'fd'):
            print('FanDuel lateswap not yet supported.')
            quit()

        self.load_config()
        self.load_rules()

        self.problem = plp.LpProblem('NBA', plp.LpMaximize)

        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        ownership_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['ownership_path']))
        self.load_ownership(ownership_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

        boom_bust_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        self.load_boom_bust(boom_bust_path)

        live_lineups_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['late_swap_path']))
        self.load_lineups_dk(live_lineups_path)

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json'), encoding='utf-8-sig') as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                if player_name in self.player_dict:

                    if self.site == 'dk':
                        self.player_dict[player_name]['RealID'] = int(
                            row['ID'])
                        self.player_dict[player_name]['ID'] = int(
                            row['ID'][-3:])
                        self.player_dict[player_name]['Matchup'] = row['Game Info'].split(' ')[
                            0]
                        self.player_dict[player_name]['Start Time'] = row['Game Info'].split(
                            ' ', 1)[1]
                    else:
                        self.player_dict[player_name]['RealID'] = str(
                            row['Id'])
                        self.player_dict[player_name]['ID'] = int(
                            row['Id'].split('-')[1])
                        self.player_dict[player_name]['Matchup'] = row['Game'].split(' ')[
                            0]

    # Instantiate config rules
    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]

    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    # if this player has a very low chance of reaching a GPP target score, do not play them
                    if float(row['Boom%']) < 0.0:
                        del self.player_dict[player_name]
                        continue
                    self.player_dict[player_name]['Leverage'] = float(
                        row['Leverage'])
                    self.player_dict[player_name]['StdDev'] = float(
                        row['Std Dev'])
                    self.player_dict[player_name]['Boom'] = float(row['Boom%'])
                    self.player_dict[player_name]['Bust'] = float(row['Bust%'])
                    self.player_dict[player_name]['Ceiling'] = float(
                        row['Projection']) + float(row['Std Dev'])
                    if row['Optimal%'] != '':
                        self.player_dict[player_name]['Optimal'] = float(
                            row['Optimal%'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if float(row['Fpts']) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {'Fpts': 0.1, 'Position': None, 'ID': 0, 'Salary': 1000, 'Name': '', 'RealID': 0, 'Matchup': '', 'Leverage': 0,
                                                 'StdDev': 0.1, 'Team': '', 'Ownership': 0.1, 'Optimal': 0.1, 'Minutes': 1, 'Boom': 0.1, 'Bust': 0.1, 'Ceiling': 0.1}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(
                    row['Salary'].replace(',', ''))
                self.player_dict[player_name]['Minutes'] = float(
                    row['Minutes'])
                self.player_dict[player_name]['Name'] = row['Name']
                self.player_dict[player_name]['Team'] = row['Team']
                self.player_dict[player_name]['FPPM'] = float(
                    row['Fpts']) / float(row['Minutes'])
                self.player_dict[player_name]['PPD'] = float(
                    row['Fpts']) / int(row['Salary'].replace(',', ''))*1000
                if row['Team'] not in self.team_list:
                    self.team_list.append(row['Team'])

                # Need to handle MPE
                self.player_dict[player_name]['Position'] = [
                    pos for pos in row['Position'].split('/')]
                if ('PG' in self.player_dict[player_name]['Position'] or 'SG' in self.player_dict[player_name]['Position']):
                    self.player_dict[player_name]['Position'].append('G')
                if ('PF' in self.player_dict[player_name]['Position'] or 'SF' in self.player_dict[player_name]['Position']):
                    self.player_dict[player_name]['Position'].append('F')
                self.player_dict[player_name]['Position'].append('UTIL')
                print(player_name, self.player_dict[player_name]['Position'])

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(
                        row['Ownership %'])

    def load_lineups_dk(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if (row['Entry ID']):
                    self.live_lineups[row['Entry ID']] = [
                        row['PG'], row['SG'], row['SF'], row['PF'], row['C'], row['G'], row['F'], row['UTIL']]

    def swaptimize(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        if self.site == 'dk':
            i = 0
            for entry_id, lineup in self.live_lineups.items():
                print(f'Late swapping lineup #{i + 1}')
                print(entry_id, lineup)
                locked_players = {}
                unlocked_players = {}
                available_salary = 50000
                for idx, player in enumerate(lineup):
                    player_name = player.split('(')[0].strip()
                    start_time = self.player_dict[player_name]['Start Time']
                    parsed = parser.parse(start_time, tzinfos={
                                          'ET': -5 * 3600}).astimezone(UTC)

                    # if this game has already started
                    if (idx > 6):  # bool(random.getrandbits(1))):
                        parsed = now

                    if (parsed <= now):
                        locked_players[self.dk_positionality[idx]
                                       ] = player_name
                        available_salary -= self.player_dict[player_name]['Salary']
                    else:
                        unlocked_players[self.dk_positionality[idx]
                                         ] = player_name

                print(
                    f'locked players: {len(locked_players.keys())} {locked_players}')
                print(
                    f'available to swap: {len(unlocked_players.keys())} {unlocked_players}')
                print(f'remaining salary: {available_salary}')

                # Instantiate opto rules based on locked players, available positions and remaining salary
                # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
                # We will use PuLP as our solver - https://coin-or.github.io/pulp/

                # We want to create a variable for each roster slot.
                # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
                lp_variables = {player: plp.LpVariable(
                    player, cat='Binary') for player, _ in self.player_dict.items()}

                # set the objective - maximize fpts & set randomness amount from config
                self.problem += plp.lpSum(np.random.normal(self.player_dict[player]['Fpts'],
                                                           (self.player_dict[player]['StdDev'] * self.randomness_amount / 100))
                                          * lp_variables[player] for player in self.player_dict), 'Objective'

                # Address limit rules if any
                for limit, groups in self.at_least.items():
                    for group in groups:
                        self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                                  for player in group) >= int(limit)

                for limit, groups in self.at_most.items():
                    for group in groups:
                        self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                                  for player in group) <= int(limit)

                for matchup, limit in self.matchup_limits.items():
                    self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                              for player in self.player_dict if self.player_dict[player]['Matchup'] == matchup) <= int(limit)

                for matchup, limit in self.matchup_at_least.items():
                    self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                              for player in self.player_dict if self.player_dict[player]['Matchup'] == matchup) >= int(limit)

                # Address team limits
                for team, limit in self.team_limits.items():
                    self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                              for player in self.player_dict if self.player_dict[player]['Team'] == team) <= int(limit)
                if self.global_team_limit is not None:
                    for team in self.team_list:
                        self.problem += plp.lpSum(lp_variables[player]
                                                  for player in self.player_dict if self.player_dict[player]['Team'] == team) <= int(self.global_team_limit)

                # Remaining Salary
                self.problem += plp.lpSum(self.player_dict[player.replace('-', '#')]['Salary'] *
                                          lp_variables[player] for player in self.player_dict) <= available_salary

                # Locked players
                self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                          for player in locked_players.values()) <= 0

                # Positions
                for position in unlocked_players:
                    self.problem += plp.lpSum(lp_variables[player]
                                              for player in self.player_dict if position in self.player_dict[player]['Position']) >= 1
                    self.problem += plp.lpSum(lp_variables[player]
                                              for player in self.player_dict if position in self.player_dict[player]['Position']) <= 1

                # Need at least 1 of each position unlocked
                self.problem += plp.lpSum(lp_variables[player] for player in self.player_dict if not set(
                    unlocked_players.keys()).isdisjoint(self.player_dict[player]['Position'])) >= len(unlocked_players.keys())

                self.problem += plp.lpSum(lp_variables[player] for player in self.player_dict if not set(
                    unlocked_players.keys()).isdisjoint(self.player_dict[player]['Position'])) <= len(unlocked_players.keys())

                # Num Players
                self.problem += plp.lpSum(lp_variables[player]
                                          for player in self.player_dict) >= len(unlocked_players.keys())
                self.problem += plp.lpSum(lp_variables[player]
                                          for player in self.player_dict) <= len(unlocked_players.keys())
                self.problem += plp.lpSum(lp_variables[player]
                                          for player in self.player_dict) == len(unlocked_players.keys())

                # Solve!
                print(self.problem)
                try:
                    self.problem.solve(plp.PULP_CBC_CMD(msg=0))
                except plp.PulpSolverError:
                    print('An error occured :( contact dev pls')

                score = str(self.problem.objective)
                for v in self.problem.variables():
                    score = score.replace(v.name, str(v.varValue))

                player_names = [v.name.replace(
                    '_', ' ') for v in self.problem.variables() if v.varValue != 0]

                salary = 0
                for player in player_names:
                    print(self.player_dict[player]['Position'])
                    salary += self.player_dict[player]['Salary']

                print(available_salary, salary)
                print(f'{len(player_names)} {player_names}')
                fpts = eval(score)
                # print(fpts)
                self.swapped_lineups[entry_id] = player_names
                i += 1
