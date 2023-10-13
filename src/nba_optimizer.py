import json
import csv
import os
import datetime
import numpy as np
import pulp as plp
from random import shuffle, choice
import itertools

class NBA_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = []
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    global_team_limit = None
    projection_minimum = None
    randomness_amount = 0
    min_salary = None

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.load_config()
        self.load_rules()

        self.problem = plp.LpProblem('NBA', plp.LpMaximize)

        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json'), encoding='utf-8-sig') as json_file:
            self.config = json.load(json_file)
            
    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                team = row['TeamAbbrev'] if self.site == 'dk' else row['Team']
                position = row['Position']
                if (player_name, position, team) in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[(player_name, position, team)]['ID'] = int(row['ID'])
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game Info'].split(' ')[0]
                        self.player_dict[(player_name, position, team)]['GameTime'] = ' '.join(row['Game Info'].split()[1:])
                    else:
                        print(row)
                        self.player_dict[(player_name, position, team)]['ID'] = row['Id'].replace('-', '#')
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game']
        
    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]
        self.min_salary = int(self.config["min_lineup_salary"])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row['name'].replace('-', '#')
                if float(row['fpts']) < self.projection_minimum:
                    continue
                
                position = row['position']
                team = row['team']
                self.player_dict[(player_name, position, team)] = {
                    'Fpts': float(row['fpts']),
                    'Salary': int(row['salary'].replace(',', '')),
                    'Minutes': float(row['minutes']),
                    'Name': row['name'],
                    'Team': row['team'],
                    'Ownership': float(row['own%']),
                    'StdDev': float(row['stddev']),
                    'Position': [pos for pos in row['position'].split('/')],
                }
                if row['team'] not in self.team_list:
                    self.team_list.append(row['team'])
                
    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        for (player, pos_str, team) in self.player_dict:
            print((player, pos_str, team), self.player_dict[(player, pos_str, team)])
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["ID"]), cat="Binary"
            )
            for (player, pos_str, team) in self.player_dict
        }
        
        
        # set the objective - maximize fpts & set randomness amount from config
        if self.randomness_amount != 0:
            self.problem += (
                plp.lpSum(
                    np.random.normal(
                        self.player_dict[(player, pos_str, team)]["Fpts"],
                        (
                            self.player_dict[(player, pos_str, team)]["StdDev"]
                            * self.randomness_amount
                            / 100
                        ),
                    )
                    * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                ),
                "Objective",
            )
        else:
            self.problem += (
                plp.lpSum(
                    self.player_dict[(player, pos_str, team)]["Fpts"]
                    * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                ),
                "Objective",
            )
        
        
        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 49000 if self.site == 'dk' else 59000
        
        if self.projection_minimum is not None:
            min_salary = self.min_salary
        
        self.problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= max_salary,
            "Max Salary",
        )
        self.problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            >= min_salary,
            "Min Salary",
        )

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                tuple_name_list = []
                for key, value in self.player_dict.items():
                    if value["Name"] in group:
                        tuple_name_list.append(key)

                self.problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in tuple_name_list
                    )
                    >= int(limit),
                    f"At least {limit} players {tuple_name_list}",
                )

        for limit, groups in self.at_most.items():
            for group in groups:
                tuple_name_list = []
                for key, value in self.player_dict.items():
                    if value["Name"] in group:
                        tuple_name_list.append(key)

                self.problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in tuple_name_list
                    )
                    <= int(limit),
                    f"At most {limit} players {tuple_name_list}",
                )

        for matchup, limit in self.matchup_limits.items():
            tuple_name_list = []
            for key, value in self.player_dict.items():
                if value["Matchup"] == matchup:
                    tuple_name_list.append(key)
                    
            self.problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]] for (player, pos_str, team) in tuple_name_list) <= int(limit), "At most {} players from {}".format(limit, matchup)

        for matchup, limit in self.matchup_at_least.items():
            tuple_name_list = []
            for key, value in self.player_dict.items():
                if value["Matchup"] == matchup:
                    tuple_name_list.append(key)
                    
            self.problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]] for (player, pos_str, team) in tuple_name_list) >= int(limit), "At least {} players from {}".format(limit, matchup)

        # Address team limits
        for teamIdent, limit in self.team_limits.items():
            self.problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                      for (player, pos_str, team) in self.player_dict if team == teamIdent) <= int(limit), "At most {} players from {}".format(limit, teamIdent)
            
        if self.global_team_limit is not None:
            for teamIdent in self.team_list:
                self.problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                          for (player, pos_str, team) in self.player_dict if team == teamIdent) <= int(self.global_team_limit), "Global team limit - at most {} players from {}".format(self.global_team_limit, teamIdent)

        if self.site == 'dk':
            # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
            point_guards = [player for player in self.player_dict if 'PG' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in point_guards) >= 1
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in point_guards) <= 3
            
            # Need at least 1 shooting guard, can have up to 3 if utilizing G and UTIL slots
            shooting_guards = [player for player in self.player_dict if 'SG' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in shooting_guards) >= 1
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in shooting_guards) <= 3
            
            # Need at least 1 small forward, can have up to 3 if utilizing F and UTIL slots
            small_forwards = [player for player in self.player_dict if 'SF' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in small_forwards) >= 1
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in small_forwards) <= 3
            
            # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
            power_forwards = [player for player in self.player_dict if 'PF' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in power_forwards) >= 1
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in power_forwards) <= 3
            
            # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
            centers = [player for player in self.player_dict if 'C' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in centers) >= 1
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in centers) <= 2
            
            # Need at least 3 guards (PG,SG,G)
            guards = [player for player in self.player_dict if 'PG' in self.player_dict[player]['Position'] or 'SG' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in guards) >= 3
            
            # Need at least 3 forwards (SF,PF,F)
            forwards = [player for player in self.player_dict if 'SF' in self.player_dict[player]['Position'] or 'PF' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in forwards) >= 3
            
            # Max 4 of PG and C single-eligbility players to prevent rare edge case where you can end up with 3 PG and 2 C, even though this build is infeasible
            point_guards_and_centers = [player for player in self.player_dict if 'PG' in self.player_dict[player]['Position'] or 'C' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in point_guards_and_centers) <= 4
            
            # Can only roster 8 total players
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in self.player_dict) == 8
        else:
            # PG MPE
            point_guards = [player for player in self.player_dict if 'PG' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in point_guards) >= 2
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in point_guards) <= 5
            
            # SG MPE
            shooting_guards = [player for player in self.player_dict if 'SG' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in shooting_guards) >= 2
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in shooting_guards) <= 5
            
            # SF MPE
            small_forwards = [player for player in self.player_dict if 'SF' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in small_forwards) >= 2
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in small_forwards) <= 5
            
            # PF MPE
            power_forwards = [player for player in self.player_dict if 'PF' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in power_forwards) >= 2
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in power_forwards) <= 5
            
            # C MPE
            centers = [player for player in self.player_dict if 'C' in self.player_dict[player]['Position']]
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in centers) >= 1
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in centers) <= 3

            # PG Alignment
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in point_guards) <= 2

            # SG Alignment
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in shooting_guards) <= 2

            # SF Alignment
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in small_forwards) <= 2

            # PF Alignment
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in power_forwards) <= 2

            # C Alignment
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]] for player in centers) <= 1

            # Can only roster 9 total players
            self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]]for player in self.player_dict) == 9
            
            # Max 4 per team
            for team in self.team_list:
                self.problem += plp.lpSum(lp_variables[self.player_dict[player]["ID"]]
                                          for player in self.player_dict if self.player_dict[player]['Team'] == team) <= 4

        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.num_lineups), self.num_lineups))

            # Get the lineup and add it to our list
            player_ids = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]
            players = []
            for key, value in self.player_dict.items():
                if value["ID"] in player_ids:
                    players.append(key)

            fpts_used = self.problem.objective.value()
            print(fpts_used, players)
            self.lineups.append((fpts_used, players))
            
            if i % 100 == 0:
                print(i)

            # Ensure this lineup isn't picked again
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[player]["ID"]] for player in players
                )
                <= len(players) - self.num_uniques,
                f"Lineup {i}",
            )

            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += (
                    plp.lpSum(
                        np.random.normal(
                            self.player_dict[(player, pos_str, team)]["Fpts"],
                            (
                                self.player_dict[(player, pos_str, team)]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        )
                        * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                    ),
                    "Objective",
                )

    def output(self):
        print('Lineups done generating. Outputting.')
       
        sorted_lineups = []
        for fpts, lineup in self.lineups:
            print(lineup)
            sorted_lineup = self.sort_lineup_dk(lineup)
            print(sorted_lineup)
            late_player_adjusted_lineup = self.adjust_roster_for_late_swap_dk(sorted_lineup)
            print(late_player_adjusted_lineup)
            sorted_lineups.append(sorted_lineup)

        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_optimal_lineups.csv'.format(self.site))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write(
                    'PG,SG,SF,PF,C,G,F,UTIL,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for fpts, x in self.lineups.items():
                    # print(id_sum, tple)
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    own_s = sum(self.player_dict[player]['Ownership'] for player in x)
                    mins = sum([self.player_dict[player]
                               ['Minutes'] for player in x])
                    stddev = sum(
                        [self.player_dict[player]['StdDev'] for player in x])
                    # print(sum(self.player_dict[player]['Ownership'] for player in x))
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{}'.format(
                        x[0].replace(
                            '#', '-'), self.player_dict[x[0]]['RealID'],
                        x[1].replace(
                            '#', '-'), self.player_dict[x[1]]['RealID'],
                        x[2].replace(
                            '#', '-'), self.player_dict[x[2]]['RealID'],
                        x[3].replace(
                            '#', '-'), self.player_dict[x[3]]['RealID'],
                        x[4].replace(
                            '#', '-'), self.player_dict[x[4]]['RealID'],
                        x[5].replace(
                            '#', '-'), self.player_dict[x[5]]['RealID'],
                        x[6].replace(
                            '#', '-'), self.player_dict[x[6]]['RealID'],
                        x[7].replace(
                            '#', '-'), self.player_dict[x[7]]['RealID'],
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write(
                    'PG,PG,SG,SG,SF,SF,PF,PF,C,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for fpts, x in self.lineups.items():
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    own_s = sum(self.player_dict[player]['Ownership'] for player in x)
                    mins = sum([self.player_dict[player]
                               ['Minutes'] for player in x])
                    stddev = sum(
                        [self.player_dict[player]['StdDev'] for player in x])
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['RealID'], x[0].replace(
                            '#', '-'),
                        self.player_dict[x[1]]['RealID'], x[1].replace(
                            '#', '-'),
                        self.player_dict[x[2]]['RealID'], x[2].replace(
                            '#', '-'),
                        self.player_dict[x[3]]['RealID'], x[3].replace(
                            '#', '-'),
                        self.player_dict[x[4]]['RealID'], x[4].replace(
                            '#', '-'),
                        self.player_dict[x[5]]['RealID'], x[5].replace(
                            '#', '-'),
                        self.player_dict[x[6]]['RealID'], x[6].replace(
                            '#', '-'),
                        self.player_dict[x[7]]['RealID'], x[7].replace(
                            '#', '-'),
                        self.player_dict[x[8]]['RealID'], x[8].replace(
                            '#', '-'),
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
        print('Output done.')


    def sort_lineup_dk(self, lineup):
        order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        sorted_lineup = [None] * 8

        def find_and_remove(positions):
            for pos in positions:
                for player in lineup:
                    if player[1] == pos:
                        lineup.remove(player)
                        return player
            return None
        
        # Fill definite positions first
        for index, pos in enumerate(order[:-3]):
            player = find_and_remove([pos])
            if player:
                sorted_lineup[index] = player

        # Fill dual positions
        if sorted_lineup[order.index('SG')] is None:
            sg_sf_player = find_and_remove(['SG/SF'])
            if sg_sf_player:
                sorted_lineup[order.index('SG')] = sg_sf_player

        # Fill G and F positions
        if sorted_lineup[order.index('G')] is None:
            g_player = find_and_remove(['PG', 'SG'])
            if g_player:
                sorted_lineup[order.index('G')] = g_player

        if sorted_lineup[order.index('F')] is None:
            f_player = find_and_remove(['SF', 'PF'])
            if f_player:
                sorted_lineup[order.index('F')] = f_player

        # Fill UTIL position
        if sorted_lineup[order.index('UTIL')] is None and lineup:
            util_player = lineup[0]
            sorted_lineup[order.index('UTIL')] = util_player

        return sorted_lineup
        
    def sort_lineup_fd(self, lineup):
        order = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
        sorted_lineup = [None] * 9  # Total 9 players

        def find_and_remove(positions):
            for pos in positions:
                for player in lineup:
                    if player[1] == pos:
                        lineup.remove(player)
                        return player
            return None
        
        # Fill the first occurrences of each position
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            player = find_and_remove([pos])
            if player:
                sorted_lineup[order.index(pos)] = player

        # Now, fill the second occurrences or use dual positions if available
        for pos in ['PG', 'SG', 'SF', 'PF']:
            index = order.index(pos) + 1  # The next position after the first occurrence
            player = find_and_remove([pos]) or find_and_remove([f"{pos}/{order[index+1]}"])
            if player:
                sorted_lineup[index] = player

        # Fill any remaining spots with remaining players
        for i, player in enumerate(sorted_lineup):
            if not player and lineup:
                sorted_lineup[i] = lineup.pop(0)

        return sorted_lineup

    def adjust_roster_for_late_swap_dk(self, lineup):
       for player in lineup:
           print(player, self.player_dict[player]['GameTime'])
                    