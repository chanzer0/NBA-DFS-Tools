import json
import csv
import os
import datetime
import numpy as np
import pulp as plp
import random
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
    team_replacement_dict = {
        'PHO': 'PHX',
        'GS': 'GSW',
    }
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    matchup_list = []
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
                
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                
                if (player_name, position, team) in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[(player_name, position, team)]['ID'] = int(row['ID'])
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game Info'].split(' ')[0]
                        if row['Game Info'].split(' ')[0] not in self.matchup_list:
                            self.matchup_list.append(row['Game Info'].split(' ')[0])
                        self.player_dict[(player_name, position, team)]['GameTime'] = ' '.join(row['Game Info'].split()[1:])
                        self.player_dict[(player_name, position, team)]['GameTime'] = datetime.datetime.strptime(self.player_dict[(player_name, position, team)]['GameTime'][:-3], '%m/%d/%Y %I:%M%p')
                    else:
                        self.player_dict[(player_name, position, team)]['ID'] = row['Id'].replace('-', '#')
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game']
                        if row['Game'] not in self.matchup_list:
                            self.matchup_list.append(row['Game'])
                
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
                if self.site == 'dk':
                    if 'PG' in row['position'] or 'SG' in row['position']:
                        self.player_dict[(player_name, position, team)]['Position'].append('G')
                    if 'SF' in row['position'] or 'PF' in row['position']:
                        self.player_dict[(player_name, position, team)]['Position'].append('F')
                    
                    self.player_dict[(player_name, position, team)]['Position'].append('UTIL')
                            
                if row['team'] not in self.team_list:
                    self.team_list.append(row['team'])
                
    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        # Create a binary decision variable for each player for each of their positions
        lp_variables = {}
        for player, attributes in self.player_dict.items():
            player_id = attributes['ID']
            for pos in attributes['Position']:
                lp_variables[(player, pos, player_id)] = plp.LpVariable(name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary)
        
        
        # set the objective - maximize fpts & set randomness amount from config
        if self.randomness_amount != 0:
            self.problem += (
                plp.lpSum(
                    np.random.normal(
                        self.player_dict[player]["Fpts"],
                        (
                            self.player_dict[player]["StdDev"]
                            * self.randomness_amount
                            / 100
                        ),
                    )
                    * lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position']
                ),
                "Objective",
            )
        else:
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Fpts"]
                    * lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position']
                ),
                "Objective",
            )
        
        
        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 49000 if self.site == 'dk' else 59000
        
        if self.projection_minimum is not None:
            min_salary = self.min_salary
        
        # Maximum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, pos, attributes['ID'])]
                for player, attributes in self.player_dict.items()
                for pos in attributes['Position']
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, pos, attributes['ID'])]
                for player, attributes in self.player_dict.items()
                for pos in attributes['Position']
            )
            >= min_salary,
            "Min Salary",
        )
        
        # Must not play all 8 or 9 players from the same team (8 if dk, 9 if fd)
        for matchup in self.matchup_list:
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position'] if attributes['Matchup'] == matchup
                )
                <= 8 if self.site == 'dk' else 9,
                f"Must not play all players from same matchup {matchup}",
            )


        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position'] if attributes['Name'] in group
                    )
                    >= int(limit),
                    f"At least {limit} players {group}",
                )

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position'] if attributes['Name'] in group
                    )
                    <= int(limit),
                    f"At most {limit} players {group}",
                )

        for matchup, limit in self.matchup_limits.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position'] if attributes['Matchup'] == matchup
                )
                <= int(limit),
                "At most {} players from {}".format(limit, matchup)
            )

        for matchup, limit in self.matchup_at_least.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position'] if attributes['Matchup'] == matchup
                )
                >= int(limit),
                "At least {} players from {}".format(limit, matchup)
            )

        # Address team limits
        for teamIdent, limit in self.team_limits.items():
            self.problem += plp.lpSum(lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                                      for (player, pos_str, team) in self.player_dict if team == teamIdent) <= int(limit), "At most {} players from {}".format(limit, teamIdent)
            
        if self.global_team_limit is not None:
            for teamIdent in self.team_list:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position']
                        if attributes["Team"] == teamIdent
                    )
                    <= int(self.global_team_limit),
                    f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                )

        if self.site == 'dk':
            # Constraints for specific positions
            for pos in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']:
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) == 1, f"Must have 1 {pos}"

            # Constraint to ensure each player is only selected once
            for player in self.player_dict:
                player_id = self.player_dict[player]['ID']
                self.problem += plp.lpSum(lp_variables[(player, pos, player_id)] for pos in self.player_dict[player]['Position']) <= 1, f"Can only select {player} once"

        else:
           # Constraints for specific positions
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                if pos == 'C':
                    self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) == 1, f"Must have 1 {pos}"
                else:
                    self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) == 2, f"Must have 2 {pos}"

            # Max 4 players from one team 
            for team in self.team_list:
                self.problem += plp.lpSum(
                    lp_variables[(player, pos, attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes['Position']
                    if team in attributes['Team']
                ) <= 4, f"Max 4 players from {team}"

            # Constraint to ensure each player is only selected once
            for player in self.player_dict:
                player_id = self.player_dict[player]['ID']
                self.problem += plp.lpSum(lp_variables[(player, pos, player_id)] for pos in self.player_dict[player]['Position']) <= 1, f"Can only select {player} once"

        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.lineups), self.num_lineups))
                break
                
            # Check for infeasibility
            if plp.LpStatus[self.problem.status] != 'Optimal':
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.lineups), self.num_lineups))
                break

            # Get the lineup and add it to our list
            selected_vars = [player for player in lp_variables if lp_variables[player].varValue != 0]
            # print(selected_vars)
            self.lineups.append(selected_vars)
            
            if i % 100 == 0:
                print(i)
            
            # Ensure this lineup isn't picked again
            self.problem += (
                plp.lpSum(lp_variables[x] for x in selected_vars) <= len(selected_vars) - self.num_uniques,
                f"Lineup {i}",
            )
            
            
            # self.problem.writeLP("problem.lp")

            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += (
                    plp.lpSum(
                        np.random.normal(
                            self.player_dict[player]["Fpts"],
                            (
                                self.player_dict[player]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        )
                        * lp_variables[(player, pos, attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes['Position']
                    ),
                    "Objective",
                )

    def output(self):
        print('Lineups done generating. Outputting.')
        
        sorted_lineups = []
        for lineup in self.lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineup = self.adjust_roster_for_late_swap(sorted_lineup)
            sorted_lineups.append(sorted_lineup)
         

        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_optimal_lineups_{}.csv'.format(self.site, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write(
                    'PG,SG,SF,PF,C,G,F,UTIL,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for x in sorted_lineups:
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
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['Name'], self.player_dict[x[0]]['ID'],
                        self.player_dict[x[1]]['Name'], self.player_dict[x[1]]['ID'],
                        self.player_dict[x[2]]['Name'], self.player_dict[x[2]]['ID'],
                        self.player_dict[x[3]]['Name'], self.player_dict[x[3]]['ID'],
                        self.player_dict[x[4]]['Name'], self.player_dict[x[4]]['ID'],
                        self.player_dict[x[5]]['Name'], self.player_dict[x[5]]['ID'],
                        self.player_dict[x[6]]['Name'], self.player_dict[x[6]]['ID'],
                        self.player_dict[x[7]]['Name'], self.player_dict[x[7]]['ID'],
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write(
                    'PG,PG,SG,SG,SF,SF,PF,PF,C,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for x in sorted_lineups:
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
                        self.player_dict[x[0]]['ID'].replace('#', '-'), self.player_dict[x[0]]['Name'],
                        self.player_dict[x[1]]['ID'].replace('#', '-'), self.player_dict[x[1]]['Name'],
                        self.player_dict[x[2]]['ID'].replace('#', '-'), self.player_dict[x[2]]['Name'],
                        self.player_dict[x[3]]['ID'].replace('#', '-'), self.player_dict[x[3]]['Name'],
                        self.player_dict[x[4]]['ID'].replace('#', '-'), self.player_dict[x[4]]['Name'],
                        self.player_dict[x[5]]['ID'].replace('#', '-'), self.player_dict[x[5]]['Name'],
                        self.player_dict[x[6]]['ID'].replace('#', '-'), self.player_dict[x[6]]['Name'],
                        self.player_dict[x[7]]['ID'].replace('#', '-'), self.player_dict[x[7]]['Name'],
                        self.player_dict[x[8]]['ID'].replace('#', '-'), self.player_dict[x[8]]['Name'],
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
        print('Output done.')



    def sort_lineup(self, lineup):
        if self.site == 'dk':
            order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            sorted_lineup = [None] * 8
        else:
            order = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
            sorted_lineup = [None] * 9
        
        for player in lineup:
            player_key, pos, _ = player
            order_idx = order.index(pos)
            if sorted_lineup[order_idx] is None:
                sorted_lineup[order_idx] = player_key
            else:
                sorted_lineup[order_idx + 1] = player_key
        return sorted_lineup

    def adjust_roster_for_late_swap(self, lineup):
        if self.site == 'fd':
            return lineup
        
        sorted_lineup = list(lineup)
        # A function to swap two players if the conditions are met
        def swap_if_needed(primary_pos, flex_pos):
            primary_player = sorted_lineup[primary_pos]
            flex_player = sorted_lineup[flex_pos]

            # Check if the primary player's game time is later than the flexible player's
            if self.player_dict[primary_player]['GameTime'] > self.player_dict[flex_player]['GameTime']:
                primary_positions = self.position_map[primary_pos]
                
                # Check if the flexible player is eligible for the primary position
                if any(pos in primary_positions for pos in self.player_dict[flex_player]['Position']):
                    sorted_lineup[primary_pos], sorted_lineup[flex_pos] = sorted_lineup[flex_pos], sorted_lineup[primary_pos]

        # Define eligible positions for each spot on the roster
        self.position_map = {
            0: ['PG'],
            1: ['SG'],
            2: ['SF'],
            3: ['PF'],
            4: ['C'],
            5: ['PG', 'SG'],
            6: ['SF', 'PF'],
            7: ['PG', 'SG', 'SF', 'PF', 'C']
        }
        

        # Check each primary position against all flexible positions
        for i in range(5):
            for j in range(5, 8):
                swap_if_needed(i, j)

        return sorted_lineup
