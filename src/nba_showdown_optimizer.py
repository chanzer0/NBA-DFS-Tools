import json
import csv
import os
import datetime
import numpy as np
import pulp as plp
import random
import itertools

class NBA_Showdown_Optimizer:
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
        'SA': 'SAS',
        'NO': 'NOP',
        'NY': 'NYK',
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
                position = row['Roster Position']
                
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                    
                if self.site == 'dk':
                    if (player_name, position, team) in self.player_dict:
                        self.player_dict[(player_name, position, team)]['ID'] = int(row['ID'])
                        self.player_dict[(player_name, position, team)]['Matchup'] = row['Game Info'].split(' ')[0]
                        if row['Game Info'].split(' ')[0] not in self.matchup_list:
                            self.matchup_list.append(row['Game Info'].split(' ')[0])
                        self.player_dict[(player_name, position, team)]['GameTime'] = ' '.join(row['Game Info'].split()[1:])
                        self.player_dict[(player_name, position, team)]['GameTime'] = datetime.datetime.strptime(self.player_dict[(player_name, position, team)]['GameTime'][:-3], '%m/%d/%Y %I:%M%p')
                else:
                    for pos in ["MVP", "STAR", "PRO", "UTIL"]:
                            if (player_name, pos, team) in self.player_dict:
                                self.player_dict[(player_name, pos, team)]['ID'] = row['Id'].replace('-', '#')
                                self.player_dict[(player_name, pos, team)]['Matchup'] = row['Game']
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
                
                team = row['team']
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                
                if self.site == 'dk':
                    self.player_dict[(player_name, "CPT", team)] = {
                        'Fpts': 1.5 * float(row['fpts']),
                        'Salary': 1.5 * int(row['salary'].replace(',', '')),
                        'Minutes': float(row['minutes']),
                        'Name': row['name'],
                        'Team': row['team'],
                        'Ownership': float(row['cptown%']),
                        'StdDev': 1.5 * float(row['stddev']),
                        'Position': [pos for pos in row['position'].split('/')],
                    }
                    self.player_dict[(player_name, "UTIL", team)] = {
                        'Fpts': float(row['fpts']),
                        'Salary': int(row['salary'].replace(',', '')),
                        'Minutes': float(row['minutes']),
                        'Name': row['name'],
                        'Team': row['team'],
                        'Ownership': float(row['own%']),
                        'StdDev': float(row['stddev']),
                        'Position': [pos for pos in row['position'].split('/')],
                    }
                else:
                    self.player_dict[(player_name, "MVP", team)] = {
                        'Fpts': 2 * float(row['fpts']),
                        'Salary': int(row['salary'].replace(',', '')),
                        'Minutes': float(row['minutes']),
                        'Name': row['name'],
                        'Team': row['team'],
                        'Ownership': float(row['mvpown%']),
                        'StdDev': 2 * float(row['stddev']),
                        'Position': [pos for pos in row['position'].split('/')],
                    }
                    self.player_dict[(player_name, "STAR", team)] = {
                        'Fpts': 1.5 * float(row['fpts']),
                        'Salary': int(row['salary'].replace(',', '')),
                        'Minutes': float(row['minutes']),
                        'Name': row['name'],
                        'Team': row['team'],
                        'Ownership': float(row['starown%']),
                        'StdDev': 1.5 * float(row['stddev']),
                        'Position': [pos for pos in row['position'].split('/')],
                    }
                    self.player_dict[(player_name, "PRO", team)] = {
                        'Fpts': 1.2 * float(row['fpts']),
                        'Salary': int(row['salary'].replace(',', '')),
                        'Minutes': float(row['minutes']),
                        'Name': row['name'],
                        'Team': row['team'],
                        'Ownership': float(row['proown%']),
                        'StdDev': 1.2 * float(row['stddev']),
                        'Position': [pos for pos in row['position'].split('/')],
                    }
                    self.player_dict[(player_name, "UTIL", team)] = {
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
        # Create a binary decision variable for each player for each of their positions
        lp_variables = {}
        for player, attributes in self.player_dict.items():
            player_id = attributes['ID']
            lp_variables[(player, player[1], player_id)] = plp.LpVariable(name=f"{player}_{player[1]}_{player_id}", cat=plp.LpBinary)

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
                    * lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                ),
                "Objective",
            )
        else:
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Fpts"]
                    * lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                ),
                "Objective",
            )
        
        
        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 48000 if self.site == 'dk' else 58000
        
        if self.projection_minimum is not None:
            min_salary = self.min_salary
        
        # Maximum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, player[1], attributes['ID'])]
                for player, attributes in self.player_dict.items()
            )
            <= max_salary,
            "Max Salary",
        )

        # Minimum Salary Constraint
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"]
                * lp_variables[(player, player[1], attributes['ID'])]
                for player, attributes in self.player_dict.items()
            )
            >= min_salary,
            "Min Salary",
        )

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, player[1], attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        if attributes['Name'] in group
                    )
                    >= int(limit),
                    f"At least {limit} players {group}",
                )
               
                    

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, player[1], attributes['ID'])]
                        for player, attributes in self.player_dict.items()
                        if attributes['Name'] in group
                    )
                    <= int(limit),
                    f"At most {limit} players {group}",
                )

        for matchup, limit in self.matchup_limits.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    if attributes['Matchup'] == matchup
                )
                <= int(limit),
                "At most {} players from {}".format(limit, matchup)
            )
           

        for matchup, limit in self.matchup_at_least.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    if attributes['Matchup'] == matchup
                )
                >= int(limit),
                "At least {} players from {}".format(limit, matchup)
            )
           
                

        # Address team limits
        for teamIdent, limit in self.team_limits.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    if team == teamIdent
                )
                <= int(limit),
                "At most {} players from {}".format(limit, teamIdent)
            )
            
        if self.global_team_limit is not None:
            if not (self.site == 'fd' and self.global_team_limit >= 4):
                for teamIdent in self.team_list:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, player[1], attributes['ID'])]
                            for player, attributes in self.player_dict.items()
                            if attributes["Team"] == teamIdent
                        )
                        <= int(self.global_team_limit),
                        f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                    )

        if self.site == 'dk':
            # 1 CPT
            cpt_players = [(player, "CPT", attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "CPT"]
            self.problem += plp.lpSum(lp_variables[player] for player in cpt_players) == 1, "Must have 1 CPT"
            
            # 5 UTIL
            util_players = [(player, "UTIL", attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "UTIL"]
            self.problem += plp.lpSum(lp_variables[player] for player in util_players) == 5, "Must have 5 UTIL"
            
            # Constraint to ensure each player is only selected once
            players_grouped_by_name = {v['Name']: [] for v in self.player_dict.values()}
            for key, value in self.player_dict.items():
                players_grouped_by_name[value['Name']].append(key)

            player_key_groups = [group for group in players_grouped_by_name.values()]
                    
            for player_key_list in player_key_groups:
                self.problem += (
                    plp.lpSum(lp_variables[(pk, pk[1], self.player_dict[pk]['ID'])] for pk in player_key_list) <= 1,
                    f"Can only select {player_key_list} once",
                )
                
        else:
            # 1 MVP
            mvp_players = [(player, "MVP", attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "MVP"]
            self.problem += plp.lpSum(lp_variables[player] for player in mvp_players) == 1, "Must have 1 MVP"
            
            # 1 STAR
            star_players = [(player, "STAR", attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "STAR"]
            self.problem += plp.lpSum(lp_variables[player] for player in star_players) == 1, "Must have 1 STAR"
            
            # 1 PRO
            pro_players = [(player, "PRO", attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "PRO"]
            self.problem += plp.lpSum(lp_variables[player] for player in pro_players) == 1, "Must have 1 PRO"
            
            # 2 UTIL
            util_players = [(player, "UTIL", attributes['ID']) for player, attributes in self.player_dict.items() if player[1] == "UTIL"]
            self.problem += plp.lpSum(lp_variables[player] for player in util_players) == 2, "Must have 2 UTIL"

            # Max 4 players from one team 
            for team in self.team_list:
                self.problem += plp.lpSum(
                    lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                    if team in attributes['Team']
                ) <= 4, f"Max 4 players from {team}"

            # Constraint to ensure each player is only selected once
            players_grouped_by_name = {v['Name']: [] for v in self.player_dict.values()}
            for key, value in self.player_dict.items():
                players_grouped_by_name[value['Name']].append(key)

            player_key_groups = [group for group in players_grouped_by_name.values()]
                    
            for player_key_list in player_key_groups:
                self.problem += (
                    plp.lpSum(lp_variables[(pk, pk[1], self.player_dict[pk]['ID'])] for pk in player_key_list) <= 1,
                    f"Can only select {player_key_list} once",
                )

        # Crunch!
        for i in range(self.num_lineups):
            self.problem.writeLP("problem.lp")
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
                    * lp_variables[(player, player[1], attributes['ID'])]
                    for player, attributes in self.player_dict.items()
                ),
                "Objective",
            )

    def output(self):
        print('Lineups done generating. Outputting.')
        
        sorted_lineups = []
        for lineup in self.lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineups.append(sorted_lineup)
         

        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_sd_optimal_lineups_{}.csv'.format(self.site, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write(
                    'CPT,UTIL,UTIL,UTIL,UTIL,UTIL,Salary,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n')
                for x in sorted_lineups:
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] / 100 for player in x])
                    own_s = sum(self.player_dict[player]['Ownership'] for player in x)
                    mins = sum([self.player_dict[player]
                               ['Minutes'] for player in x])
                    stddev = sum(
                        [self.player_dict[player]['StdDev'] for player in x])
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['Name'], self.player_dict[x[0]]['ID'],
                        self.player_dict[x[1]]['Name'], self.player_dict[x[1]]['ID'],
                        self.player_dict[x[2]]['Name'], self.player_dict[x[2]]['ID'],
                        self.player_dict[x[3]]['Name'], self.player_dict[x[3]]['ID'],
                        self.player_dict[x[4]]['Name'], self.player_dict[x[4]]['ID'],
                        self.player_dict[x[5]]['Name'], self.player_dict[x[5]]['ID'],
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
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['ID'].replace('#', '-'), self.player_dict[x[0]]['Name'],
                        self.player_dict[x[1]]['ID'].replace('#', '-'), self.player_dict[x[1]]['Name'],
                        self.player_dict[x[2]]['ID'].replace('#', '-'), self.player_dict[x[2]]['Name'],
                        self.player_dict[x[3]]['ID'].replace('#', '-'), self.player_dict[x[3]]['Name'],
                        self.player_dict[x[4]]['ID'].replace('#', '-'), self.player_dict[x[4]]['Name'],
                        salary, round(
                            fpts_p, 2), own_p, own_s, mins, stddev
                    )
                    f.write('%s\n' % lineup_str)
        print('Output done.')



    def sort_lineup(self, lineup):
        if self.site == 'dk':
            order = ['CPT', 'UTIL', 'UTIL', 'UTIL', 'UTIL', 'UTIL']
            sorted_lineup = [None] * 6
        else:
            order = ['MVP', 'STAR', 'PRO', 'UTIL', 'UTIL']
            sorted_lineup = [None] * 5

        for player in lineup:
            player_key, pos, _ = player
            assigned = False
            for idx, position in enumerate(order):
                if position == pos and sorted_lineup[idx] is None:
                    sorted_lineup[idx] = player_key
                    assigned = True
                    break

            # In case all positions are filled, try to put player in any free spot
            if not assigned:
                for idx, spot in enumerate(sorted_lineup):
                    if spot is None:
                        sorted_lineup[idx] = player_key
                        break

        return sorted_lineup

