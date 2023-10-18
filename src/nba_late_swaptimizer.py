import json
import csv
import os
import datetime
import re
import numpy as np
import pulp as plp
from random import shuffle, choice
import itertools

class NBA_Late_Swaptimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = []
    output_lineups = []
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_list = []
    matchup_at_least = {}
    ids_to_gametime = {}
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

        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)
        
        late_swap_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['late_swap_path']))
        self.load_player_lineups(late_swap_path)

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
                        if row['Game Info'].split(' ')[0] not in self.matchup_list:
                            self.matchup_list.append(row['Game Info'].split(' ')[0])
                        self.player_dict[(player_name, position, team)]['GameTime'] = ' '.join(row['Game Info'].split()[1:])
                        self.player_dict[(player_name, position, team)]['GameTime'] = datetime.datetime.strptime(self.player_dict[(player_name, position, team)]['GameTime'][:-3], '%m/%d/%Y %I:%M%p')
                        self.ids_to_gametime[int(row['ID'])] = self.player_dict[(player_name, position, team)]['GameTime']
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
                if row['team'] not in self.team_list:
                    self.team_list.append(row['team'])
       
    # Load user lineups for late swap
    def load_player_lineups(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if row['entry id'] != '' and self.site == 'dk':
                    current_time = datetime.datetime.now()  # get the current time
                    # current_time = datetime.datetime(2023, 10, 24, 20, 0)
                    PG_id = re.search(r"\((\d+)\)", row['pg']).group(1)
                    SG_id = re.search(r"\((\d+)\)", row['sg']).group(1)
                    SF_id = re.search(r"\((\d+)\)", row['sf']).group(1)
                    PF_id = re.search(r"\((\d+)\)", row['pf']).group(1)
                    C_id = re.search(r"\((\d+)\)", row['c']).group(1)
                    G_id = re.search(r"\((\d+)\)", row['g']).group(1)
                    F_id = re.search(r"\((\d+)\)", row['f']).group(1)
                    UTIL_id = re.search(r"\((\d+)\)", row['util']).group(1)
                    self.lineups.append(
                        {
                            'entry_id': row['entry id'],
                            'contest_id': row['contest id'],
                            'PG': row['pg'].replace('-', '#'),
                            'SG': row['sg'].replace('-', '#'),
                            'SF': row['sf'].replace('-', '#'),
                            'PF': row['pf'].replace('-', '#'),
                            'C': row['c'].replace('-', '#'),
                            'G': row['g'].replace('-', '#'),
                            'F': row['f'].replace('-', '#'),
                            'UTIL': row['util'].replace('-', '#'),
                            'PG_is_locked': current_time > self.ids_to_gametime[int(PG_id)],
                            'SG_is_locked': current_time > self.ids_to_gametime[int(SG_id)],
                            'SF_is_locked': current_time > self.ids_to_gametime[int(SF_id)],
                            'PF_is_locked': current_time > self.ids_to_gametime[int(PF_id)],
                            'C_is_locked': current_time > self.ids_to_gametime[int(C_id)],
                            'G_is_locked': current_time > self.ids_to_gametime[int(G_id)],
                            'F_is_locked': current_time > self.ids_to_gametime[int(F_id)],
                            'UTIL_is_locked': current_time > self.ids_to_gametime[int(UTIL_id)],
                        }
                    )
       
                
    def swaptimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        
        
        for lineup_obj in self.lineups:
            # print(lineup_obj)
            already_used_players = [lineup_obj['PG'], lineup_obj['SG'], lineup_obj['SF'], lineup_obj['PF'], lineup_obj['C'], lineup_obj['G'], lineup_obj['F'], lineup_obj['UTIL']]
            self.problem = plp.LpProblem('NBA', plp.LpMaximize)
            
            # for (player, pos_str, team) in self.player_dict:
            #     print((player, pos_str, team))
            #     print(self.player_dict[(player, pos_str, team)]["ID"])

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
                for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) >= 1, f"Must have at least 1 {pos}"

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]['ID']
                    self.problem += plp.lpSum(lp_variables[(player, pos, player_id)] for pos in self.player_dict[player]['Position']) <= 1, f"Can only select {player} once"

                # Handle the G, F, and UTIL spots
                guards = ['PG', 'SG']
                forwards = ['SF', 'PF']

                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in guards if pos in attributes['Position']) >= 3, f"Must have at least 3 guards"
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in forwards if pos in attributes['Position']) >= 3, f"Must have at least 3 forwards"

                # UTIL can be from any position. But you don't really need a separate constraint for UTIL.
                # It's automatically handled because you're ensuring every other position is filled and each player is selected at most once. 
                # If you've correctly handled the other positions, UTIL will be the remaining player.

                # Total players constraint
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in attributes['Position']) == 8, f"Must have 8 players"
            else:
            # Constraints for specific positions
                for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    if pos == 'C':
                        self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) >= 1, f"Must have at least 1 {pos}"
                    else:
                        self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() if pos in attributes['Position']) >= 2, f"Must have at least 2 {pos}"


                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]['ID']
                    self.problem += plp.lpSum(lp_variables[(player, pos, player_id)] for pos in self.player_dict[player]['Position']) <= 1, f"Can only select {player} once"

                
                # Total players constraint
                self.problem += plp.lpSum(lp_variables[(player, pos, attributes['ID'])] for player, attributes in self.player_dict.items() for pos in attributes['Position']) == 9, f"Must have 9 players"

            # Don't dupe a lineup we already used
            i = 0
            for lineup, _ in self.output_lineups:
                # Ensure this lineup isn't picked again
                unique_constraints = []
                for player_key in lineup:
                    player_id = self.player_dict[player_key]['ID']
                    for position in self.player_dict[player_key]['Position']:
                        unique_constraints.append(lp_variables[(player_key, position, player_id)])

                self.problem += (
                    plp.lpSum(unique_constraints) <= len(lineup) - self.num_uniques,
                    f"Lineup {i}",
                )
                i += 1
            
           
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.output_lineups), self.lineups))
                break
                
            # Check for infeasibility
            if plp.LpStatus[self.problem.status] != 'Optimal':
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.output_lineups), self.lineups))
                break

            # Get the lineup and add it to our list
            selected_vars = [player for player in lp_variables if lp_variables[player].varValue != 0]
            # print(selected_vars)

            selected_players_info = [var[0] for var in selected_vars]

            players = []
            for key in self.player_dict.keys():
                if key in selected_players_info:
                    players.append(key)
                      
            # fpts_used = self.problem.objective.value()
            # print(fpts_used, players)
            self.output_lineups.append((players, lineup_obj))


    def output(self):
        print('Lineups done generating. Outputting.')
       
        sorted_lineups = []
        for lineup, old_lineup in self.output_lineups:
            # print(lineup)
            sorted_lineup = self.sort_lineup_dk(lineup, old_lineup)
            # print(sorted_lineup)
            if self.site == 'dk':
                sorted_lineup = self.adjust_roster_for_late_swap_dk(sorted_lineup, old_lineup)
                # print(sorted_lineup)
                # print('-------------------')
            sorted_lineups.append((sorted_lineup, old_lineup))
            
        late_swap_lineups_contest_entry_dict = {
            (old_lineup['contest_id'], old_lineup['entry_id']): new_lineup for new_lineup, old_lineup in sorted_lineups
        }
        
        # for tuple in late_swap_lineups_contest_entry_dict:
        #     print(tuple)

        late_swap_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(self.site, self.config['late_swap_path']))
                         

        # Read the existing data first
        fieldnames = []
        with open(late_swap_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            rows = [row for row in reader]


        PLACEHOLDER = "PLACEHOLDER_FOR_NONE"
        # If any row has a None key, ensure the placeholder is in the fieldnames
        for row in rows:
            if None in row and PLACEHOLDER not in fieldnames:
                fieldnames.append(PLACEHOLDER)

        # Now, modify the rows
        updated_rows = []
        for row in rows:
            if row['Entry ID'] != '':
                contest_id = row['Contest ID']
                entry_id = row['Entry ID']
                matching_lineup = late_swap_lineups_contest_entry_dict.get((contest_id, entry_id))
                if matching_lineup:
                    row['PG'] = f"{self.player_dict[matching_lineup[0]]['Name']} ({self.player_dict[matching_lineup[0]]['ID']})"
                    row['SG'] = f"{self.player_dict[matching_lineup[1]]['Name']} ({self.player_dict[matching_lineup[1]]['ID']})"
                    row['SF'] = f"{self.player_dict[matching_lineup[2]]['Name']} ({self.player_dict[matching_lineup[2]]['ID']})"
                    row['PF'] = f"{self.player_dict[matching_lineup[3]]['Name']} ({self.player_dict[matching_lineup[3]]['ID']})"
                    row['C'] = f"{self.player_dict[matching_lineup[4]]['Name']} ({self.player_dict[matching_lineup[4]]['ID']})"
                    row['G'] = f"{self.player_dict[matching_lineup[5]]['Name']} ({self.player_dict[matching_lineup[5]]['ID']})"
                    row['F'] = f"{self.player_dict[matching_lineup[6]]['Name']} ({self.player_dict[matching_lineup[6]]['ID']})"
                    row['UTIL'] = f"{self.player_dict[matching_lineup[7]]['Name']} ({self.player_dict[matching_lineup[7]]['ID']})"
                
            updated_rows.append(row)

        new_late_swap_path = os.path.join(os.path.dirname(
            __file__), '../output/late_swap_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

        with open(new_late_swap_path, 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[PLACEHOLDER if f is None else f for f in fieldnames])
            writer.writeheader()
            for row in updated_rows:
                if None in row:
                    row[PLACEHOLDER] = row.pop(None)
                writer.writerow(row)

                
        with open(new_late_swap_path, 'r', encoding='utf-8-sig') as file:
            content = file.read().replace(PLACEHOLDER, '')

        with open(new_late_swap_path, 'w', encoding='utf-8-sig') as file:
            file.write(content)
        
        print('Output done.')


    def sort_lineup_dk(self, lineup, old_lineup):
        # 1. Initialize our final lineup
        sorted_lineup = {
            'PG': None, 'SG': None, 'SF': None, 'PF': None, 
            'C': None, 'G': None, 'F': None, 'UTIL': None
        }

        # This will store full position and team information
        player_info = {player: (position, team) for player, position, team in lineup}

        # 2. Fill in locked players
        for pos, player in old_lineup.items():
            if pos.endswith('_is_locked') and player:
                main_pos = pos.split('_is_locked')[0]
                if old_lineup[main_pos]:
                    player_name = old_lineup[main_pos].split(' ')[0]
                    sorted_lineup[main_pos] = player_name
                    # remove the locked player from the lineup list
                    lineup = [p for p in lineup if p[0] != player_name]

        # 3. Fill in the remaining players using recursion
        def place_player(idx, current_lineup):
            # base case: all players placed
            if idx == len(lineup):
                return True
            
            player, positions, _ = lineup[idx]
            for pos, _ in current_lineup.items():
                if not current_lineup[pos] and (pos in positions or pos == 'UTIL'):
                    # place the player
                    current_lineup[pos] = player
                    if place_player(idx + 1, current_lineup):
                        return True
                    # backtrack
                    current_lineup[pos] = None

            return False

        place_player(0, sorted_lineup)

        # Convert the dictionary to a list of tuples
        final_sorted_lineup = [(player, *player_info[player]) for pos, player in sorted_lineup.items() if player]

        return final_sorted_lineup

        

    """
    Takes an old_lineup, which dictates which players are already locked and cannot be moved,
    lineup which is a newly-constructed roster, and returns a new sorted lineup optimized for
    late swapping, where the later a players gametime is, the further down the roster they go,
    so long as both player being swapped are eligible for either position
    """
    def adjust_roster_for_late_swap_dk(self, lineup, old_lineup):
        print(old_lineup)
        print(lineup)
        POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        def is_locked(position_index):
            if 0 <= position_index < 8:
                position_name = POSITIONS[position_index]
                return old_lineup[f"{position_name}_is_locked"]
            else:
                raise ValueError(f"Invalid position index: {position_index}")

        # 1. Map each player in lineup to positions they are eligible to be swapped into
        player_to_swappable_positions = {}
        for i, player in enumerate(lineup):
            if is_locked(i):
                continue
            positions = self.player_dict[player]['Position']
            player_to_swappable_positions[player] = [
                "PG" if "PG" in positions else None,
                "SG" if "SG" in positions else None,
                "SF" if "SF" in positions else None,
                "PF" if "PF" in positions else None,
                "C" if "C" in positions else None,
                "G" if "PG" in positions or "SG" in positions else None,
                "F" if "SF" in positions or "PF" in positions else None,
                "UTIL"                                         
            ]
            # remove nones from the array
            player_to_swappable_positions[player] = [pos for pos in player_to_swappable_positions[player] if pos is not None]

        has_player_been_swapped = {player: False for player in lineup}
        # 2. Attempt to swap players in lineup with positions they are eligible for, only swapping if both players are eligible for one another's positions and the later GameTime is later in the roster
        for i, player in enumerate(lineup):
            if is_locked(i):
                continue
            eligible_positions = player_to_swappable_positions[player]
            # print(player, eligible_positions)
            for pos in eligible_positions:
                pos_idx = POSITIONS.index(pos)
                if pos_idx == i:
                    continue
                # print(pos)
                player_at_swappable_pos = lineup[pos_idx]
                
                if has_player_been_swapped[player_at_swappable_pos]:
                    continue
                
                is_positionally_eligible = POSITIONS[i] in player_to_swappable_positions[player_at_swappable_pos]
                # print(player_at_swappable_pos, is_positionally_eligible)
                if not is_positionally_eligible:
                    continue
                
                # print(self.player_dict[player]['GameTime'])
                # print(self.player_dict[player_at_swappable_pos]['GameTime'])
                if self.player_dict[player]['GameTime'] < self.player_dict[player_at_swappable_pos]['GameTime']:
                    # print(f"Swapping {player} and {player_at_swappable_pos}")
                    lineup[i], lineup[pos_idx] = lineup[pos_idx], lineup[i]
                    has_player_been_swapped[player] = True
                    break
       
       
        print(lineup)
        print('-------------------')
        return lineup

