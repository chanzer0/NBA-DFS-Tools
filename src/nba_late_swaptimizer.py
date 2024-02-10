import json
import csv
import os
import datetime
import re
import numpy as np
import pulp as plp
import random
import itertools
import pytz

class NBA_Late_Swaptimizer:
    site = None
    config = None
    problem = None
    output_dir = None
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
    position_map = {
        0: ["PG"],
        1: ["SG"],
        2: ["SF"],
        3: ["PF"],
        4: ["C"],
        5: ["PG", "SG"],
        6: ["SF", "PF"],
        7: ["PG", "SG", "SF", "PF", "C"],
    }
    global_team_limit = None
    projection_minimum = None
    randomness_amount = 0
    min_salary = None

    def __init__(self, site=None, num_uniques=1):
        self.site = site
        self.num_uniques = int(num_uniques)
        self.load_config()
        self.load_rules()
        self.eastern = pytz.timezone('US/Eastern')

        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)

        late_swap_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["late_swap_path"]),
        )
        self.load_player_lineups(late_swap_path)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = "Name" if self.site == "dk" else "Nickname"
                player_name = row[name_key].replace("-", "#")
                team = row["TeamAbbrev"] if self.site == "dk" else row["Team"]
                position = row["Position"]

                self.ids_to_gametime[row["ID"].replace("-", "#")] = self.player_dict[
                    (player_name, position, team)
                ]["GameTime"]

                if (player_name, position, team) in self.player_dict:
                    if self.site == "dk":
                        self.player_dict[(player_name, position, team)]["ID"] = int(
                            row["ID"]
                        )
                        self.player_dict[(player_name, position, team)][
                            "Matchup"
                        ] = row["Game Info"].split(" ")[0]
                        if row["Game Info"].split(" ")[0] not in self.matchup_list:
                            self.matchup_list.append(row["Game Info"].split(" ")[0])
                        self.player_dict[(player_name, position, team)][
                            "GameTime"
                        ] = " ".join(row["Game Info"].split()[1:])
                        self.player_dict[(player_name, position, team)][
                            "GameTime"
                        ] = datetime.datetime.strptime(
                            self.player_dict[(player_name, position, team)]["GameTime"][
                                :-3
                            ],
                            "%m/%d/%Y %I:%M%p",
                        )
                    else:
                        self.player_dict[(player_name, position, team)]["ID"] = row[
                            "Id"
                        ].replace("-", "#")
                        self.player_dict[(player_name, position, team)][
                            "Matchup"
                        ] = row["Game"]
                        if row["Game"] not in self.matchup_list:
                            self.matchup_list.append(row["Game"])
    
                # Update ids_to_gametime with all players in player_ids
                if self.site == "dk":
                    game_time = " ".join(row["Game Info"].split()[1:])
                    game_time = self.eastern.localize(
                        datetime.datetime.strptime(game_time[:-3], "%m/%d/%Y %I:%M%p")
                    )
                    self.ids_to_gametime[int(row["ID"])] = game_time

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
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#")
                if float(row["fpts"]) < self.projection_minimum:
                    continue

                position = row["position"]
                team = row["team"]

                self.player_dict[(player_name, position, team)] = {
                    "Fpts": float(row["fpts"]),
                    "Salary": int(row["salary"].replace(",", "")),
                    "Minutes": float(row["minutes"]),
                    "Name": row["name"],
                    "Team": row["team"],
                    "Ownership": float(row["own%"]),
                    "StdDev": float(row["stddev"]),
                    "Position": [pos for pos in row["position"].split("/")],
                }
                if "PG" in row["position"] or "SG" in row["position"]:
                    self.player_dict[(player_name, position, team)]["Position"].append(
                        "G"
                    )
                if "SF" in row["position"] or "PF" in row["position"]:
                    self.player_dict[(player_name, position, team)]["Position"].append(
                        "F"
                    )

                self.player_dict[(player_name, position, team)]["Position"].append(
                    "UTIL"
                )

                if row["team"] not in self.team_list:
                    self.team_list.append(row["team"])

    # Load user lineups for late swap
    def load_player_lineups(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            current_time_utc = datetime.datetime.now(pytz.utc)  # get the current UTC time
            current_time = current_time_utc.astimezone(self.eastern) # convert UTC to 'US/Eastern'
            # current_time = datetime.datetime(2023, 10, 24, 20, 0) # testing time, such that LAL/DEN is locked
            print(f"Current time (ET): {current_time}")
            for row in reader:
                if row["entry id"] != "" and self.site == "dk":
                    PG_id = re.search(r"\((\d+)\)", row["pg"]).group(1)
                    SG_id = re.search(r"\((\d+)\)", row["sg"]).group(1)
                    SF_id = re.search(r"\((\d+)\)", row["sf"]).group(1)
                    PF_id = re.search(r"\((\d+)\)", row["pf"]).group(1)
                    C_id = re.search(r"\((\d+)\)", row["c"]).group(1)
                    G_id = re.search(r"\((\d+)\)", row["g"]).group(1)
                    F_id = re.search(r"\((\d+)\)", row["f"]).group(1)
                    UTIL_id = re.search(r"\((\d+)\)", row["util"]).group(1)
                    self.lineups.append(
                        {
                            "entry_id": row["entry id"],
                            "contest_id": row["contest id"],
                            "contest_name": row["contest name"],
                            "PG": row["pg"].replace("-", "#"),
                            "SG": row["sg"].replace("-", "#"),
                            "SF": row["sf"].replace("-", "#"),
                            "PF": row["pf"].replace("-", "#"),
                            "C": row["c"].replace("-", "#"),
                            "G": row["g"].replace("-", "#"),
                            "F": row["f"].replace("-", "#"),
                            "UTIL": row["util"].replace("-", "#"),
                            "PG_is_locked": current_time > self.ids_to_gametime[PG_id],
                            "SG_is_locked": current_time > self.ids_to_gametime[SG_id],
                            "SF_is_locked": current_time > self.ids_to_gametime[SF_id],
                            "PF_is_locked": current_time > self.ids_to_gametime[PF_id],
                            "C_is_locked": current_time > self.ids_to_gametime[C_id],
                            "G_is_locked": current_time > self.ids_to_gametime[G_id],
                            "F_is_locked": current_time > self.ids_to_gametime[F_id],
                            "UTIL_is_locked": current_time
                            > self.ids_to_gametime[UTIL_id],
                        }
                    )
        print(f"Successfully loaded {len(self.lineups)} lineups for late swap.")

    def swaptimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        current_time_utc = datetime.datetime.now(pytz.utc)  # get the current UTC time
        current_time = current_time_utc.astimezone(self.eastern) # convert UTC to 'US/Eastern'

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        for lineup_obj in self.lineups:
            print(
                f"Swaptimizing lineup {lineup_obj['entry_id']} in contest {lineup_obj['contest_name']}"
            )
            self.problem = plp.LpProblem("NBA", plp.LpMaximize)

            lp_variables = {}
            for player, attributes in self.player_dict.items():
                player_id = attributes["ID"]
                for pos in attributes["Position"]:
                    lp_variables[(player, pos, player_id)] = plp.LpVariable(
                        name=f"{player}_{pos}_{player_id}", cat=plp.LpBinary
                    )

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
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if (self.ids_to_gametime.get(attributes["ID"], current_time) > current_time)
                    ),
                    "Objective",
                )
            else:
                self.problem += (
                    plp.lpSum(
                        self.player_dict[player]["Fpts"]
                        * lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if (self.ids_to_gametime.get(attributes["ID"], current_time) > current_time)
                    ),
                    "Objective",
                )

            # Set the salary constraints
            max_salary = 50000 if self.site == "dk" else 60000
            min_salary = 49000 if self.site == "dk" else 59000

            if self.projection_minimum is not None:
                min_salary = self.min_salary

            # Maximum Salary Constraint
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Salary"]
                    * lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                )
                <= max_salary,
                "Max Salary",
            )

            # Minimum Salary Constraint
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Salary"]
                    * lp_variables[(player, pos, attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    for pos in attributes["Position"]
                )
                >= min_salary,
                "Min Salary",
            )

            # Must not play all 8 or 9 players from the same team (8 if dk, 9 if fd)
            for matchup in self.matchup_list:
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Matchup"] == matchup
                    )
                    <= 8
                    if self.site == "dk"
                    else 9,
                    f"Must not play all players from same matchup {matchup}",
                )

            # Address limit rules if any
            for limit, groups in self.at_least.items():
                for group in groups:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Name"] in group
                        )
                        >= int(limit),
                        f"At least {limit} players {group}",
                    )

            for limit, groups in self.at_most.items():
                for group in groups:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Name"] in group
                        )
                        <= int(limit),
                        f"At most {limit} players {group}",
                    )

            for matchup, limit in self.matchup_limits.items():
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Matchup"] == matchup
                    )
                    <= int(limit),
                    "At most {} players from {}".format(limit, matchup),
                )

            for matchup, limit in self.matchup_at_least.items():
                self.problem += (
                    plp.lpSum(
                        lp_variables[(player, pos, attributes["ID"])]
                        for player, attributes in self.player_dict.items()
                        for pos in attributes["Position"]
                        if attributes["Matchup"] == matchup
                    )
                    >= int(limit),
                    "At least {} players from {}".format(limit, matchup),
                )

            # Address team limits
            for teamIdent, limit in self.team_limits.items():
                self.problem += plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if team == teamIdent
                ) <= int(limit), "At most {} players from {}".format(limit, teamIdent)

            if self.global_team_limit is not None:
                for teamIdent in self.team_list:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            for pos in attributes["Position"]
                            if attributes["Team"] == teamIdent
                        )
                        <= int(self.global_team_limit),
                        f"Global team limit - at most {self.global_team_limit} players from {teamIdent}",
                    )

            # Force players to be used if they are locked
            POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
            FORCE_PLAYERS = []
            for position in POSITIONS:
                if lineup_obj[position + "_is_locked"]:
                    player_id = re.search(r"\((\d+)\)", lineup_obj[position]).group(1)
                    for p_tuple, attributes in self.player_dict.items():
                        if str(attributes["ID"]) == str(player_id):
                            FORCE_PLAYERS.append((p_tuple, position, attributes["ID"]))

            for forced_player in FORCE_PLAYERS:
                self.problem += (
                    lp_variables[forced_player] == 1,
                    f"Force player {forced_player}",
                )

            if self.site == "dk":
                # Constraints for specific positions
                for pos in ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]:
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, attributes["ID"])]
                            for player, attributes in self.player_dict.items()
                            if pos in attributes["Position"]
                        )
                        == 1,
                        f"Must have at 1 {pos}",
                    )

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]["ID"]
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, player_id)]
                            for pos in self.player_dict[player]["Position"]
                        )
                        <= 1,
                        f"Can only select {player} once",
                    )

            else:
                # Constraints for specific positions
                for pos in ["PG", "SG", "SF", "PF", "C"]:
                    if pos == "C":
                        self.problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                if pos in attributes["Position"]
                            )
                            == 1,
                            f"Must have 1 {pos}",
                        )
                    else:
                        self.problem += (
                            plp.lpSum(
                                lp_variables[(player, pos, attributes["ID"])]
                                for player, attributes in self.player_dict.items()
                                if pos in attributes["Position"]
                            )
                            == 2,
                            f"Must have 2 {pos}",
                        )

                # Constraint to ensure each player is only selected once
                for player in self.player_dict:
                    player_id = self.player_dict[player]["ID"]
                    self.problem += (
                        plp.lpSum(
                            lp_variables[(player, pos, player_id)]
                            for pos in self.player_dict[player]["Position"]
                        )
                        <= 1,
                        f"Can only select {player} once",
                    )

            # Don't dupe a lineup we already used
            i = 0
            for lineup, _ in self.output_lineups:
                player_ids = [tpl[2] for tpl in lineup]
                player_keys_to_exlude = []
                for key, attr in self.player_dict.items():
                    if attr["ID"] in player_ids:
                        for pos in attr["Position"]:
                            player_keys_to_exlude.append((key, pos, attr["ID"]))
                self.problem += (
                    plp.lpSum(lp_variables[x] for x in player_keys_to_exlude)
                    <= len(selected_vars) - self.num_uniques,
                    f"Lineup {i}",
                )
                i += 1

            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.output_lineups), self.lineups
                    )
                )
                break

            ## Check for infeasibility
            if plp.LpStatus[self.problem.status] != "Optimal":
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.lineups), self.num_lineups
                    )
                )
                break

            # Get the lineup and add it to our list
            selected_vars = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]
            self.output_lineups.append((selected_vars, lineup_obj))
            print(self.outout_lineups)

    def output(self):
        print("Lineups done generating. Outputting.")

        sorted_lineups = []
        for lineup, old_lineup in self.output_lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineup = self.adjust_roster_for_late_swap(sorted_lineup, old_lineup)
            new_sorted_lineup = []
            sorted_lineups.append((sorted_lineup, old_lineup))

        late_swap_lineups_contest_entry_dict = {
            (old_lineup["contest_id"], old_lineup["entry_id"]): new_lineup
            for new_lineup, old_lineup in sorted_lineups
        }

        late_swap_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, self.config["late_swap_path"]),
        )

        # Read the existing data first
        fieldnames = []
        with open(late_swap_path, "r", encoding="utf-8-sig") as file:
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
            if row["Entry ID"] != "":
                contest_id = row["Contest ID"]
                entry_id = row["Entry ID"]
                matching_lineup = late_swap_lineups_contest_entry_dict.get(
                    (contest_id, entry_id)
                )
                if matching_lineup:
                    row[
                        "PG"
                    ] = f"{self.player_dict[matching_lineup[0]]['Name']} ({self.player_dict[matching_lineup[0]]['ID']})"
                    row[
                        "SG"
                    ] = f"{self.player_dict[matching_lineup[1]]['Name']} ({self.player_dict[matching_lineup[1]]['ID']})"
                    row[
                        "SF"
                    ] = f"{self.player_dict[matching_lineup[2]]['Name']} ({self.player_dict[matching_lineup[2]]['ID']})"
                    row[
                        "PF"
                    ] = f"{self.player_dict[matching_lineup[3]]['Name']} ({self.player_dict[matching_lineup[3]]['ID']})"
                    row[
                        "C"
                    ] = f"{self.player_dict[matching_lineup[4]]['Name']} ({self.player_dict[matching_lineup[4]]['ID']})"
                    row[
                        "G"
                    ] = f"{self.player_dict[matching_lineup[5]]['Name']} ({self.player_dict[matching_lineup[5]]['ID']})"
                    row[
                        "F"
                    ] = f"{self.player_dict[matching_lineup[6]]['Name']} ({self.player_dict[matching_lineup[6]]['ID']})"
                    row[
                        "UTIL"
                    ] = f"{self.player_dict[matching_lineup[7]]['Name']} ({self.player_dict[matching_lineup[7]]['ID']})"

            updated_rows.append(row)

        new_late_swap_path = os.path.join(
            os.path.dirname(__file__),
            "../output/late_swap_{}.csv".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        )

        with open(new_late_swap_path, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=[PLACEHOLDER if f is None else f for f in fieldnames]
            )
            writer.writeheader()
            for row in updated_rows:
                if None in row:
                    row[PLACEHOLDER] = row.pop(None)
                writer.writerow(row)

        with open(new_late_swap_path, "r", encoding="utf-8-sig") as file:
            content = file.read().replace(PLACEHOLDER, "")

        with open(new_late_swap_path, "w", encoding="utf-8-sig") as file:
            file.write(content)

        print("Output done.")

    def sort_lineup(self, lineup):
        if self.site == "dk":
            order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
            sorted_lineup = [None] * 8
        else:
            order = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]
            sorted_lineup = [None] * 9

        for player in lineup:
            player_key, pos, _ = player
            order_idx = order.index(pos)
            if sorted_lineup[order_idx] is None:
                sorted_lineup[order_idx] = player_key
            else:
                sorted_lineup[order_idx + 1] = player_key
        return sorted_lineup
    
    def is_valid_for_position(self, player, position_idx):
        return any(
            pos in self.position_map[position_idx]
            for pos in self.player_dict[player]['Position']
        )

    def adjust_roster_for_late_swap(self, lineup, old_lineup):
        if self.site == "fd":
            return lineup

        POSITIONS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        def is_locked(position_index):
            if 0 <= position_index < 8:
                position_name = POSITIONS[position_index]
                return old_lineup[f"{position_name}_is_locked"]
            else:
                raise ValueError(f"Invalid position index: {position_index}")

        for i, position in enumerate(POSITIONS):
            # Only check G, F, and UTIL positions
            if position in ["G", "F", "UTIL"]:
                current_player = lineup[i]
                current_player_start_time = self.player_dict[current_player]["GameTime"]

                # Look for a swap candidate among primary positions
                for primary_i, primary_pos in enumerate(
                    POSITIONS[:5]
                ):  # Only the primary positions (0 to 4)
                    primary_player = lineup[primary_i]
                    primary_player_start_time = self.player_dict[primary_player][
                        "GameTime"
                    ]

                    if is_locked(primary_i) or is_locked(i):
                        print(
                            f"Skipping swap between {primary_player} and {current_player} because one of them is locked"
                        )
                        continue

                    # Check the conditions for the swap
                    primary_player_positions = self.player_dict[primary_player][
                        "Position"
                    ]
                    current_player_positions = self.player_dict[current_player][
                        "Position"
                    ]
                    # Check the conditions for the swap
                    if (
                        primary_player_start_time > current_player_start_time

                        and self.is_valid_for_position(primary_player, i)
                        and self.is_valid_for_position(current_player, primary_i)
                    ):
                        print(
                            f"Swapping {primary_player} and {current_player}, for positions {primary_pos} and {position}"
                        )
                        lineup[i], lineup[primary_i] = lineup[primary_i], lineup[i]
                        break  # Break out of the inner loop once a swap is made
        return lineup