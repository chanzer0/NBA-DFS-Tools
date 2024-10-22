import json
import csv
import os
import datetime
import numpy as np
import pulp as plp
import random
import itertools


class NBA_Pick5_Optimizer:
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
        "PHO": "PHX",
        "GS": "GSW",
        "SA": "SAS",
        "NO": "NOP",
        "NY": "NYK",
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

        self.problem = plp.LpProblem("NBA", plp.LpMaximize)

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
                name_key = "Name"
                player_name = row[name_key].replace("-", "#")
                team = row["TeamAbbrev"]
                position = row["Position"]

                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]

                if (player_name, team) in self.player_dict:
                    self.player_dict[(player_name, team)]["ID"] = int(row["ID"])
                    self.player_dict[(player_name, team)]["Matchup"] = row[
                        "Game Info"
                    ].split(" ")[0]

                    # Set position
                    self.player_dict[(player_name, team)]["Position"] = position

                    if row["Game Info"].split(" ")[0] not in self.matchup_list:
                        self.matchup_list.append(row["Game Info"].split(" ")[0])
                    self.player_dict[(player_name, team)]["GameTime"] = " ".join(
                        row["Game Info"].split()[1:]
                    )
                    self.player_dict[(player_name, team)]["GameTime"] = (
                        datetime.datetime.strptime(
                            self.player_dict[(player_name, team)]["GameTime"][:-3],
                            "%m/%d/%Y %I:%M%p",
                        )
                    )

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

                team = row["team"]
                if team in self.team_replacement_dict:
                    team = self.team_replacement_dict[team]
                self.player_dict[(player_name, team)] = {
                    "Fpts": float(row["fpts"]),
                    "Minutes": float(row["minutes"]) if "minutes" in row else 0,
                    "Name": row["name"],
                    "Team": row["team"],
                    "Ownership": float(row["own%"]),
                    "StdDev": float(row["stddev"]),
                    "Position": "",
                }

                if team not in self.team_list:
                    self.team_list.append(team)

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        # Create a binary decision variable for each player for each of their positions
        lp_variables = {}
        for player, attributes in self.player_dict.items():
            if "ID" in attributes:
                player_id = attributes["ID"]
            else:
                print(
                    f"Player in player_dict does not have an ID: {player}. Check for mis-matches between names, teams or positions in projections.csv and player_ids.csv"
                )
            position = attributes["Position"]
            lp_variables[(player, position, player_id)] = plp.LpVariable(
                name=f"{player}_{position}_{player_id}", cat=plp.LpBinary
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
                    * lp_variables[(player, attributes["Position"], attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                ),
                "Objective",
            )
        else:
            self.problem += (
                plp.lpSum(
                    self.player_dict[player]["Fpts"]
                    * lp_variables[(player, attributes["Position"], attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                ),
                "Objective",
            )

        # Must have 1 PG, 1 SG, 1 SF, 1 PF and 1 C.
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            self.problem += (
                plp.lpSum(
                    lp_variables[(player, attributes["Position"], attributes["ID"])]
                    for player, attributes in self.player_dict.items()
                    if pos == attributes["Position"]
                )
                == 1,
                f"Must have 1 {pos}",
            )

        self.problem.writeLP("problem.lp")
        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.lineups), self.num_lineups
                    )
                )
                break

            # Check for infeasibility
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
            self.lineups.append(selected_vars)

            if i % 100 == 0:
                print(i)

            # Ensure this lineup isn't picked again
            player_ids = [tpl[2] for tpl in selected_vars]
            player_keys_to_exlude = []
            for key, attr in self.player_dict.items():
                if attr["ID"] in player_ids:
                    player_keys_to_exlude.append((key, attr["Position"], attr["ID"]))

            self.problem += (
                plp.lpSum(lp_variables[x] for x in player_keys_to_exlude)
                <= len(selected_vars) - self.num_uniques,
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
                        * lp_variables[
                            (player, attributes["Position"], attributes["ID"])
                        ]
                        for player, attributes in self.player_dict.items()
                    ),
                    "Objective",
                )

    def output(self):
        print("Lineups done generating. Outputting.")

        sorted_lineups = []
        for lineup in self.lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineups.append(sorted_lineup)

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_pick5_opto_lineups_{}.csv".format(
                self.site, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        )
        with open(out_path, "w") as f:
            f.write("PG,SG,SF,PF,C,Fpts Proj,Own. Prod.,Own. Sum.,Minutes,StdDev\n")
            for x in sorted_lineups:
                fpts_p = sum(self.player_dict[player]["Fpts"] for player in x)
                own_p = np.prod(
                    [self.player_dict[player]["Ownership"] / 100 for player in x]
                )
                own_s = sum(self.player_dict[player]["Ownership"] for player in x)
                mins = sum([self.player_dict[player]["Minutes"] for player in x])
                stddev = sum([self.player_dict[player]["StdDev"] for player in x])
                lineup_str = (
                    "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{}".format(
                        self.player_dict[x[0]]["Name"],
                        self.player_dict[x[0]]["ID"],
                        self.player_dict[x[1]]["Name"],
                        self.player_dict[x[1]]["ID"],
                        self.player_dict[x[2]]["Name"],
                        self.player_dict[x[2]]["ID"],
                        self.player_dict[x[3]]["Name"],
                        self.player_dict[x[3]]["ID"],
                        self.player_dict[x[4]]["Name"],
                        self.player_dict[x[4]]["ID"],
                        round(fpts_p, 2),
                        own_p,
                        own_s,
                        mins,
                        stddev,
                    )
                )
                f.write("%s\n" % lineup_str)

        print("Output done.")

    def sort_lineup(self, lineup):
        order = ["PG", "SG", "SF", "PF", "C"]
        sorted_lineup = [None] * 5

        for player in lineup:
            player_key, pos, _ = player
            order_idx = order.index(pos)
            if sorted_lineup[order_idx] is None:
                sorted_lineup[order_idx] = player_key
            else:
                sorted_lineup[order_idx + 1] = player_key
        return sorted_lineup
