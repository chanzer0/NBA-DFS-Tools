import sys
from nfl_optimizer import *
from nba_optimizer import *
from nba_gpp_simulator import *
from nba_evolutionary_lineup_selector import *

def main(flag, rest):
    if flag.lower() == 'nba':
        o = NBA_Optimizer()
        o.optimize(rest[2])
        o.output()

    elif flag.lower() == 'nfl':
        o = NFL_Optimizer()
        o.optimize()
        o.output()

    elif flag.lower() == 'sim':
        sim = NBA_GPP_Simulator()
        sim.generate_field_lineups(rest[2])
        sim.run_tournament_simulation(rest[3])
        sim.output()

    elif flag.lower() == 'all':
        o = NBA_Optimizer()
        o.optimize()
        o.output()
        sim = NBA_GPP_Simulator()
        sim.generate_field_lineups(rest[2])
        sim.run_tournament_simulation(rest[3])
        sim.output()

    elif flag.lower() == 'evolutionary':
        selector = NBA_Evolutionary_Lineup_Selector()
        selector.run_evolution()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv)