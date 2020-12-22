import sys
from nfl_optimizer import *
from nba_optimizer import *
from nba_gpp_simulator import *

def main(flag, rest):
    if flag.lower() == 'nba':
        o = NBA_Optimizer()
        o.optimize()
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

if __name__ == "__main__":
    main(sys.argv[1], sys.argv)