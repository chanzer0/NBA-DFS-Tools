

import sys
from nfl_optimizer import *
from nba_optimizer import *
from ilya_optimizer import *
from nba_gpp_simulator import *
from fanduel_nba_optimizer import *
from fanduel_nba_gpp_simulator import *
from nba_evolutionary_lineup_selector import *
from windows_inhibitor import *

def main(flag, rest):
    if flag.lower() == 'nba':
        o = NBA_Optimizer()
        o.optimize(rest[2])
        o.output()

    elif flag.lower() == 'fd':
        osSleep = WindowsInhibitor()
        osSleep.inhibit()
        o = FD_NBA_Optimizer()
        o.optimize('rand')
        o.output()
        sim = FD_NBA_GPP_Simulator()
        sim.generate_field_lineups(10000)
        sim.run_tournament_simulation(10000)
        sim.output()
        osSleep.uninhibit()

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
        osSleep = WindowsInhibitor()
        osSleep.inhibit()
    
        o = NBA_Optimizer()
        o.optimize('rand')
        o.output()
        sim = NBA_GPP_Simulator()
        sim.generate_field_lineups(10000)
        sim.run_tournament_simulation(10000)
        sim.output()

        osSleep.uninhibit()

    elif flag.lower() == 'evolutionary':
        selector = NBA_Evolutionary_Lineup_Selector()
        selector.run_evolution()

    elif flag.lower() == 'ilya':
        o = NBA_Ilya_Optimizer()
        o.optimize(rest[2])
        o.output()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv)