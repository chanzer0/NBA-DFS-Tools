import sys
from nfl_optimizer import *
from nba_optimizer import *

def main(optimizerType):
    if optimizerType == 'NBA':
        o = NBA_Optimizer()
        o.optimize()
        # o.output()

    elif optimizerType == 'NFL':
        o = NFL_Optimizer()
        o.optimize()
        # o.output()

if __name__ == "__main__":
    main(sys.argv[1])