from pyradioss.engine.engine import run_engine

def run():
    deck = r"tests\data\rd_decks\rd_e\RD-E-1000_Bending\10_Bending\DKT18\Sf_0.1\ROLLING_0001.rad"
    # we can mock or patch the engine loop to print
    run_engine(deck)

if __name__ == "__main__":
    run()
