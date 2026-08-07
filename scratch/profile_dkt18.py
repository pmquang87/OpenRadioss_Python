import sys
import threading
import time
import traceback

def run():
    from pyradioss.engine.engine import run_engine
    deck = r"tests\data\rd_decks\rd_e\RD-E-1000_Bending\10_Bending\DKT18\Sf_0.1\ROLLING_0001.rad"
    run_engine(deck)

t = threading.Thread(target=run)
t.start()

time.sleep(5)

if t.is_alive():
    print("Thread is still alive. Dumping stack traces...")
    for th in threading.enumerate():
        if th is not threading.current_thread():
            print(f"--- Thread {th.name} ---")
            frame = sys._current_frames().get(th.ident, None)
            if frame:
                traceback.print_stack(frame)
            print("-----------------------")
    # force exit
    import os
    os._exit(1)
print("Finished normally.")
