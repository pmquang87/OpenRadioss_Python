from pyradioss.engine import engine

original_integrate = engine._integrate

def patched_integrate(model, controls, log, out_dir, run_name, engine_start_time, resume_n=0, resume_t=0.0):
    print("Integration started!")
    
    import sys
    def trace_calls(frame, event, arg):
        if event == 'call' and frame.f_code.co_name == 'forces' and 'shell_tri3' in frame.f_code.co_filename:
            print("called forces in shell_tri3")
        return trace_calls

    sys.settrace(trace_calls)
    return original_integrate(model, controls, log, out_dir, run_name, engine_start_time, resume_n, resume_t)

engine._integrate = patched_integrate

engine.run_engine(r"tests\data\rd_decks\rd_e\RD-E-1000_Bending\10_Bending\DKT18\Sf_0.1\ROLLING_0001.rad")
