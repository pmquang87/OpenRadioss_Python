import re
import numpy as np

def parse_data(filename):
    with open(filename, 'r') as f:
        text = f.read()

    blocks = {}
    current_name = None
    current_data = []

    for line in text.splitlines():
        if "DATA" in line and "/" in line:
            m = re.search(r"DATA\s+(\w+)\s+/", line)
            if m:
                if current_name:
                    blocks[current_name] = current_data
                current_name = m.group(1)
                current_data = []
        elif current_name:
            # parse the line
            line = line.strip()
            if not line: continue
            if "C-----" in line: continue
            if ">" in line: line = line.split(">", 1)[1]
            if not line: continue
            
            # format: 1 2. , 0., 0.,
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                vals = parts[1].split(',')
                for v in vals:
                    v = v.strip()
                    if v and '/' not in v:
                        current_data.append(float(v))
                    elif '/' in v:
                        v = v.replace('/', '').strip()
                        if v: current_data.append(float(v))
                        
    if current_name:
        blocks[current_name] = current_data

    for name, data in blocks.items():
        arr = np.array(data).reshape(9, 9)
        print(f"{name} = np.array([")
        for row in arr:
            print("    [" + ", ".join(f"{x:18.15f}" for x in row) + "],")
        print("])\n")

parse_data('scratch/s16_gauss.txt')
