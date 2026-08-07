import re

with open("pyradioss/model/entities.py", "r", encoding="utf-8") as f:
    text = f.read()

new_classes = """
@dataclass
class MonitoredVolume:
    \"\"\"A /MONVOL monitored volume (e.g., AIRBAG1).\"\"\"
    id: int
    title: str = ""
    vol_type: str = "AIRBAG1"  # "AIRBAG1", "COMMU", etc.
    surf_id: int = 0
    hconv: float = 0.0
    
    # Material properties
    matid: int = 0
    mu: float = 0.0
    pext: float = 0.0
    t_initial: float = 293.0
    iequil: int = 0
    ittf: int = 0
    
    # Scaling factors (typically 1.0)
    scale_t: float = 1.0
    scale_p: float = 1.0
    scale_s: float = 1.0
    scale_a: float = 1.0
    scale_d: float = 1.0

    # Injectors (jets)
    injectors: List[Dict] = field(default_factory=list)
    # Ventholes and porous surfaces
    vents: List[Dict] = field(default_factory=list)
    porous_surfaces: List[Dict] = field(default_factory=list)

"""

if "class MonitoredVolume" not in text:
    text += new_classes
    with open("pyradioss/model/entities.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Added MonitoredVolume to entities.py")
else:
    print("Already exists")
