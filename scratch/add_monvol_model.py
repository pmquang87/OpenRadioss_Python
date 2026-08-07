import re

with open("pyradioss/model/model.py", "r", encoding="utf-8") as f:
    text = f.read()

if "self.monitored_volumes" not in text:
    text = text.replace(
        "self.surfaces: Dict[int, Surface] = {}",
        "self.surfaces: Dict[int, Surface] = {}\n        self.monitored_volumes: Dict[int, 'MonitoredVolume'] = {}"
    )
    with open("pyradioss/model/model.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("Added monitored_volumes to Model")
