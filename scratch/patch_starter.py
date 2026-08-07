import re

with open('pyradioss/starter/starter.py', 'r') as f:
    text = f.read()

text = text.replace(
    'from .initialization import (',
    'from .airbag import initialize_monitored_volumes\nfrom .initialization import ('
)

text = text.replace(
    'initialize_rigid_bodies(model, log)',
    'initialize_rigid_bodies(model, log)\n            initialize_monitored_volumes(model)'
)

with open('pyradioss/starter/starter.py', 'w') as f:
    f.write(text)
