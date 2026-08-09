with open('pyradioss/engine/engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace(
    'sensors = Sensors(model, log)          # /SENSOR (M6)\n    if resumed:',
    'sensors = Sensors(model, log)          # /SENSOR (M6)\n    model.sensors_state = sensors\n    if resumed:'
)

text = text.replace(
    'dt = min(dt, controls.t_end - state.t)  # land exactly on t_end',
    'dt = min(dt, controls.t_end - state.t)  # land exactly on t_end\n        model.t = state.t'
)

with open('pyradioss/engine/engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
