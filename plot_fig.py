# %%
import plotly.graph_objects as go

group = ['G1', 'G2', 'G3', 'G4']

fig = go.Figure()
fig.add_trace(go.Bar(
    name='L1',
    x=group, y=[0.879, 0.879, 0.818, 0.829, ],
    error_y=dict(type='data', array=[0.076, 0.040, 0.014, 0.015, ])
))

fig.add_trace(go.Bar(
    name='L2',
    x=group, y=[0.872, 0.879, 0.808, 0.802, ],
    error_y=dict(type='data', array=[0.068, 0.040, 0.049, 0.018, ])
))

fig.add_trace(go.Bar(
    name='L3',
    x=group, y=[0.882, 0.859, 0.788, 0.797, ],
    error_y=dict(type='data', array=[0.051, 0.047, 0.030, 0.038, ])
))

fig.update_layout(barmode='group')
fig.show()
