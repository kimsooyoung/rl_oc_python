import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

platform = None 
if os.name == 'nt':
    platform = "Windows"

l = 0.5

fig = plt.figure(figsize=(5, 5))
animation_ax = plt.axes()
animation_plots = []

animation_ax.set_xlim(-2, 2)
animation_ax.set_ylim(-2, 2)

line, = animation_ax.plot([], [], 'bo')
(bar_plot,) = animation_ax.plot([], [], "-", lw=5, color="black")
(ee_plot,) = animation_ax.plot([], [], "o", markersize=10.0, color="blue")
text_plot = animation_ax.text(0.15, 0.85, [], fontsize=10, transform=fig.transFigure)

animation_plots.append(bar_plot)
animation_plots.append(ee_plot)
animation_plots.append(text_plot)

def update(frame):
    x = [0.0,  l*np.sin(frame)]
    y = [0.0, -l*np.cos(frame)]
    animation_plots[0].set_data(x, y)
    # for linux
    if platform == "Windows":
        animation_plots[1].set_data([x[1]], [y[1]])
    else:
        animation_plots[1].set_data(x[1], y[1])
    animation_plots[2].set_text(f"t = {frame}")
    return animation_plots

ani = FuncAnimation(
    fig, 
    update, 
    frames=np.linspace(0, 2*np.pi, 128),
    blit=True,
    repeat=False,
    interval=100,
)
plt.show()
