import visilibity as vis
import matplotlib.pyplot as plt

# Set up the global variables
epsilon = 0.0000001
snap_dis = 10
boundary_points = []
observer = None
isovist = None
polygon_complete = False

# Define the limits for the plot
x_min, x_max = 0, 1000
y_min, y_max = 0, 1000

def on_click(event):
    global boundary_points, polygon_complete
    
    if event.button == 1 and not polygon_complete:  # Left-click to add points to the boundary
        boundary_points.append(vis.Point(event.xdata, event.ydata))
        update_plot()

    # Right-click to complete the polygon
    elif event.button == 3 and len(boundary_points) > 2 and not polygon_complete:
        polygon_complete = True
        boundary_points.append(vis.Point(boundary_points[0].x(), boundary_points[0].y()))  # Close the polygon
        update_plot()

def on_mouse_move(event):
    global observer, isovist, env
    
    if not polygon_complete or event.xdata is None or event.ydata is None:
        return
    
    observer = vis.Point(event.xdata, event.ydata)
    
    if len(boundary_points) > 2 and polygon_complete:  # We need at least three points for a valid polygon
        walls = vis.Polygon(boundary_points)
        env = vis.Environment([walls])
        observer.snap_to_boundary_of(env, snap_dis)
        observer.snap_to_vertices_of(env, snap_dis)
        isovist = vis.Visibility_Polygon(observer, env, epsilon)
    
    update_plot()

def update_plot():
    global observer, isovist
    
    ax.clear()
    
    # Draw the boundary
    if len(boundary_points) > 0:
        x = [p.x() for p in boundary_points]
        y = [p.y() for p in boundary_points]
        ax.plot(x, y, 'black')
    
    # Draw the observer
    if observer:
        ax.plot(observer.x(), observer.y(), 'go')
    
    # Draw the visibility polygon
    if isovist:
        x = [isovist[j].x() for j in range(isovist.n())] + [isovist[0].x()]
        y = [isovist[j].y() for j in range(isovist.n())] + [isovist[0].y()]
        ax.plot(x, y, 'blue')

        child = isovist.get_growing_vertices()
        child_x = [child[j].x() for j in range(child.n())]
        child_y = [child[j].y() for j in range(child.n())]
        ax.plot(child_x, child_y, 'g*')
    
    # Maintain the axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.draw()

def reset(event):
    global boundary_points, observer, isovist, polygon_complete
    boundary_points = []
    observer = None
    isovist = None
    polygon_complete = False
    update_plot()

# Initialize the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_title('Left-click to add wall points, Right-click to complete polygon, Move mouse to set observer')

# Set the initial axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Add a reset button
reset_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
reset_button = plt.Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
reset_button.on_clicked(reset)

# Connect the click and mouse move events
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Start with an empty plot
update_plot()

plt.show()
