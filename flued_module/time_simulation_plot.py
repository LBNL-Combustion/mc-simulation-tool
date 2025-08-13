"""
Time simulation plotting for chimney - shows radial temperature distribution over time
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import Optional

def plot_time_simulation(chimney, timesteps_to_plot: Optional[list] = None, 
                        save_animation: bool = False, fps: int = 10,
                        figsize: tuple = (10, 8)):
    """
    Create a plot showing radial temperature distribution evolving over time
    
    Args:
        chimney: Chimney object with simulation results
        timesteps_to_plot: List of timestep indices to plot. If None, plots every 10th step
        save_animation: Whether to save as animated gif
        fps: Frames per second for animation
        figsize: Figure size tuple
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    if timesteps_to_plot is None:
        # Default: plot every 10th timestep
        timesteps_to_plot = list(range(0, chimney.time_steps + 1, 10))
        if chimney.time_steps not in timesteps_to_plot:
            timesteps_to_plot.append(chimney.time_steps)
    
    # Calculate segment heights
    segment_heights = []
    segment_centers = []
    
    for seg in range(chimney.total_segments):
        part_idx, local_seg, part = chimney.get_segment_info(seg)
        seg_height = part.height_m / part.num_segments
        bottom = part.z_m + local_seg * seg_height
        center = bottom + seg_height / 2
        segment_heights.append(seg_height)
        segment_centers.append(center)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for temperature
    cmap = plt.cm.hot
    
    # Temperature range for consistent coloring
    T_min = chimney.parts[0].T_ambient_K - 273.15  # Ambient in Celsius
    T_max = np.max(chimney.T_gas) - 273.15  # Max gas temp in Celsius
    
    def draw_frame(step_idx):
        """Draw a single frame of the animation"""
        ax.clear()
        
        step = timesteps_to_plot[step_idx]
        time = step * chimney.dt
        
        # For each segment
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            
            # Get temperatures
            T_gas_C = chimney.T_gas[step, seg] - 273.15
            T_ambient_C = part.T_ambient_K - 273.15
            
            # Segment geometry
            y_bottom = segment_centers[seg] - segment_heights[seg] / 2
            y_top = segment_centers[seg] + segment_heights[seg] / 2
            
            # Radial positions
            r_inner = part.inner_diameter_m / 2
            
            if part.type == 'single':
                # Single wall
                r_wall_inner = r_inner
                r_wall_outer = r_inner + part.thickness_w_m
                
                # Make thin walls visible (minimum 5% of radius for visibility)
                wall_thickness_display = max(part.thickness_w_m, 0.05 * r_inner)
                r_wall_outer_display = r_inner + wall_thickness_display
                
                T_wall_C = chimney.T_wall[step, seg] - 273.15
                
                # Draw gas region
                gas_color = cmap((T_gas_C - T_min) / (T_max - T_min + 1e-6))
                gas_rect = Rectangle((-r_inner, y_bottom), 2 * r_inner, y_top - y_bottom,
                                   facecolor=gas_color, edgecolor='none')
                ax.add_patch(gas_rect)
                
                # Draw walls
                wall_color = cmap((T_wall_C - T_min) / (T_max - T_min + 1e-6))
                # Left wall
                wall_rect_left = Rectangle((-r_wall_outer_display, y_bottom), 
                                         wall_thickness_display, y_top - y_bottom,
                                         facecolor=wall_color, edgecolor='black', linewidth=0.5)
                ax.add_patch(wall_rect_left)
                # Right wall
                wall_rect_right = Rectangle((r_inner, y_bottom), 
                                          wall_thickness_display, y_top - y_bottom,
                                          facecolor=wall_color, edgecolor='black', linewidth=0.5)
                ax.add_patch(wall_rect_right)
                
            else:  # triple wall
                # Get all temperatures
                T_liner_C = chimney.T_liner[step, seg] - 273.15
                T_middle_C = chimney.T_middle[step, seg] - 273.15
                T_outer_C = chimney.T_outer[step, seg] - 273.15
                
                # Calculate radii
                r_liner_outer = part.liner_outer_diameter_m / 2
                r_middle_inner = part.middle_inner_diameter_m / 2
                r_middle_outer = part.middle_outer_diameter_m / 2
                r_outer_inner = part.outer_inner_diameter_m / 2
                r_outer_outer = part.outer_diameter_m / 2
                
                # Make layers visible (minimum thickness for display)
                min_layer_thickness = 0.02 * r_inner
                
                # Adjust for display
                liner_thickness_display = max(part.thickness_inner_m, min_layer_thickness)
                middle_thickness_display = max(part.thickness_middle_m, min_layer_thickness * 2)
                outer_thickness_display = max(part.thickness_outer_m, min_layer_thickness)
                
                r_liner_outer_display = r_inner + liner_thickness_display
                r_middle_outer_display = r_liner_outer_display + middle_thickness_display
                r_outer_outer_display = r_middle_outer_display + outer_thickness_display
                
                # Draw gas region
                gas_color = cmap((T_gas_C - T_min) / (T_max - T_min + 1e-6))
                gas_rect = Rectangle((-r_inner, y_bottom), 2 * r_inner, y_top - y_bottom,
                                   facecolor=gas_color, edgecolor='none')
                ax.add_patch(gas_rect)
                
                # Draw liner (left and right)
                liner_color = cmap((T_liner_C - T_min) / (T_max - T_min + 1e-6))
                liner_left = Rectangle((-r_liner_outer_display, y_bottom), 
                                     liner_thickness_display, y_top - y_bottom,
                                     facecolor=liner_color, edgecolor='black', linewidth=0.5)
                liner_right = Rectangle((r_inner, y_bottom), 
                                      liner_thickness_display, y_top - y_bottom,
                                      facecolor=liner_color, edgecolor='black', linewidth=0.5)
                ax.add_patch(liner_left)
                ax.add_patch(liner_right)
                
                # Draw middle layer (insulation)
                middle_color = cmap((T_middle_C - T_min) / (T_max - T_min + 1e-6))
                middle_left = Rectangle((-r_middle_outer_display, y_bottom), 
                                      middle_thickness_display, y_top - y_bottom,
                                      facecolor=middle_color, edgecolor='black', linewidth=0.5)
                middle_right = Rectangle((r_liner_outer_display, y_bottom), 
                                       middle_thickness_display, y_top - y_bottom,
                                       facecolor=middle_color, edgecolor='black', linewidth=0.5)
                ax.add_patch(middle_left)
                ax.add_patch(middle_right)
                
                # Draw outer wall
                outer_color = cmap((T_outer_C - T_min) / (T_max - T_min + 1e-6))
                outer_left = Rectangle((-r_outer_outer_display, y_bottom), 
                                     outer_thickness_display, y_top - y_bottom,
                                     facecolor=outer_color, edgecolor='black', linewidth=0.5)
                outer_right = Rectangle((r_middle_outer_display, y_bottom), 
                                      outer_thickness_display, y_top - y_bottom,
                                      facecolor=outer_color, edgecolor='black', linewidth=0.5)
                ax.add_patch(outer_left)
                ax.add_patch(outer_right)
            
            # Draw ambient air outside
            ambient_color = cmap((T_ambient_C - T_min) / (T_max - T_min + 1e-6))
            max_r = r_outer_outer_display if part.type == 'triple' else r_wall_outer_display
            
            # Left ambient
            ambient_left = Rectangle((-max_r * 1.5, y_bottom), 
                                   max_r * 0.5, y_top - y_bottom,
                                   facecolor=ambient_color, edgecolor='none', alpha=0.3)
            # Right ambient
            ambient_right = Rectangle((max_r, y_bottom), 
                                    max_r * 0.5, y_top - y_bottom,
                                    facecolor=ambient_color, edgecolor='none', alpha=0.3)
            ax.add_patch(ambient_left)
            ax.add_patch(ambient_right)
        
        # Set axis properties
        max_radius = max(part.outer_diameter_m / 2 * 1.2 for part in chimney.parts)
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(0, sum(part.height_m for part in chimney.parts))
        
        ax.set_xlabel('Radial Distance from Center (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title(f'Temperature Distribution at t = {time:.1f}s')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=T_min, vmax=T_max))
        sm.set_array([])
        if step_idx == 0:  # Only add colorbar once
            cbar = fig.colorbar(sm, ax=ax, label='Temperature (°C)')
        
        # Add annotations
        ax.text(0, ax.get_ylim()[1] * 0.95, 'Gas Flow', ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add centerline
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Add labels for different regions
        if chimney.parts[0].type == 'single':
            ax.text(0, -0.1, 'Gas', ha='center', va='top', fontsize=8)
            ax.text(chimney.parts[0].inner_diameter_m/2 + 0.01, -0.1, 'Wall', 
                   ha='left', va='top', fontsize=8)
        else:
            ax.text(0, -0.1, 'Gas', ha='center', va='top', fontsize=8)
            ax.text(chimney.parts[0].inner_diameter_m/2 + 0.01, -0.1, 'Liner', 
                   ha='left', va='top', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        
    # Create animation or static plots
    if save_animation and len(timesteps_to_plot) > 1:
        ani = animation.FuncAnimation(fig, draw_frame, frames=len(timesteps_to_plot),
                                    interval=1000/fps, blit=False)
        ani.save('chimney_temperature_evolution.gif', writer='pillow', fps=fps)
        print(f"Animation saved as chimney_temperature_evolution.gif")
    else:
        # Just draw the last frame
        draw_frame(len(timesteps_to_plot) - 1)
    
    return fig, ax


def plot_time_simulation_snapshots(chimney, num_snapshots: int = 6, 
                                 figsize: tuple = (15, 10)):
    """
    Create a grid of snapshots showing temperature evolution
    
    Args:
        chimney: Chimney object with simulation results
        num_snapshots: Number of time snapshots to show
        figsize: Figure size tuple
    
    Returns:
        fig: matplotlib figure object
    """
    
    # Select timesteps evenly spaced
    timesteps = np.linspace(0, chimney.time_steps, num_snapshots, dtype=int)
    
    # Create subplot grid
    cols = 3
    rows = int(np.ceil(num_snapshots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                           sharex=True, sharey=True)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Temperature range for consistent coloring
    T_min = chimney.parts[0].T_ambient_K - 273.15
    T_max = np.max(chimney.T_gas) - 273.15
    
    # Color map
    cmap = plt.cm.hot
    
    for idx, step in enumerate(timesteps):
        ax = axes_flat[idx]
        time = step * chimney.dt
        
        # Draw each segment
        for seg in range(chimney.total_segments):
            part_idx, local_seg, part = chimney.get_segment_info(seg)
            
            # Get temperatures
            T_gas_C = chimney.T_gas[step, seg] - 273.15
            
            # Segment geometry
            seg_height = part.height_m / part.num_segments
            y_bottom = part.z_m + local_seg * seg_height
            y_top = y_bottom + seg_height
            
            # Radial positions
            r_inner = part.inner_diameter_m / 2
            
            if part.type == 'single':
                r_wall_outer = r_inner + max(part.thickness_w_m, 0.05 * r_inner)
                T_wall_C = chimney.T_wall[step, seg] - 273.15
                
                # Draw gas
                gas_color = cmap((T_gas_C - T_min) / (T_max - T_min + 1e-6))
                gas_rect = Rectangle((-r_inner, y_bottom), 2 * r_inner, seg_height,
                                   facecolor=gas_color, edgecolor='none')
                ax.add_patch(gas_rect)
                
                # Draw walls
                wall_color = cmap((T_wall_C - T_min) / (T_max - T_min + 1e-6))
                for x in [-r_wall_outer, r_inner]:
                    wall_rect = Rectangle((x, y_bottom), r_wall_outer - r_inner, seg_height,
                                        facecolor=wall_color, edgecolor='black', linewidth=0.5)
                    ax.add_patch(wall_rect)
            
            else:  # triple wall
                # Similar drawing logic but simplified for snapshots
                T_liner_C = chimney.T_liner[step, seg] - 273.15
                T_middle_C = chimney.T_middle[step, seg] - 273.15
                T_outer_C = chimney.T_outer[step, seg] - 273.15
                
                # Draw gas
                gas_color = cmap((T_gas_C - T_min) / (T_max - T_min + 1e-6))
                gas_rect = Rectangle((-r_inner, y_bottom), 2 * r_inner, seg_height,
                                   facecolor=gas_color, edgecolor='none')
                ax.add_patch(gas_rect)
                
                # Simplified triple wall drawing
                thicknesses = [0.02, 0.04, 0.02]  # Display thicknesses
                temps = [T_liner_C, T_middle_C, T_outer_C]
                r_current = r_inner
                
                for t_idx, (thick, temp) in enumerate(zip(thicknesses, temps)):
                    color = cmap((temp - T_min) / (T_max - T_min + 1e-6))
                    for x in [-r_current - thick * r_inner, r_current]:
                        rect = Rectangle((x, y_bottom), thick * r_inner, seg_height,
                                       facecolor=color, edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)
                    r_current += thick * r_inner
        
        # Set axis properties
        max_radius = max(part.outer_diameter_m / 2 * 1.2 for part in chimney.parts)
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(0, sum(part.height_m for part in chimney.parts))
        ax.set_title(f't = {time:.1f}s')
        ax.grid(True, alpha=0.3)
        
        if idx >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel('Radial Distance (m)')
        if idx % cols == 0:  # Left column
            ax.set_ylabel('Height (m)')
    
    # Remove unused subplots
    for idx in range(num_snapshots, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=T_min, vmax=T_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label='Temperature (°C)', pad=0.02)
    
    fig.suptitle('Chimney Temperature Evolution', fontsize=14)
    plt.tight_layout()
    
    return fig