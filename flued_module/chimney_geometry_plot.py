# """
# Chimney geometry visualization with temperature animation
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.patches import Rectangle, Polygon
# from matplotlib.collections import PatchCollection
# import matplotlib.cm as cm
# from typing import List, Optional, Tuple


# def plot_chimney_geometry(chimney, timesteps_to_plot: Optional[List[int]] = None, 
#                          save_animation: bool = False, fps: int = 10, 
#                          figsize: Tuple[float, float] = (10, 8),
#                          wall_scale: float = 5.0):
#     """
#     Plot chimney geometry with temperature colors animated over time.
    
#     Args:
#         chimney: Chimney object with simulation results
#         timesteps_to_plot: List of timestep indices to animate. If None, samples 20 frames
#         save_animation: Whether to save as animated GIF
#         fps: Frames per second for animation
#         figsize: Figure size (width, height)
#         wall_scale: Factor to scale wall thickness for visibility (default 5x)
#     """
    
#     # Select timesteps to plot
#     if timesteps_to_plot is None:
#         # Sample 20 frames evenly throughout simulation
#         n_frames = min(20, chimney.time_steps)
#         timesteps_to_plot = np.linspace(0, chimney.time_steps, n_frames, dtype=int)
    
#     # Calculate geometry for each segment
#     segment_geometries = []
#     x_pos = 0.0  # Current x position (for horizontal sections)
#     y_pos = 0.0  # Current y position (bottom of current section)
    
#     for part_idx, part in enumerate(chimney.parts):
#         z_start = part.z_m
        
#         for local_seg in range(part.num_segments):
#             seg_height = part.height_m / part.num_segments
            
#             # Calculate inner and outer radii
#             r_inner = part.inner_diameter_m / 2
            
#             if part.type == 'single':
#                 thickness = part.thickness_w_m * wall_scale
#                 r_outer = r_inner + thickness
#             else:  # triple
#                 # Total thickness of all layers
#                 total_thickness = (part.thickness_inner_m + part.thickness_middle_m + 
#                                  part.thickness_outer_m + part.air_gap_inner_m + 
#                                  part.air_gap_outer_m) * wall_scale
#                 r_outer = r_inner + total_thickness
            
#             # Store geometry based on orientation
#             if part.orientation.lower() == 'vertical':
#                 # Vertical segment: x stays same, y increases
#                 seg_geom = {
#                     'type': 'vertical',
#                     'x_center': x_pos,
#                     'y_bottom': y_pos + local_seg * seg_height,
#                     'y_top': y_pos + (local_seg + 1) * seg_height,
#                     'r_inner': r_inner,
#                     'r_outer': r_outer,
#                     'part_idx': part_idx,
#                     'part_type': part.type
#                 }
#             else:  # horizontal
#                 # Horizontal segment: y stays same, x increases
#                 seg_geom = {
#                     'type': 'horizontal',
#                     'x_left': x_pos + local_seg * seg_height,
#                     'x_right': x_pos + (local_seg + 1) * seg_height,
#                     'y_center': y_pos,
#                     'r_inner': r_inner,
#                     'r_outer': r_outer,
#                     'part_idx': part_idx,
#                     'part_type': part.type
#                 }
            
#             segment_geometries.append(seg_geom)
        
#         # Update position for next part
#         if part.orientation.lower() == 'vertical':
#             y_pos += part.height_m
#         else:  # horizontal
#             x_pos += part.height_m
    
#     # Set up figure and axis
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Calculate plot bounds
#     max_x = max([g.get('x_center', 0) + g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'vertical'] +
#                 [g.get('x_right', 0) + g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'horizontal'])
#     min_x = min([g.get('x_center', 0) - g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'vertical'] +
#                 [g.get('x_left', 0) - g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'horizontal'])
#     max_y = max([g.get('y_top', 0) for g in segment_geometries if g['type'] == 'vertical'] +
#                 [g.get('y_center', 0) + g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'horizontal'])
    
#     # Add margins
#     margin_x = 0.1 * (max_x - min_x)
#     margin_y = 0.1 * max_y
#     ax.set_xlim(min_x - margin_x, max_x + margin_x)
#     ax.set_ylim(-margin_y, max_y + margin_y)
#     ax.set_aspect('equal')
    
#     # Initialize patches for walls and gas
#     wall_patches = []
#     gas_patches = []
    
#     for seg_idx, geom in enumerate(segment_geometries):
#         if geom['type'] == 'vertical':
#             # Vertical segment - create rectangles
#             # Outer wall
#             wall_rect = Rectangle((geom['x_center'] - geom['r_outer'], geom['y_bottom']),
#                                 2 * geom['r_outer'], geom['y_top'] - geom['y_bottom'],
#                                 facecolor='gray', edgecolor='black', linewidth=0.5)
#             wall_patches.append(wall_rect)
            
#             # Inner gas
#             gas_rect = Rectangle((geom['x_center'] - geom['r_inner'], geom['y_bottom']),
#                                2 * geom['r_inner'], geom['y_top'] - geom['y_bottom'],
#                                facecolor='red', edgecolor='none')
#             gas_patches.append(gas_rect)
            
#         else:  # horizontal
#             # Horizontal segment - create rectangles
#             # Outer wall
#             wall_rect = Rectangle((geom['x_left'], geom['y_center'] - geom['r_outer']),
#                                 geom['x_right'] - geom['x_left'], 2 * geom['r_outer'],
#                                 facecolor='gray', edgecolor='black', linewidth=0.5)
#             wall_patches.append(wall_rect)
            
#             # Inner gas
#             gas_rect = Rectangle((geom['x_left'], geom['y_center'] - geom['r_inner']),
#                                geom['x_right'] - geom['x_left'], 2 * geom['r_inner'],
#                                facecolor='red', edgecolor='none')
#             gas_patches.append(gas_rect)
    
#     # Add all patches to the plot
#     for patch in wall_patches:
#         ax.add_patch(patch)
#     for patch in gas_patches:
#         ax.add_patch(patch)
    
#     # Set up colormaps
#     gas_cmap = cm.get_cmap('hot')
#     wall_cmap = cm.get_cmap('viridis')
    
#     # Temperature ranges for normalization
#     T_gas_min = np.min(chimney.T_gas[timesteps_to_plot, :]) - 273.15
#     T_gas_max = np.max(chimney.T_gas[timesteps_to_plot, :]) - 273.15
#     T_wall_min = np.min(chimney.T_wall[timesteps_to_plot, :]) - 273.15
#     T_wall_max = np.max(chimney.T_wall[timesteps_to_plot, :]) - 273.15
    
#     # Add colorbars
#     # Gas colorbar
#     sm_gas = cm.ScalarMappable(cmap=gas_cmap, 
#                                norm=plt.Normalize(vmin=T_gas_min, vmax=T_gas_max))
#     sm_gas.set_array([])
#     cbar_gas = plt.colorbar(sm_gas, ax=ax, pad=0.02, fraction=0.046, location='right')
#     cbar_gas.set_label('Gas Temperature (°C)', rotation=270, labelpad=20)
    
#     # Wall colorbar
#     sm_wall = cm.ScalarMappable(cmap=wall_cmap, 
#                                 norm=plt.Normalize(vmin=T_wall_min, vmax=T_wall_max))
#     sm_wall.set_array([])
#     cbar_wall = plt.colorbar(sm_wall, ax=ax, pad=0.12, fraction=0.046, location='right')
#     cbar_wall.set_label('Wall Temperature (°C)', rotation=270, labelpad=20)
    
#     # Labels
#     ax.set_xlabel('Horizontal Distance (m)')
#     ax.set_ylabel('Height (m)')
#     ax.grid(True, alpha=0.3)
    
#     # Title with time
#     title = ax.set_title(f'Chimney Temperature Distribution - Time: 0.0 s')
    
#     def update(frame_idx):
#         """Update function for animation"""
#         timestep = timesteps_to_plot[frame_idx]
#         time_s = timestep * chimney.dt
        
#         # Update title
#         title.set_text(f'Chimney Temperature Distribution - Time: {time_s:.1f} s')
        
#         # Update colors for each segment
#         for seg_idx in range(chimney.total_segments):
#             # Get temperatures
#             T_gas_C = chimney.T_gas[timestep, seg_idx] - 273.15
#             T_wall_C = chimney.T_wall[timestep, seg_idx] - 273.15
            
#             # Normalize temperatures to [0, 1]
#             gas_norm = (T_gas_C - T_gas_min) / (T_gas_max - T_gas_min + 1e-6)
#             wall_norm = (T_wall_C - T_wall_min) / (T_wall_max - T_wall_min + 1e-6)
            
#             # Get colors
#             gas_color = gas_cmap(gas_norm)
#             wall_color = wall_cmap(wall_norm)
            
#             # Update patch colors
#             wall_patches[seg_idx].set_facecolor(wall_color)
#             gas_patches[seg_idx].set_facecolor(gas_color)
        
#         return wall_patches + gas_patches + [title]
    
#     # Create animation
#     anim = animation.FuncAnimation(fig, update, frames=len(timesteps_to_plot),
#                                   interval=1000/fps, blit=True)
    
#     if save_animation:
#         # Save as GIF
#         anim.save('chimney_temperature_animation.gif', writer='pillow', fps=fps)
#         print("Animation saved as 'chimney_temperature_animation.gif'")
    
#     plt.tight_layout()
#     return fig, anim


# def plot_chimney_snapshot(chimney, timestep: int, figsize: Tuple[float, float] = (10, 8),
#                          wall_scale: float = 5.0, save_path: Optional[str] = None):
#     """
#     Plot a single snapshot of the chimney geometry with temperatures.
    
#     Args:
#         chimney: Chimney object with simulation results
#         timestep: Timestep index to plot
#         figsize: Figure size (width, height)
#         wall_scale: Factor to scale wall thickness for visibility
#         save_path: Optional path to save the figure
#     """
    
#     # Use the same geometry calculation as the animation
#     segment_geometries = []
#     x_pos = 0.0
#     y_pos = 0.0
    
#     for part_idx, part in enumerate(chimney.parts):
#         for local_seg in range(part.num_segments):
#             seg_height = part.height_m / part.num_segments
            
#             r_inner = part.inner_diameter_m / 2
            
#             if part.type == 'single':
#                 thickness = part.thickness_w_m * wall_scale
#                 r_outer = r_inner + thickness
#             else:
#                 total_thickness = (part.thickness_inner_m + part.thickness_middle_m + 
#                                  part.thickness_outer_m + part.air_gap_inner_m + 
#                                  part.air_gap_outer_m) * wall_scale
#                 r_outer = r_inner + total_thickness
            
#             if part.orientation.lower() == 'vertical':
#                 seg_geom = {
#                     'type': 'vertical',
#                     'x_center': x_pos,
#                     'y_bottom': y_pos + local_seg * seg_height,
#                     'y_top': y_pos + (local_seg + 1) * seg_height,
#                     'r_inner': r_inner,
#                     'r_outer': r_outer,
#                 }
#             else:
#                 seg_geom = {
#                     'type': 'horizontal',
#                     'x_left': x_pos + local_seg * seg_height,
#                     'x_right': x_pos + (local_seg + 1) * seg_height,
#                     'y_center': y_pos,
#                     'r_inner': r_inner,
#                     'r_outer': r_outer,
#                 }
            
#             segment_geometries.append(seg_geom)
        
#         if part.orientation.lower() == 'vertical':
#             y_pos += part.height_m
#         else:
#             x_pos += part.height_m
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Set bounds
#     max_x = max([g.get('x_center', 0) + g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'vertical'] +
#                 [g.get('x_right', 0) + g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'horizontal'])
#     min_x = min([g.get('x_center', 0) - g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'vertical'] +
#                 [g.get('x_left', 0) - g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'horizontal'])
#     max_y = max([g.get('y_top', 0) for g in segment_geometries if g['type'] == 'vertical'] +
#                 [g.get('y_center', 0) + g.get('r_outer', 0) for g in segment_geometries if g['type'] == 'horizontal'])
    
#     margin_x = 0.1 * (max_x - min_x)
#     margin_y = 0.1 * max_y
#     ax.set_xlim(min_x - margin_x, max_x + margin_x)
#     ax.set_ylim(-margin_y, max_y + margin_y)
#     ax.set_aspect('equal')
    
#     # Get temperature ranges
#     T_gas_C = chimney.T_gas[timestep, :] - 273.15
#     T_wall_C = chimney.T_wall[timestep, :] - 273.15
    
#     gas_cmap = cm.get_cmap('hot')
#     wall_cmap = cm.get_cmap('viridis')
    
#     # Draw segments
#     for seg_idx, geom in enumerate(segment_geometries):
#         # Get colors based on temperature
#         gas_color = gas_cmap((T_gas_C[seg_idx] - T_gas_C.min()) / (T_gas_C.max() - T_gas_C.min() + 1e-6))
#         wall_color = wall_cmap((T_wall_C[seg_idx] - T_wall_C.min()) / (T_wall_C.max() - T_wall_C.min() + 1e-6))
        
#         if geom['type'] == 'vertical':
#             # Wall
#             wall_rect = Rectangle((geom['x_center'] - geom['r_outer'], geom['y_bottom']),
#                                 2 * geom['r_outer'], geom['y_top'] - geom['y_bottom'],
#                                 facecolor=wall_color, edgecolor='black', linewidth=0.5)
#             ax.add_patch(wall_rect)
            
#             # Gas
#             gas_rect = Rectangle((geom['x_center'] - geom['r_inner'], geom['y_bottom']),
#                                2 * geom['r_inner'], geom['y_top'] - geom['y_bottom'],
#                                facecolor=gas_color, edgecolor='none')
#             ax.add_patch(gas_rect)
            
#         else:
#             # Wall
#             wall_rect = Rectangle((geom['x_left'], geom['y_center'] - geom['r_outer']),
#                                 geom['x_right'] - geom['x_left'], 2 * geom['r_outer'],
#                                 facecolor=wall_color, edgecolor='black', linewidth=0.5)
#             ax.add_patch(wall_rect)
            
#             # Gas
#             gas_rect = Rectangle((geom['x_left'], geom['y_center'] - geom['r_inner']),
#                                geom['x_right'] - geom['x_left'], 2 * geom['r_inner'],
#                                facecolor=gas_color, edgecolor='none')
#             ax.add_patch(gas_rect)
    
#     # Add colorbars
#     sm_gas = cm.ScalarMappable(cmap=gas_cmap, 
#                                norm=plt.Normalize(vmin=T_gas_C.min(), vmax=T_gas_C.max()))
#     sm_gas.set_array([])
#     cbar_gas = plt.colorbar(sm_gas, ax=ax, pad=0.02, fraction=0.046)
#     cbar_gas.set_label('Gas Temperature (°C)', rotation=270, labelpad=20)
    
#     sm_wall = cm.ScalarMappable(cmap=wall_cmap, 
#                                 norm=plt.Normalize(vmin=T_wall_C.min(), vmax=T_wall_C.max()))
#     sm_wall.set_array([])
#     cbar_wall = plt.colorbar(sm_wall, ax=ax, pad=0.12, fraction=0.046)
#     cbar_wall.set_label('Wall Temperature (°C)', rotation=270, labelpad=20)
    
#     # Labels
#     ax.set_xlabel('Horizontal Distance (m)')
#     ax.set_ylabel('Height (m)')
#     ax.set_title(f'Chimney Temperature Distribution - Time: {timestep * chimney.dt:.1f} s')
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Figure saved to {save_path}")
    
#     return fig, ax