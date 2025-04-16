import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from io import StringIO
# from stl import mesh
# import io

# Calculation Functions
def elliptical_calculator(target_volume, cylinder_radius, ellipse_ratio):
    target_volume = target_volume * 1e6 # litre to mm3
    head_height = ellipse_ratio * cylinder_radius
    head_volume = (2/3) * math.pi * head_height * (cylinder_radius**2)
    cylinder_volume = target_volume - (2 * head_volume)
    if cylinder_volume <= 0:
        raise ValueError("Head volume too large for target volume. Increase cylinder radius or reduce ellipse ratio.")
    cylinder_length = cylinder_volume / (math.pi * (cylinder_radius ** 2))
    return head_height, head_volume, cylinder_length 

def torispherical_calculator(target_volume, cylinder_radius, crown_radius, knuckle_radius):
    if crown_radius < cylinder_radius:
        raise ValueError("Crown radius must be greater than or equal to cylinder radius")
    if knuckle_radius >= cylinder_radius:
        raise ValueError("Knuckle radius must be less than or equal to cylinder radius")
    
    h = crown_radius - math.sqrt(crown_radius**2 - cylinder_radius**2)
    V_tori_head = (math.pi * h/3) * (3 * cylinder_radius**2 + h**2)
    target_volume_mm3 = target_volume * 1e6
    V_cylinder = target_volume_mm3 - (2 * V_tori_head)
    length = V_cylinder / (math.pi * (cylinder_radius ** 2))
    return h, V_tori_head, length

def hydrogen_amount_calculator(pressure, volume, temperature):
    R = 8.314
    temperature_kelvin = temperature + 273.15
    volume = volume * 0.001
    pressure = pressure * 1e5
    n = (pressure * volume) / (R * temperature_kelvin)
    m = n * (2.016 * 1e-3)
    return m

# Visualization Functions with Thickness
def generate_cylinder_with_thickness(radius, height, thickness, num_points=50):
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(0, height, num_points)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_outer = radius * np.cos(theta_grid)
    y_outer = radius * np.sin(theta_grid)
    x_inner = (radius - thickness) * np.cos(theta_grid)
    y_inner = (radius - thickness) * np.sin(theta_grid)
    
    return x_outer, y_outer, z_grid, x_inner, y_inner, z_grid


# Elliptical Function for Top Ellipsoidal 


def generate_elliptical_head_with_opening_top(radius, head_height, thickness, center_z, Rto, num_points=50, flipped=False):
    # Define parametric grid
    u = np.linspace(0, np.pi/2, num_points)  # Latitude (0 to 90 degrees)
    v = np.linspace(0, 2*np.pi, num_points) # Longitude (0 to 360 degrees)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Error Handling
    if thickness >= radius or thickness >= head_height:
        raise ValueError("Thickness cannot be greater than or equal to radius or head height.")

    # Inner surface
    x_inner = radius * np.sin(u_grid) * np.cos(v_grid)
    y_inner = radius * np.sin(u_grid) * np.sin(v_grid)
    z_inner = center_z + head_height * np.cos(u_grid)
    
    # Outer surface
    outer_radius = radius + thickness
    outer_height = head_height + thickness

    x_outer = outer_radius * np.sin(u_grid) * np.cos(v_grid)
    y_outer = outer_radius * np.sin(u_grid) * np.sin(v_grid)
    z_outer = center_z + outer_height * np.cos(u_grid)
    
    # Apply the cylindrical cutout condition
    mask_inner = (x_inner**2 + y_inner**2) < Rto**2
    mask_outer = (x_outer**2 + y_outer**2) < Rto**2
    
    x_inner[mask_inner] = np.nan
    y_inner[mask_inner] = np.nan
    z_inner[mask_inner] = np.nan

    x_outer[mask_outer] = np.nan
    y_outer[mask_outer] = np.nan
    z_outer[mask_outer] = np.nan
    
    # Flip if needed
    if flipped:
        z_outer = 2 * center_z - z_outer
        z_inner = 2 * center_z - z_inner
    
    return x_outer, y_outer, z_outer, x_inner, y_inner, z_inner


# Elliptical Function for Bottom Ellipsoidal


def generate_elliptical_head_with_opening_bottom(radius, head_height, thickness, center_z, Rbo, num_points=50, flipped=False):
    # Define parametric grid
    u = np.linspace(0, np.pi/2, num_points)  # Latitude (0 to 90 degrees)
    v = np.linspace(0, 2*np.pi, num_points) # Longitude (0 to 360 degrees)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Error Handling
    if thickness >= radius or thickness >= head_height:
        raise ValueError("Thickness cannot be greater than or equal to radius or head height.")

    # Inner surface
    x_inner = radius * np.sin(u_grid) * np.cos(v_grid)
    y_inner = radius * np.sin(u_grid) * np.sin(v_grid)
    z_inner = center_z + head_height * np.cos(u_grid)
    
    # Outer surface
    outer_radius = radius + thickness
    outer_height = head_height + thickness

    x_outer = outer_radius * np.sin(u_grid) * np.cos(v_grid)
    y_outer = outer_radius * np.sin(u_grid) * np.sin(v_grid)
    z_outer = center_z + outer_height * np.cos(u_grid)
    
    # Apply the cylindrical cutout condition
    mask_inner = (x_inner**2 + y_inner**2) < Rbo**2
    mask_outer = (x_outer**2 + y_outer**2) < Rbo**2
    
    x_inner[mask_inner] = np.nan
    y_inner[mask_inner] = np.nan
    z_inner[mask_inner] = np.nan

    x_outer[mask_outer] = np.nan
    y_outer[mask_outer] = np.nan
    z_outer[mask_outer] = np.nan
    
    # Flip if needed
    if flipped:
        z_outer = 2 * center_z - z_outer
        z_inner = 2 * center_z - z_inner
    
    return x_outer, y_outer, z_outer, x_inner, y_inner, z_inner


# Torispherical Dome Top Opening Function

def generate_torispherical_head_with_top_opening(cylinder_radius, crown_radius, knuckle_radius, thickness, center_z, Rto, num_points=50, flipped=False):
    if crown_radius < cylinder_radius:
        crown_radius = cylinder_radius
    
    # Outer surface calculations
    sin_alpha = cylinder_radius / crown_radius
    alpha = math.asin(sin_alpha)
    h = crown_radius - math.sqrt(crown_radius**2 - cylinder_radius**2)
    
    u_crown = np.linspace(0, alpha, num_points)
    v_crown = np.linspace(0, 2 * np.pi, num_points)
    u_grid_crown, v_grid_crown = np.meshgrid(u_crown, v_crown)
    
    x_crown_outer = crown_radius * np.sin(u_grid_crown) * np.cos(v_grid_crown)
    y_crown_outer = crown_radius * np.sin(u_grid_crown) * np.sin(v_grid_crown)
    z_crown_outer = center_z + crown_radius * np.cos(u_grid_crown) - crown_radius + h
    
    phi = np.linspace(alpha, math.pi/2, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    x_knuckle_outer = knuckle_radius * np.cos(phi_grid) * np.cos(theta_grid)
    y_knuckle_outer = knuckle_radius * np.cos(phi_grid) * np.sin(theta_grid)
    z_knuckle_outer = center_z + knuckle_radius * np.sin(phi_grid) - knuckle_radius + h
    
    # Inner surface calculations
    inner_crown_radius = crown_radius - thickness
    inner_cylinder_radius = cylinder_radius - thickness
    inner_knuckle_radius = knuckle_radius - thickness
    inner_h = inner_crown_radius - math.sqrt(inner_crown_radius**2 - inner_cylinder_radius**2)
    
    x_crown_inner = inner_crown_radius * np.sin(u_grid_crown) * np.cos(v_grid_crown)
    y_crown_inner = inner_crown_radius * np.sin(u_grid_crown) * np.sin(v_grid_crown)
    z_crown_inner = center_z + inner_crown_radius * np.cos(u_grid_crown) - inner_crown_radius + inner_h
    
    x_knuckle_inner = inner_knuckle_radius * np.cos(phi_grid) * np.cos(theta_grid)
    y_knuckle_inner = inner_knuckle_radius * np.cos(phi_grid) * np.sin(theta_grid)
    z_knuckle_inner = center_z + inner_knuckle_radius * np.sin(phi_grid) - inner_knuckle_radius + inner_h

    # Apply cylindrical cutout condition (Opening Radius Ro)
    mask_crown_outer = (x_crown_outer**2 + y_crown_outer**2) < Rto**2
    mask_knuckle_outer = (x_knuckle_outer**2 + y_knuckle_outer**2) < Rto**2
    
    mask_crown_inner = (x_crown_inner**2 + y_crown_inner**2) < Rto**2
    mask_knuckle_inner = (x_knuckle_inner**2 + y_knuckle_inner**2) < Rto**2
    
    x_crown_outer[mask_crown_outer] = np.nan
    y_crown_outer[mask_crown_outer] = np.nan
    z_crown_outer[mask_crown_outer] = np.nan
    
    x_knuckle_outer[mask_knuckle_outer] = np.nan
    y_knuckle_outer[mask_knuckle_outer] = np.nan
    z_knuckle_outer[mask_knuckle_outer] = np.nan

    x_crown_inner[mask_crown_inner] = np.nan
    y_crown_inner[mask_crown_inner] = np.nan
    z_crown_inner[mask_crown_inner] = np.nan

    x_knuckle_inner[mask_knuckle_inner] = np.nan
    y_knuckle_inner[mask_knuckle_inner] = np.nan
    z_knuckle_inner[mask_knuckle_inner] = np.nan

    # Apply flipping if required
    if flipped:
        z_crown_outer = center_z - (z_crown_outer - center_z)
        z_knuckle_outer = center_z - (z_knuckle_outer - center_z)
        z_crown_inner = center_z - (z_crown_inner - center_z)
        z_knuckle_inner = center_z - (z_knuckle_inner - center_z)

    return (x_crown_outer, y_crown_outer, z_crown_outer, 
            x_knuckle_outer, y_knuckle_outer, z_knuckle_outer,
            x_crown_inner, y_crown_inner, z_crown_inner,
            x_knuckle_inner, y_knuckle_inner, z_knuckle_inner)

# Torispherical for Bottom Opening


def generate_torispherical_head_with_bottom_opening(cylinder_radius, crown_radius, knuckle_radius, thickness, center_z, Rbo, num_points=50, flipped=False):
    if crown_radius < cylinder_radius:
        crown_radius = cylinder_radius
    
    # Outer surface calculations
    sin_alpha = cylinder_radius / crown_radius
    alpha = math.asin(sin_alpha)
    h = crown_radius - math.sqrt(crown_radius**2 - cylinder_radius**2)
    
    u_crown = np.linspace(0, alpha, num_points)
    v_crown = np.linspace(0, 2 * np.pi, num_points)
    u_grid_crown, v_grid_crown = np.meshgrid(u_crown, v_crown)
    
    x_crown_outer = crown_radius * np.sin(u_grid_crown) * np.cos(v_grid_crown)
    y_crown_outer = crown_radius * np.sin(u_grid_crown) * np.sin(v_grid_crown)
    z_crown_outer = center_z + crown_radius * np.cos(u_grid_crown) - crown_radius + h
    
    phi = np.linspace(alpha, math.pi/2, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    x_knuckle_outer = knuckle_radius * np.cos(phi_grid) * np.cos(theta_grid)
    y_knuckle_outer = knuckle_radius * np.cos(phi_grid) * np.sin(theta_grid)
    z_knuckle_outer = center_z + knuckle_radius * np.sin(phi_grid) - knuckle_radius + h
    
    # Inner surface calculations
    inner_crown_radius = crown_radius - thickness
    inner_cylinder_radius = cylinder_radius - thickness
    inner_knuckle_radius = knuckle_radius - thickness
    inner_h = inner_crown_radius - math.sqrt(inner_crown_radius**2 - inner_cylinder_radius**2)
    
    x_crown_inner = inner_crown_radius * np.sin(u_grid_crown) * np.cos(v_grid_crown)
    y_crown_inner = inner_crown_radius * np.sin(u_grid_crown) * np.sin(v_grid_crown)
    z_crown_inner = center_z + inner_crown_radius * np.cos(u_grid_crown) - inner_crown_radius + inner_h
    
    x_knuckle_inner = inner_knuckle_radius * np.cos(phi_grid) * np.cos(theta_grid)
    y_knuckle_inner = inner_knuckle_radius * np.cos(phi_grid) * np.sin(theta_grid)
    z_knuckle_inner = center_z + inner_knuckle_radius * np.sin(phi_grid) - inner_knuckle_radius + inner_h

    # Apply cylindrical cutout condition (Opening Radius Ro)
    mask_crown_outer = (x_crown_outer**2 + y_crown_outer**2) < Rbo**2
    mask_knuckle_outer = (x_knuckle_outer**2 + y_knuckle_outer**2) < Rbo**2
    
    mask_crown_inner = (x_crown_inner**2 + y_crown_inner**2) < Rbo**2
    mask_knuckle_inner = (x_knuckle_inner**2 + y_knuckle_inner**2) < Rbo**2
    
    x_crown_outer[mask_crown_outer] = np.nan
    y_crown_outer[mask_crown_outer] = np.nan
    z_crown_outer[mask_crown_outer] = np.nan
    
    x_knuckle_outer[mask_knuckle_outer] = np.nan
    y_knuckle_outer[mask_knuckle_outer] = np.nan
    z_knuckle_outer[mask_knuckle_outer] = np.nan

    x_crown_inner[mask_crown_inner] = np.nan
    y_crown_inner[mask_crown_inner] = np.nan
    z_crown_inner[mask_crown_inner] = np.nan

    x_knuckle_inner[mask_knuckle_inner] = np.nan
    y_knuckle_inner[mask_knuckle_inner] = np.nan
    z_knuckle_inner[mask_knuckle_inner] = np.nan

    # Apply flipping if required
    if flipped:
        z_crown_outer = center_z - (z_crown_outer - center_z)
        z_knuckle_outer = center_z - (z_knuckle_outer - center_z)
        z_crown_inner = center_z - (z_crown_inner - center_z)
        z_knuckle_inner = center_z - (z_knuckle_inner - center_z)

    return (x_crown_outer, y_crown_outer, z_crown_outer, 
            x_knuckle_outer, y_knuckle_outer, z_knuckle_outer,
            x_crown_inner, y_crown_inner, z_crown_inner,
            x_knuckle_inner, y_knuckle_inner, z_knuckle_inner)


def plot_vessel_with_openings(cylinder_radius, cylinder_length, thickness, head_type, **kwargs):
    if cylinder_length <= 0:
        fig = go.Figure()
        fig.update_layout(
            title="Invalid Parameters - Adjust for Valid Visualization",
            scene=dict(aspectmode="data")
        )
        return fig
    
    try:
        fig = go.Figure()
        
        # Generate cylinder with thickness
        x_cyl_outer, y_cyl_outer, z_cyl, x_cyl_inner, y_cyl_inner, _ = generate_cylinder_with_thickness(
            cylinder_radius, cylinder_length, thickness
        )
        
        # Outer cylinder
        fig.add_trace(go.Surface(
            x=x_cyl_outer, y=y_cyl_outer, z=z_cyl, 
            colorscale="Blues", 
            opacity=0.8,
            name="Outer Cylinder",
            showscale=False
        ))
        
        # Inner cylinder
        fig.add_trace(go.Surface(
            x=x_cyl_inner, y=y_cyl_inner, z=z_cyl, 
            colorscale="Reds", 
            opacity=0.8,
            name="Inner Cylinder",
            showscale=False
        ))
        
        if head_type == "Torispherical Head":
            crown_radius = kwargs.get("crown_radius", cylinder_radius * 1.2)
            knuckle_radius = kwargs.get("knuckle_radius", cylinder_radius * 0.06)
            
            # Generate heads with thickness
            (x_crown_top_outer, y_crown_top_outer, z_crown_top_outer,
             x_knuckle_top_outer, y_knuckle_top_outer, z_knuckle_top_outer,
             x_crown_top_inner, y_crown_top_inner, z_crown_top_inner,
             x_knuckle_top_inner, y_knuckle_top_inner, z_knuckle_top_inner) = generate_torispherical_head_with_top_opening(
                cylinder_radius, crown_radius, knuckle_radius, thickness, cylinder_length, Rto, flipped=False
            )
            
            # Add outer surfaces
            fig.add_trace(go.Surface(
                x=x_crown_top_outer, y=y_crown_top_outer, z=z_crown_top_outer, 
                colorscale="Reds", 
                opacity=0.8,
                name="Top Crown (Outer)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_knuckle_top_outer, y=y_knuckle_top_outer, z=z_knuckle_top_outer, 
                colorscale="Reds", 
                opacity=0.8,
                name="Top Knuckle (Outer)",
                showscale=False
            ))
            
            # Add inner surfaces
            fig.add_trace(go.Surface(
                x=x_crown_top_inner, y=y_crown_top_inner, z=z_crown_top_inner, 
                colorscale="Oranges", 
                opacity=0.8,
                name="Top Crown (Inner)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_knuckle_top_inner, y=y_knuckle_top_inner, z=z_knuckle_top_inner, 
                colorscale="Oranges", 
                opacity=0.8,
                name="Top Knuckle (Inner)",
                showscale=False
            ))
            
            # Repeat for bottom head
            (x_crown_bottom_outer, y_crown_bottom_outer, z_crown_bottom_outer,
             x_knuckle_bottom_outer, y_knuckle_bottom_outer, z_knuckle_bottom_outer,
             x_crown_bottom_inner, y_crown_bottom_inner, z_crown_bottom_inner,
             x_knuckle_bottom_inner, y_knuckle_bottom_inner, z_knuckle_bottom_inner) = generate_torispherical_head_with_bottom_opening(
                cylinder_radius, crown_radius, knuckle_radius, thickness, 0, Rbo,flipped=True
            )
            
            fig.add_trace(go.Surface(
                x=x_crown_bottom_outer, y=y_crown_bottom_outer, z=z_crown_bottom_outer, 
                colorscale="Reds", 
                opacity=0.8,
                name="Bottom Crown (Outer)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_knuckle_bottom_outer, y=y_knuckle_bottom_outer, z=z_knuckle_bottom_outer, 
                colorscale="Reds", 
                opacity=0.8,
                name="Bottom Knuckle (Outer)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_crown_bottom_inner, y=y_crown_bottom_inner, z=z_crown_bottom_inner, 
                colorscale="Oranges", 
                opacity=0.8,
                name="Bottom Crown (Inner)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_knuckle_bottom_inner, y=y_knuckle_bottom_inner, z=z_knuckle_bottom_inner, 
                colorscale="Oranges", 
                opacity=0.8,
                name="Bottom Knuckle (Inner)",
                showscale=False
            ))
            
        elif head_type == "Elliptical Head":
            ellipse_ratio = kwargs.get("ellipse_ratio", 0.25)
            dome_height = ellipse_ratio * cylinder_radius
            
            # Top head
            x_top_outer, y_top_outer, z_top_outer, x_top_inner, y_top_inner, z_top_inner = generate_elliptical_head_with_opening_top(
                cylinder_radius, dome_height, thickness, cylinder_length,Rto, flipped=False
            )
            
            fig.add_trace(go.Surface(
                x=x_top_outer, y=y_top_outer, z=z_top_outer,
                colorscale="Greens",
                opacity=0.8,
                name="Top Head (Outer)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_top_inner, y=y_top_inner, z=z_top_inner,
                colorscale="Purples",
                opacity=0.8,
                name="Top Head (Inner)",
                showscale=False
            ))
            
            # Bottom head
            x_bottom_outer, y_bottom_outer, z_bottom_outer, x_bottom_inner, y_bottom_inner, z_bottom_inner = generate_elliptical_head_with_opening_bottom(
                cylinder_radius, dome_height, thickness, 0, Rbo ,flipped=True
            )
            
            fig.add_trace(go.Surface(
                x=x_bottom_outer, y=y_bottom_outer, z=z_bottom_outer,
                colorscale="Greens",
                opacity=0.8,
                name="Bottom Head (Outer)",
                showscale=False
            ))
            
            fig.add_trace(go.Surface(
                x=x_bottom_inner, y=y_bottom_inner, z=z_bottom_inner,
                colorscale="Purples",
                opacity=0.8,
                name="Bottom Head (Inner)",
                showscale=False
            ))
        
        # Corrected layout configuration
        fig.update_layout(
            title=f"3D Pressure Vessel with {head_type} (Thickness: {thickness} mm)",
            scene=dict(
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            height=600  # Correct placement at top level
        )
        
        return fig
    
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title=f"Visualization Error: {str(e)}",
            scene=dict(aspectmode="data")
        )
        return fig
    
def check_dome_resemblance(head_type, cylinder_radius, crown_radius=None, knuckle_radius=None, ellipse_ratio=None):
    if head_type == "Torispherical Head":
        if abs(crown_radius - cylinder_radius) / cylinder_radius < 0.05:
            return "The specified parameters closely resemble a Hemispherical head."
        if 1.6 < crown_radius / cylinder_radius < 1.9:
            return "The specified parameters resemble an Ellipsoidal head (approximately 2:1 ratio)."
        if knuckle_radius / crown_radius < 0.1:
            return "The specified parameters resemble a spherical head with a very small knuckle radius."
        return "The specified parameters represent a standard Torispherical head."
    elif head_type == "Elliptical Head":
        if ellipse_ratio == 0.25:
            return "Standard elliptical heads"
        elif 0.1667 <= ellipse_ratio < 0.2:
            return "Flattened Elliptical - Low-pressure storage tanks"
        elif ellipse_ratio == 1.0:
            return "Hemispherical"
        elif ellipse_ratio <= 0.16:
            return "Too Small Elliptical Ratio! May Converge to Flattened Head"
        elif ellipse_ratio > 1.0:
            return "Prolate Ellipsoidal Condition"
        else:
            return "Standard elliptical head configuration"


# # Function for Downloading STL File
# def generate_stl_data(cylinder_radius, cylinder_length, thickness, head_type, **kwargs):
#     """
#     Creates STL mesh data for the pressure vessel.
#     """
#     vertices = []
#     faces = []

#     # Simulating a basic cylinder (for simplicity)
#     for i in range(10):
#         for j in range(10):
#             x = cylinder_radius * np.cos(2 * np.pi * i / 10)
#             y = cylinder_radius * np.sin(2 * np.pi * i / 10)
#             z = cylinder_length * (j / 10)
#             vertices.append([x, y, z])

#     num_cols = 10
#     for i in range(9):
#         for j in range(9):
#             v1 = i * num_cols + j
#             v2 = v1 + 1
#             v3 = v1 + num_cols
#             v4 = v3 + 1
#             faces.append([v1, v2, v3])
#             faces.append([v2, v4, v3])

#     # Convert to NumPy arrays
#     vertices = np.array(vertices)
#     faces = np.array(faces)

#     # Create an STL mesh
#     vessel_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
#     for i, f in enumerate(faces):
#         for j in range(3):
#             vessel_mesh.vectors[i][j] = vertices[f[j]]

#     # Save to a buffer
#     stl_buffer = io.BytesIO()
#     vessel_mesh.save(stl_buffer)
#     stl_buffer.seek(0)

#     return stl_buffer


# Streamlit UI Configuration
st.set_page_config(
    page_title="Pressure Vessel Visualizer",
    page_icon=":material/temp_preferences_eco:",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def calculate_vessel_dimensions(target_volume, cylinder_radius, crown_radius=None, knuckle_radius=None, ellipse_ratio=None):
    try:
        if head_type == "Torispherical Head":
            if crown_radius < cylinder_radius:
                return None, None, None, "Error: Crown radius must be ≥ cylinder radius"
            if knuckle_radius > cylinder_radius:
                return None, None, None, "Error: Knuckle radius must be ≤ cylinder radius"
            
            head_height, head_volume, cylinder_length = torispherical_calculator(
                target_volume, cylinder_radius, crown_radius, knuckle_radius
            )
            return head_height, head_volume, cylinder_length, None
        
        elif head_type == "Elliptical Head":
            if ellipse_ratio is None:
                return None, None, None, "Error: Ellipse ratio is required"
            head_height, head_volume, cylinder_length = elliptical_calculator(
                target_volume, cylinder_radius, ellipse_ratio
            )
            return head_height, head_volume, cylinder_length, None
        
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"




# Sidebar Inputs
with st.sidebar:
    st.header("Vessel Parameter Input")
    target_volume = st.number_input("Target Volume (Liters)", min_value=1.0, max_value=500.0, value=50.0, step=10.0)
    
    st.header("Hydrogen Calculator")
    pressure = st.number_input("Pressure (bar)", min_value=1.0, max_value=700.0, value=10.0, step=1.0)
    temperature = st.slider("Temperature (°C)", min_value=-50.0, max_value=100.0, value=25.0, step=1.0)
    hydrogen_mass = hydrogen_amount_calculator(pressure, target_volume, temperature)
    st.write(f"### Hydrogen Amount: {hydrogen_mass:.4f} kg")

    cylinder_radius = st.number_input("Cylinder Radius (mm)", min_value=1.0, value=100.0, step=10.0)
    thickness = st.number_input("Wall Thickness (mm)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    
    head_type = st.selectbox("Select Dome Type", ["Torispherical Head", "Elliptical Head"])
    if head_type == "Torispherical Head":
        crown_radius = st.number_input("Crown Radius (mm)", min_value=cylinder_radius, value=cylinder_radius * 1.2, step=1.0)
        knuckle_radius = st.number_input("Knuckle Radius (mm)", min_value=0.01 * cylinder_radius, value=0.06 * cylinder_radius, step=1.0)
    else:
        ellipse_ratio = st.number_input("Ellipse Ratio (H/R)", min_value=0.1, max_value=10.0, value=0.25, step=0.01)

    st.header("Dome Openings")
    Rto=st.number_input("Top Opening(mm)",min_value=0.0 , max_value=cylinder_radius , value=0.0,step=10.0)
    Rbo=st.number_input("Bottom Opening(mm)",min_value=0.0 , max_value=cylinder_radius , value=0.0,step=10.0)



# Main Content
col1, col2 = st.columns([4, 2])

if head_type == "Torispherical Head":
    head_height, head_volume, cylinder_length, error_msg = calculate_vessel_dimensions(
        target_volume, cylinder_radius, crown_radius, knuckle_radius, None
    )
else:
    head_height, head_volume, cylinder_length, error_msg = calculate_vessel_dimensions(
        target_volume, cylinder_radius, None, None, ellipse_ratio
    )

if error_msg or cylinder_length <= 0:
    with col1:
        st.error(error_msg or "Invalid parameters! Cylinder length is non-positive.")
        st.warning("Try increasing the target volume or adjusting head parameters.")
        fig = go.Figure()
        fig.update_layout(title="Invalid Parameters", scene=dict(aspectmode="data"))
        st.plotly_chart(fig, use_container_width=True)
else:
    kwargs = {}
    if head_type == "Torispherical Head":
        kwargs.update({"crown_radius": crown_radius, "knuckle_radius": knuckle_radius})
    else:
        kwargs.update({"ellipse_ratio": ellipse_ratio})
    
    fig = plot_vessel_with_openings(cylinder_radius, cylinder_length, thickness, head_type, **kwargs)
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Vessel Dimensions")
        st.write(f"**Cylinder Length:** {cylinder_length:.2f} mm")
        st.write(f"**Head Height:** {head_height:.2f} mm")
        st.write(f"**Volume per Head:** {head_volume/1e6:.4f} Liters")
        st.write(f"**Total Vessel Length:** {cylinder_length + 2*head_height:.2f} mm")
        st.write(f"**Total Target Volume:** {target_volume:.2f} Liters")
        st.write(f"**Wall Thickness:** {thickness:.1f} mm")

st.header("Head Type Analysis")
resemblance = check_dome_resemblance(
    head_type, cylinder_radius, 
    crown_radius if head_type == "Torispherical Head" else None,
    knuckle_radius if head_type == "Torispherical Head" else None,
    ellipse_ratio if head_type == "Elliptical Head" else None
)
st.info(resemblance)

st.header("Parameter Ratios")
if head_type == "Torispherical Head":
    st.write(f"**Crown Radius / Cylinder Radius:** {crown_radius/cylinder_radius:.2f}")
    st.write(f"**Knuckle Radius / Crown Radius:** {knuckle_radius/crown_radius:.2f}")
    st.write(f"**Knuckle Radius / Cylinder Radius:** {knuckle_radius/cylinder_radius:.2f}")
else:
    st.write(f"**Ellipse Ratio (h/R):** {ellipse_ratio:.2f}")

# st.header("Download File")
# if st.button("Generate 3D Model"):
#     stl_file = plot_vessel_with_openings(cylinder_radius, cylinder_length, thickness, head_type, **kwargs)
    
#     # Add a Download Button
#     st.download_button(
#         label="Download STL File",
#         data=stl_file,
#         file_name="pressure_vessel.stl",
#         mime="application/octet-stream"
#     )

# Thickness Calcualtor
#Fibre Section


def calculate_layers(P, r, t_f, sigma_f):
    P = P * 1e5
    t_req = (P * r) / sigma_f  # Required thickness
    N = t_req / t_f  # Number of layers
    return max(1, round(N)),t_req*1e3  # Ensure at least 1 layer

st.title("Carbon Fiber Wrapping Calculator")
fibre_thickness = st.number_input("Fiber Layer Thickness (mm)", min_value=0.01, max_value=5.0, value=0.2) / 1000  # Convert mm to m
sigma_f = st.number_input("Tensile Strength of Carbon Fiber (MPa)", min_value=100, max_value=5000, value=2000) * 1e6  # Convert MPa to Pa
Num_layers,thick_comp = calculate_layers(pressure,cylinder_radius,fibre_thickness,sigma_f)
st.write(f"Total thickness of Composite Layer(mm):{thick_comp:.2f}")
st.write(f"Number of Layers : {Num_layers:.2f}")


# Computing Thickness

def compute_thickness(angle, base_thickness):
    """Calculate thickness for a given angle."""
    if angle == 90:  # Hoop Winding
        return base_thickness
    elif np.cos(np.radians(angle)) == 0:  # Prevent division by zero
        return float('inf')  
    return base_thickness / np.cos(np.radians(angle))


st.title("Filament Winding Thickness Calculator")

# Base Thickness Input
base_thickness = st.number_input("Enter Base Thickness (t0) in mm:", min_value=0.01, value=1.0, step=0.01)
# Dynamic Angle Inputs
st.subheader("Enter Winding Angles:")
angles_mat = []
num_layers = st.number_input("Number of Layers:", min_value=1, value=1, step=1)
# for i in range(num_layers):
#     winding_type = st.selectbox("Select Winding Type:", ["Helical", "Hoop"])
#     angle = st.number_input(f"Enter Winding Angle for Layer {i+1} (°):", min_value=1, max_value=89, value=45, step=1)
#     if winding_type=="Hoop":
#         angle=90
#         angles_mat.append(angle)
#     else:
#         angles_mat.append(angle)

# Get Winding Angles & Types Separately for Each Layer
for i in range(num_layers):
    col1, col2 = st.columns([1, 1])  # Create two columns for layout
    with col1:
        winding_type = st.selectbox(f"Layer {i+1} - Winding Type:", ["Helical", "Hoop"], key=f"type_{i}")
    with col2:
        if winding_type == "Hoop":
            angle = 90  # Automatically set to 90° for Hoop
        else:
            angle = st.number_input(f"Layer {i+1} - Angle (°):", min_value=1, max_value=89, value=45, step=1, key=f"angle_{i}")
    angles_mat.append(angle)

# Compute Thickness for Each Layer
thicknesses = [compute_thickness(angle, base_thickness) for angle in angles_mat]
total_thickness = sum(thicknesses)
# Display Results
st.subheader("Results:")
for i, (angle, thickness) in enumerate(zip(angles_mat, thicknesses)):
    st.write(f"Layer {i+1}: Angle = {angle}°, Thickness = {thickness:.3f} mm")

st.write(f"**Total Thickness:** {total_thickness:.3f} mm")


