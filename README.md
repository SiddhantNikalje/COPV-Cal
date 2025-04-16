# 🧪 Streamlit App for Pressure Vessel Design & Visualization

This is a comprehensive **Streamlit-based interactive application** for designing and visualizing pressure vessels with either **torispherical** or **elliptical heads**. It combines real-time 3D modeling, precise engineering calculations, and composite wrapping estimations — all in a single intuitive interface.

---

## 🚀 Features

- **Interactive Sidebar Inputs**Customize the pressure vessel by setting:

  - Hydrogen volume, storage pressure, and temperature
  - Cylinder radius, wall thickness
  - Head type: `Torispherical` or `Elliptical`
  - Dome opening radii (Rto and Rbo)
  - Specific geometry controls:
    - Torispherical: Crown & knuckle radii
    - Elliptical: Ellipse ratio
- **Gas Mass Calculation**Uses the **Ideal Gas Law** to calculate stored hydrogen mass in kilograms.
- **Geometry & Dimension Computation**

  - Calculates total volume, cylinder length, head height
  - Validates geometrical constraints (e.g. no negative lengths)
  - Performs **head resemblance analysis** to check against common standards (hemispherical, ellipsoidal, etc.)
- **Real-time 3D Visualization**Dynamic Plotly-based rendering of:

  - Cylinder outer/inner walls
  - Torispherical or Elliptical heads (top & bottom)
  - Internal voids and dome cutouts
- **Carbon Fiber Wrapping Calculator**
  Estimates:

  - Number of composite layers
  - Composite thickness required to sustain design pressure
- **Filament Winding Module**
  Calculates:

  - Thickness contribution per layer based on winding angle
  - Summed wall thickness for `helical` and `hoop` patterns

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SiddhantNikalje/COPV-Cal.git
   cd COPV-Cal

   ```
