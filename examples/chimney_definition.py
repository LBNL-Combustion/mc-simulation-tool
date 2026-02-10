"""
Example showing how to define a complete chimney using the new material properties system
"""
import numpy as np
from flued_module.material_properties import MaterialProperties, PipeConfiguration
from flued_module.chimney import Chimney, ChimneyPart

def example_1_simple_chimney():
    """Simple single-wall chimney with uniform material"""
    print("=" * 60)
    print("Example 1: Simple Single-Wall Chimney")
    print("=" * 60)

    # Load material properties
    mat_props = MaterialProperties("flued_module/pipe_props.json")

    # Create a single wall pipe section
    pipe = mat_props.create_single_wall_pipe(
        inner_diameter_m=0.152,  # 6 inch diameter
        height_m=6.0,             # 6 meters tall
        material_name="steel_24_gauge",
        orientation="vertical",
        T_ambient_K=293.15,       # 20°C ambient
        num_segments=20,
        roughness_type="galvanized_steel"
    )

    print(f"Inner diameter: {pipe.inner_diameter_m:.3f} m")
    print(f"Outer diameter: {pipe.outer_diameter_m:.3f} m")
    print(f"Wall material: {pipe.layers[0].material.name}")
    print(f"Wall thickness: {pipe.layers[0].thickness_m*1000:.2f} mm")
    print(f"Thermal resistance: {pipe.get_total_thermal_resistance():.4f} m²·K/W")

    # Convert to ChimneyPart format for use with existing code
    chimney_part_dict = pipe.to_chimney_part_dict()
    chimney_part = ChimneyPart(**chimney_part_dict)

    return chimney_part

def example_2_triple_wall_chimney():
    """Triple-wall insulated chimney"""
    print("\n" + "=" * 60)
    print("Example 2: Triple-Wall Insulated Chimney")
    print("=" * 60)

    # Load material properties
    mat_props = MaterialProperties("flued_module/pipe_props.json")

    # Create a triple wall pipe with ceramic fiber insulation
    pipe = mat_props.create_triple_wall_pipe(
        inner_diameter_m=0.203,           # 8 inch diameter
        height_m=10.0,                    # 10 meters tall
        inner_material="stainless_316",   # Inner liner
        insulation_material="ceramic_fiber",  # Insulation
        outer_material="stainless_430",   # Outer shell
        inner_gap_m=0.003,                # 3mm air gap
        outer_gap_m=0.003,                # 3mm air gap
        orientation="vertical",
        T_ambient_K=288.15,               # 15°C ambient
        num_segments=30
    )

    print(f"Inner diameter: {pipe.inner_diameter_m:.3f} m")
    print(f"Outer diameter: {pipe.outer_diameter_m:.3f} m")
    print(f"\nLayers:")
    for i, layer in enumerate(pipe.layers):
        if layer.is_air_gap:
            print(f"  Layer {i+1}: Air gap, {layer.thickness_m*1000:.1f} mm")
        else:
            print(f"  Layer {i+1}: {layer.material.name}, {layer.thickness_m*1000:.1f} mm")
    print(f"\nTotal thermal resistance: {pipe.get_total_thermal_resistance():.4f} m²·K/W")

    # Convert to ChimneyPart format
    chimney_part_dict = pipe.to_chimney_part_dict()
    chimney_part = ChimneyPart(**chimney_part_dict)

    return chimney_part

def example_3_complex_chimney_system():
    """Complete chimney system with multiple sections"""
    print("\n" + "=" * 60)
    print("Example 3: Complex Multi-Section Chimney System")
    print("=" * 60)

    mat_props = MaterialProperties("flued_module/pipe_props.json")

    # Section 1: Single wall connector pipe (horizontal)
    connector = mat_props.create_single_wall_pipe(
        inner_diameter_m=0.152,
        height_m=2.0,  # 2m horizontal run
        material_name="steel_24_gauge",
        orientation="horizontal",
        z_m=1.5,  # 1.5m above floor
        T_ambient_K=298.15,  # Indoor temperature 25°C
        num_segments=10
    )

    # Section 2: Triple wall vertical through roof
    vertical_section = mat_props.create_triple_wall_pipe(
        inner_diameter_m=0.152,
        height_m=4.0,  # 4m vertical rise
        inner_material="stainless_316",
        insulation_material="mineral_wool",  # Different insulation
        outer_material="aluminum",  # Aluminum outer for weather resistance
        z_m=1.5,  # Starts at same elevation as connector
        T_ambient_K=293.15,  # Attic temperature 20°C
        num_segments=20
    )

    # Section 3: Triple wall chimney above roof
    roof_section = mat_props.create_triple_wall_pipe(
        inner_diameter_m=0.152,
        height_m=2.0,  # 2m above roof
        inner_material="stainless_316",
        insulation_material="ceramic_fiber",  # High-temp insulation
        outer_material="stainless_430",
        z_m=5.5,  # 5.5m elevation (1.5 + 4.0)
        T_ambient_K=283.15,  # Outside temperature 10°C
        num_segments=10
    )

    # Convert all to ChimneyPart objects
    parts = []
    for i, (name, config) in enumerate([
        ("Connector", connector),
        ("Vertical Section", vertical_section),
        ("Roof Section", roof_section)
    ]):
        print(f"\n{name}:")
        print(f"  Type: {config.wall_type}")
        print(f"  Diameter: {config.inner_diameter_m:.3f} m")
        print(f"  Length: {config.height_m:.1f} m")
        print(f"  Elevation: {config.z_m:.1f} m")
        print(f"  R-value: {config.get_total_thermal_resistance():.4f} m²·K/W")

        part_dict = config.to_chimney_part_dict()
        parts.append(ChimneyPart(**part_dict))

    # Create complete chimney system
    chimney = Chimney(
        parts=parts,
        T_flue_init_K=473.15,  # 200°C initial flue gas
        mdot_init_kg_s=0.05,    # 0.05 kg/s mass flow
        CO2_frac=0.10,          # 10% CO2
        H2O_frac=0.15,          # 15% H2O
        verbose=False
    )

    print(f"\nTotal chimney height: {sum(p.height_m for p in parts):.1f} m")
    print(f"Number of sections: {len(parts)}")

    return chimney

def example_4_custom_material_chimney():
    """Example with custom material properties"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Material Configuration")
    print("=" * 60)

    mat_props = MaterialProperties("flued_module/pipe_props.json")

    # Define a custom wall configuration with specific thicknesses
    wall_config = {
        "type": "triple",
        "layers": [
            {"material": "stainless_316", "thickness_m": 0.001},    # 1mm inner
            {"air_gap": True, "thickness_m": 0.005},                # 5mm gap
            {"material": "vermiculite", "thickness_m": 0.030},      # 30mm vermiculite
            {"air_gap": True, "thickness_m": 0.005},                # 5mm gap
            {"material": "aluminum", "thickness_m": 0.0015}         # 1.5mm aluminum
        ]
    }

    pipe = mat_props.create_pipe_configuration(
        inner_diameter_m=0.254,  # 10 inch
        height_m=8.0,
        wall_config=wall_config,
        orientation="vertical",
        T_ambient_K=290.15,
        roughness_m=2e-6,  # Custom roughness
        condensate_film_max_m=1e-4,  # Thicker film allowed
        num_segments=25
    )

    print(f"Configuration: Custom triple wall")
    print(f"Inner diameter: {pipe.inner_diameter_m:.3f} m")
    print(f"Outer diameter: {pipe.outer_diameter_m:.3f} m")
    print(f"Total wall thickness: {(pipe.outer_diameter_m - pipe.inner_diameter_m)/2*1000:.1f} mm")
    print(f"\nLayer breakdown:")

    cumulative_R = 0
    for i, layer in enumerate(pipe.layers):
        R_layer = layer.get_thermal_resistance_per_area()
        cumulative_R += R_layer
        if layer.is_air_gap:
            print(f"  {i+1}. Air gap: {layer.thickness_m*1000:.1f} mm, R={R_layer:.4f}")
        else:
            print(f"  {i+1}. {layer.material.name}: {layer.thickness_m*1000:.1f} mm, R={R_layer:.4f}")

    print(f"\nTotal R-value: {cumulative_R:.4f} m²·K/W")

    # Show thermal properties of materials used
    print(f"\nMaterial thermal properties:")
    for layer in pipe.layers:
        if not layer.is_air_gap and layer.material:
            mat = layer.material
            print(f"  {mat.name}:")
            print(f"    k = {mat.thermal_conductivity_W_m_K:.3f} W/m·K")
            print(f"    ρ = {mat.density_kg_m3:.0f} kg/m³")
            print(f"    cp = {mat.specific_heat_J_kg_K:.0f} J/kg·K")
            print(f"    ε = {mat.emissivity:.2f}")

    return pipe

def example_5_parametric_study():
    """Compare different insulation materials"""
    print("\n" + "=" * 60)
    print("Example 5: Insulation Material Comparison")
    print("=" * 60)

    mat_props = MaterialProperties("flued_module/pipe_props.json")

    insulation_materials = ["ceramic_fiber", "mineral_wool", "vermiculite"]

    results = []
    for insulation in insulation_materials:
        pipe = mat_props.create_triple_wall_pipe(
            inner_diameter_m=0.152,
            height_m=5.0,
            inner_material="stainless_316",
            insulation_material=insulation,
            outer_material="stainless_430",
            inner_gap_m=0.003,
            outer_gap_m=0.003
        )

        R_total = pipe.get_total_thermal_resistance()

        # Get insulation properties
        insulation_layer = pipe.layers[2]  # Middle layer is insulation

        results.append({
            "material": insulation,
            "R_total": R_total,
            "k": insulation_layer.material.thermal_conductivity_W_m_K,
            "thickness": insulation_layer.thickness_m,
            "density": insulation_layer.material.density_kg_m3
        })

    # Display comparison
    print(f"\n{'Material':<15} {'R-value':<12} {'k (W/m·K)':<12} {'Thickness (mm)':<15} {'Density (kg/m³)'}")
    print("-" * 70)
    for r in results:
        print(f"{r['material']:<15} {r['R_total']:<12.4f} {r['k']:<12.3f} {r['thickness']*1000:<15.1f} {r['density']:<.0f}")

    # Find best insulation
    best = max(results, key=lambda x: x['R_total'])
    print(f"\nBest insulation for thermal resistance: {best['material']} (R = {best['R_total']:.4f} m²·K/W)")

if __name__ == "__main__":
    # Run all examples
    example_1_simple_chimney()
    example_2_triple_wall_chimney()
    chimney_system = example_3_complex_chimney_system()
    example_4_custom_material_chimney()
    example_5_parametric_study()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)