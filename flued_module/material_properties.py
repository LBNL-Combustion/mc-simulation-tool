"""
Material properties loader and pipe configuration helper
"""
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

@dataclass
class MaterialData:
    """Container for material thermal properties"""
    name: str
    density_kg_m3: float
    thermal_conductivity_W_m_K: float
    specific_heat_J_kg_K: float
    emissivity: float
    roughness_m: Optional[float]  # Surface roughness for friction calculations
    common_thickness_m: float
    description: str = ""

    @property
    def thermal_diffusivity(self) -> float:
        """Calculate thermal diffusivity α = k/(ρ*cp)"""
        return self.thermal_conductivity_W_m_K / (self.density_kg_m3 * self.specific_heat_J_kg_K)

@dataclass
class WallLayer:
    """Single layer in a wall configuration"""
    material: Optional[MaterialData] = None
    thickness_m: float = 0.0
    is_air_gap: bool = False

    def get_thermal_resistance_per_area(self) -> float:
        """Calculate R-value per unit area (m²·K/W)"""
        if self.is_air_gap:
            # Air gap resistance (simplified - could be enhanced with convection correlations)
            k_air = 0.026  # W/m·K at room temperature
            return self.thickness_m / k_air
        elif self.material:
            return self.thickness_m / self.material.thermal_conductivity_W_m_K
        else:
            return 0.0

@dataclass
class PipeConfiguration:
    """Complete pipe configuration with materials and geometry"""
    # Geometry
    inner_diameter_m: float
    height_m: float
    orientation: str = "vertical"
    z_m: float = 0.0  # Elevation

    # Wall configuration
    wall_type: str = "single"  # "single" or "triple"
    layers: List[WallLayer] = field(default_factory=list)

    # Operating conditions
    T_ambient_K: float = 293.15
    Pa: float = 98457  # Atmospheric pressure at lab elevation
    humidity_ratio: float = 0.01

    # Surface properties
    roughness_m: float = 1e-6
    condensate_film_max_m: float = 5e-5

    # Computational
    num_segments: Optional[int] = None

    @property
    def outer_diameter_m(self) -> float:
        """Calculate outer diameter from layers"""
        total_thickness = sum(layer.thickness_m for layer in self.layers)
        return self.inner_diameter_m + 2 * total_thickness

    @property
    def inner_area_m2(self) -> float:
        """Flow cross-sectional area"""
        return np.pi * (self.inner_diameter_m / 2) ** 2

    @property
    def inner_perimeter_m(self) -> float:
        """Wetted perimeter"""
        return np.pi * self.inner_diameter_m

    def get_total_thermal_resistance(self) -> float:
        """Calculate total thermal resistance per unit area"""
        return sum(layer.get_thermal_resistance_per_area() for layer in self.layers)

    def to_chimney_part_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing ChimneyPart"""
        config = {
            "type": self.wall_type,
            "inner_diameter_ft": self.inner_diameter_m / 0.3048,
            "height_ft": self.height_m / 0.3048,
            "z_ft": self.z_m / 0.3048,
            "orientation": self.orientation,
            "T_ambient_K": self.T_ambient_K,
            "Pa": self.Pa,
            "num_segments": self.num_segments,
            "roughness_m": self.roughness_m,
            "humidity_ratio": self.humidity_ratio,
            "film_w": self.condensate_film_max_m
        }

        if self.wall_type == "single" and self.layers:
            # Find the main wall layer
            wall_layer = next((l for l in self.layers if not l.is_air_gap), None)
            if wall_layer and wall_layer.material:
                config.update({
                    "thickness_w_m": wall_layer.thickness_m,
                    "density_w_kg_m3": wall_layer.material.density_kg_m3,
                    "cp_w_J_kg_K": wall_layer.material.specific_heat_J_kg_K,
                    "k_w_W_m_K": wall_layer.material.thermal_conductivity_W_m_K,
                    "epsilon_w": wall_layer.material.emissivity
                })

        elif self.wall_type == "triple":
            # Map layers to triple wall configuration
            layer_mapping = {
                "inner": None,
                "inner_gap": 0.0,
                "middle": None,
                "outer_gap": 0.0,
                "outer": None
            }

            # Parse layers
            layer_idx = 0
            for layer in self.layers:
                if layer_idx == 0 and not layer.is_air_gap:  # Inner liner
                    layer_mapping["inner"] = layer
                elif layer_idx == 1 and layer.is_air_gap:  # Inner gap
                    layer_mapping["inner_gap"] = layer.thickness_m
                elif layer_idx == 2 and not layer.is_air_gap:  # Insulation
                    layer_mapping["middle"] = layer
                elif layer_idx == 3 and layer.is_air_gap:  # Outer gap
                    layer_mapping["outer_gap"] = layer.thickness_m
                elif layer_idx == 4 and not layer.is_air_gap:  # Outer shell
                    layer_mapping["outer"] = layer
                layer_idx += 1

            # Set inner wall properties
            if layer_mapping["inner"] and layer_mapping["inner"].material:
                inner = layer_mapping["inner"]
                config.update({
                    "thickness_inner_m": inner.thickness_m,
                    "density_inner_kg_m3": inner.material.density_kg_m3,
                    "cp_inner_J_kg_K": inner.material.specific_heat_J_kg_K,
                    "k_inner_W_m_K": inner.material.thermal_conductivity_W_m_K,
                    "epsilon_inner": inner.material.emissivity
                })

            # Set middle layer properties
            if layer_mapping["middle"] and layer_mapping["middle"].material:
                middle = layer_mapping["middle"]
                config.update({
                    "thickness_middle_m": middle.thickness_m,
                    "density_middle_kg_m3": middle.material.density_kg_m3,
                    "cp_middle_J_kg_K": middle.material.specific_heat_J_kg_K,
                    "k_middle_W_m_K": middle.material.thermal_conductivity_W_m_K
                })

            # Set outer wall properties
            if layer_mapping["outer"] and layer_mapping["outer"].material:
                outer = layer_mapping["outer"]
                config.update({
                    "thickness_outer_m": outer.thickness_m,
                    "density_outer_kg_m3": outer.material.density_kg_m3,
                    "cp_outer_J_kg_K": outer.material.specific_heat_J_kg_K,
                    "k_outer_W_m_K": outer.material.thermal_conductivity_W_m_K,
                    "epsilon_outer": outer.material.emissivity
                })

            # Set air gaps
            config.update({
                "air_gap_inner_m": layer_mapping["inner_gap"],
                "air_gap_outer_m": layer_mapping["outer_gap"]
            })

        return config

class MaterialProperties:
    """Load and manage material properties from JSON file"""

    def __init__(self, json_path: Optional[str] = None):
        """Initialize with path to properties JSON file.

        If no path provided, uses the default pipe_props.json from dev/exp_data/.
        """
        self.materials: Dict[str, MaterialData] = {}
        self.configurations: Dict[str, Any] = {}
        self.surface_properties: Dict[str, Any] = {}
        self.condensation_properties: Dict[str, Any] = {}
        self.vent_options: Dict[str, Any] = {}

        # Find the JSON file
        if json_path is not None:
            self.json_path = Path(json_path)
        else:
            # Default: look in dev/exp_data (one level above the flued_module package)
            package_dir = Path(__file__).parent
            self.json_path = package_dir.parent / "dev" / "exp_data" / "pipe_props.json"

        if self.json_path.exists():
            self.load_properties()
        else:
            # Try alternate paths for backward compatibility
            alt_paths = [
                Path("pipe_props.json"),
                Path("dev") / "exp_data" / "pipe_props.json",
                Path("flued_module") / "data" / "pipe_props.json",
                Path("flued_module") / "pipe_props.json",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    self.json_path = alt_path
                    self.load_properties()
                    break
            else:
                print(f"WARNING: Could not find pipe_props.json. Tried: {self.json_path}, {alt_paths}")

    def load_properties(self):
        """Load material properties from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Load materials
        if "materials" in data:
            for name, props in data["materials"].items():
                self.materials[name] = MaterialData(
                    name=name,
                    density_kg_m3=props["density_kg_m3"],
                    thermal_conductivity_W_m_K=props["thermal_conductivity_W_m_K"],
                    specific_heat_J_kg_K=props["specific_heat_J_kg_K"],
                    emissivity=props["emissivity"],
                    roughness_m=props.get("roughness_m"),  # May be None for insulation
                    common_thickness_m=props["common_thickness_m"],
                    description=props.get("description", "")
                )

        # Store configurations and surface properties
        self.configurations = data.get("pipe_configurations", {})
        self.surface_properties = data.get("surface_properties", {})
        self.condensation_properties = data.get("condensation_properties", {})
        self.vent_options = data.get("vent_options", {})

    def get_material(self, name: str) -> Optional[MaterialData]:
        """Get material by name"""
        return self.materials.get(name)

    def create_pipe_configuration(self,
                                 inner_diameter_m: float,
                                 height_m: float,
                                 wall_config: Dict[str, Any],
                                 **kwargs) -> PipeConfiguration:
        """
        Create a pipe configuration from specifications

        Args:
            inner_diameter_m: Inner diameter in meters
            height_m: Height/length in meters
            wall_config: Dictionary with 'type' and 'layers'
            **kwargs: Additional parameters (orientation, T_ambient_K, etc.)

        Returns:
            PipeConfiguration object
        """
        wall_type = wall_config.get("type", "single")
        layers = []

        # Build layers list
        for layer_spec in wall_config.get("layers", []):
            if "material" in layer_spec:
                # Material layer
                material = self.get_material(layer_spec["material"])
                if material:
                    thickness = layer_spec.get("thickness_m", material.common_thickness_m)
                    layers.append(WallLayer(
                        material=material,
                        thickness_m=thickness,
                        is_air_gap=False
                    ))
            elif "air_gap" in layer_spec or "gap" in layer_spec["type"] if "type" in layer_spec else False:
                # Air gap layer
                layers.append(WallLayer(
                    thickness_m=layer_spec.get("thickness_m", 0.003),
                    is_air_gap=True
                ))

        # Get surface roughness if specified
        roughness = kwargs.get("roughness_m", 1e-6)
        if "roughness_type" in kwargs:
            roughness = self.surface_properties.get("roughness_m", {}).get(
                kwargs["roughness_type"], roughness
            )

        return PipeConfiguration(
            inner_diameter_m=inner_diameter_m,
            height_m=height_m,
            wall_type=wall_type,
            layers=layers,
            orientation=kwargs.get("orientation", "vertical"),
            z_m=kwargs.get("z_m", 0.0),
            T_ambient_K=kwargs.get("T_ambient_K", 293.15),
            Pa=kwargs.get("Pa", 98457),
            humidity_ratio=kwargs.get("humidity_ratio", 0.01),
            roughness_m=roughness,
            condensate_film_max_m=kwargs.get("condensate_film_max_m", 5e-5),
            num_segments=kwargs.get("num_segments")
        )

    def create_single_wall_pipe(self,
                               inner_diameter_m: float,
                               height_m: float,
                               material_name: str = "steel_24_gauge",
                               thickness_m: Optional[float] = None,
                               **kwargs) -> PipeConfiguration:
        """Convenience method for single wall pipes"""
        material = self.get_material(material_name)
        if not material:
            raise ValueError(f"Material '{material_name}' not found")

        if thickness_m is None:
            thickness_m = material.common_thickness_m

        wall_config = {
            "type": "single",
            "layers": [
                {"material": material_name, "thickness_m": thickness_m}
            ]
        }

        return self.create_pipe_configuration(
            inner_diameter_m, height_m, wall_config, **kwargs
        )

    def create_triple_wall_pipe(self,
                               inner_diameter_m: float,
                               height_m: float,
                               inner_material: str = "stainless_316",
                               insulation_material: str = "ceramic_fiber",
                               outer_material: str = "stainless_430",
                               inner_gap_m: float = 0.003,
                               outer_gap_m: float = 0.003,
                               **kwargs) -> PipeConfiguration:
        """Convenience method for triple wall pipes"""
        wall_config = {
            "type": "triple",
            "layers": [
                {"material": inner_material},
                {"air_gap": True, "thickness_m": inner_gap_m},
                {"material": insulation_material},
                {"air_gap": True, "thickness_m": outer_gap_m},
                {"material": outer_material}
            ]
        }

        return self.create_pipe_configuration(
            inner_diameter_m, height_m, wall_config, **kwargs
        )

    def create_chimney_part_dict(self,
                                 pipe_type: str,
                                 inner_diameter_ft: float,
                                 height_ft: float,
                                 z_ft: float = 0,
                                 orientation: str = "vertical",
                                 T_ambient_K: float = 293.15,
                                 Pa: float = 98457,
                                 num_segments: Optional[int] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Create a ChimneyPart dictionary directly from material names.

        This is the simplest interface - just specify materials and geometry.

        Args:
            pipe_type: 'single' or 'triple'
            inner_diameter_ft: Inner diameter in feet
            height_ft: Height/length in feet
            z_ft: Elevation in feet
            orientation: 'vertical' or 'horizontal'
            T_ambient_K: Ambient temperature in Kelvin
            Pa: Atmospheric pressure in Pascals
            num_segments: Number of computational segments (default: about 0.3 m each)

        For single wall, use kwargs:
            material: Material name from pipe_props.json
            thickness_m: Wall thickness (optional, uses default from material)

        For triple wall, use kwargs:
            inner_material: Inner wall material name
            middle_material: Insulation material name
            outer_material: Outer wall material name
            inner_thickness_m: Inner wall thickness (optional)
            middle_thickness_m: Insulation thickness (optional)
            outer_thickness_m: Outer wall thickness (optional)
            air_gap_inner_m: Inner air gap (default 0)
            air_gap_outer_m: Outer air gap (default 0)

        Returns:
            Dictionary ready to pass to ChimneyPart(**dict)
        """

        # Base configuration common to all types
        config = {
            'type': pipe_type,
            'inner_diameter_ft': inner_diameter_ft,
            'height_ft': height_ft,
            'z_ft': z_ft,
            'orientation': orientation,
            'T_ambient_K': T_ambient_K,
            'Pa': Pa,
            'num_segments': num_segments,
            'humidity_ratio': kwargs.get('humidity_ratio', 0.01),
        }

        if pipe_type == 'single':
            # Get material
            material_name = kwargs.get('material', 'steel_24_gauge')
            material = self.get_material(material_name)
            if not material:
                raise ValueError(f"Material '{material_name}' not found in pipe_props.json")

            # Use provided thickness or default from material
            thickness = kwargs.get('thickness_m', material.common_thickness_m)

            # Get condensation film thickness for this material
            condensation_props = self.condensation_properties.get(material_name, {})
            default_film = condensation_props.get('max_film_thickness_m', 5e-5)

            # Add single wall properties
            config.update({
                'thickness_w_m': thickness,
                'density_w_kg_m3': material.density_kg_m3,
                'cp_w_J_kg_K': material.specific_heat_J_kg_K,
                'k_w_W_m_K': material.thermal_conductivity_W_m_K,
                'epsilon_w': material.emissivity,
                'roughness_m': kwargs.get('roughness_m', material.roughness_m if material.roughness_m else 1e-6),
                'film_w': kwargs.get('film_w', default_film),
            })

        elif pipe_type == 'triple':
            # Get all three materials
            inner_name = kwargs.get('inner_material', 'stainless_316')
            middle_name = kwargs.get('middle_material', 'ceramic_fiber')
            outer_name = kwargs.get('outer_material', 'stainless_430')

            inner_mat = self.get_material(inner_name)
            middle_mat = self.get_material(middle_name)
            outer_mat = self.get_material(outer_name)

            if not all([inner_mat, middle_mat, outer_mat]):
                missing = []
                if not inner_mat: missing.append(inner_name)
                if not middle_mat: missing.append(middle_name)
                if not outer_mat: missing.append(outer_name)
                raise ValueError(f"Materials not found in pipe_props.json: {missing}")

            # Add triple wall properties with proper naming for ChimneyPart
            config.update({
                # Inner wall
                'thickness_inner_m': kwargs.get('inner_thickness_m', inner_mat.common_thickness_m),
                'density_inner_kg_m3': inner_mat.density_kg_m3,
                'cp_inner_J_kg_K': inner_mat.specific_heat_J_kg_K,
                'k_inner_W_m_K': inner_mat.thermal_conductivity_W_m_K,
                'epsilon_inner': inner_mat.emissivity,

                # Middle layer (insulation)
                'thickness_middle_m': kwargs.get('middle_thickness_m', middle_mat.common_thickness_m),
                'density_middle_kg_m3': middle_mat.density_kg_m3,
                'cp_middle_J_kg_K': middle_mat.specific_heat_J_kg_K,
                'k_middle_W_m_K': middle_mat.thermal_conductivity_W_m_K,

                # Outer wall
                'thickness_outer_m': kwargs.get('outer_thickness_m', outer_mat.common_thickness_m),
                'density_outer_kg_m3': outer_mat.density_kg_m3,
                'cp_outer_J_kg_K': outer_mat.specific_heat_J_kg_K,
                'k_outer_W_m_K': outer_mat.thermal_conductivity_W_m_K,
                'epsilon_outer': outer_mat.emissivity,

                # Air gaps
                'air_gap_inner_m': kwargs.get('air_gap_inner_m', 0.0),
                'air_gap_outer_m': kwargs.get('air_gap_outer_m', 0.0),

                # Additional properties
                'has_fittings': kwargs.get('has_fittings', False),
                # Use inner material roughness for flow calculations
                'roughness_m': kwargs.get('roughness_m', inner_mat.roughness_m if inner_mat.roughness_m else 1e-6),
                # Use inner material condensation properties
                'film_w': kwargs.get('film_w', self.condensation_properties.get(inner_name, {}).get('max_film_thickness_m', 5e-5)),
            })
        else:
            raise ValueError(f"Unknown pipe type: {pipe_type}. Must be 'single' or 'triple'")

        return config

    def create_single_wall_chimney_part(self,
                                       inner_diameter_ft: float,
                                       height_ft: float,
                                       material: str = "steel_24_gauge",
                                       **kwargs) -> Dict[str, Any]:
        """
        Create a single wall ChimneyPart dictionary using feet and material name.

        Args:
            inner_diameter_ft: Inner diameter in feet
            height_ft: Height in feet
            material: Material name from pipe_props.json
            **kwargs: Additional parameters (z_ft, T_ambient_K, thickness_m, etc.)

        Returns:
            Dictionary ready for Chimney(parts=[...])
        """
        return self.create_chimney_part_dict(
            pipe_type='single',
            inner_diameter_ft=inner_diameter_ft,
            height_ft=height_ft,
            material=material,
            **kwargs
        )

    def list_vent_options(self) -> List[Dict[str, Any]]:
        """Return the catalog for a frontend dropdown.

        Each entry has the selectable key plus display fields. The frontend
        sends back the `key`, geometry, and operating conditions; the backend
        does the rest via create_chimney_part_from_vent_option.
        """
        out = []
        for key, spec in self.vent_options.items():
            out.append({
                "key": key,
                "name": spec.get("name", key),
                "description": spec.get("description", ""),
                "construction": spec.get("construction"),
                "simulator_type": spec.get("simulator_type"),
                "max_flue_temp_C": spec.get("max_flue_temp_C"),
                "surface_finish": spec.get("surface_finish"),
            })
        return out

    def get_vent_option(self, vent_key: str) -> Dict[str, Any]:
        """Return raw vent_options entry for a given key."""
        if vent_key not in self.vent_options:
            available = ", ".join(sorted(self.vent_options.keys()))
            raise ValueError(
                f"Vent option '{vent_key}' not found. Available: {available}"
            )
        return self.vent_options[vent_key]

    def create_chimney_part_from_vent_option(self,
                                             vent_key: str,
                                             inner_diameter_ft: float,
                                             height_ft: float,
                                             z_ft: float = 0,
                                             orientation: str = "vertical",
                                             T_ambient_K: float = 293.15,
                                             Pa: float = 98457,
                                             num_segments: Optional[int] = None,
                                             **kwargs) -> Dict[str, Any]:
        """Build a ChimneyPart dict from a vent catalog key.

        The user picks a vent name in the frontend; this resolves the catalog
        entry's layers (material + thickness) and returns a dict compatible
        with Chimney(parts=[...]).

        simulator_type=="single" -> single-wall: uses the inner layer only.
        simulator_type=="triple" -> triple-wall: uses inner/middle/outer
            layers. For double-wall vents the middle layer is `air` with a
            finite thickness; air_gap_inner/outer are kept at 0 because the
            gap is already represented as the middle layer.
        """
        spec = self.get_vent_option(vent_key)
        sim_type = spec.get("simulator_type")
        layers = spec.get("layers", {})
        # Use the catalog's surface_finish to pick a roughness default; corrugated
        # liners get the inner material's roughness (already higher in JSON).
        kwargs.setdefault("humidity_ratio", 0.01)

        if sim_type == "single":
            inner = layers.get("inner")
            if not inner:
                raise ValueError(f"Vent '{vent_key}' has simulator_type=single but no 'inner' layer")
            return self.create_chimney_part_dict(
                pipe_type="single",
                inner_diameter_ft=inner_diameter_ft,
                height_ft=height_ft,
                z_ft=z_ft,
                orientation=orientation,
                T_ambient_K=T_ambient_K,
                Pa=Pa,
                num_segments=num_segments,
                material=inner["material"],
                thickness_m=inner["thickness_m"],
                **kwargs,
            )

        if sim_type == "triple":
            inner = layers.get("inner")
            middle = layers.get("middle")
            outer = layers.get("outer")
            missing = [n for n, l in (("inner", inner), ("middle", middle), ("outer", outer)) if not l]
            if missing:
                raise ValueError(
                    f"Vent '{vent_key}' has simulator_type=triple but is missing layers: {missing}"
                )
            return self.create_chimney_part_dict(
                pipe_type="triple",
                inner_diameter_ft=inner_diameter_ft,
                height_ft=height_ft,
                z_ft=z_ft,
                orientation=orientation,
                T_ambient_K=T_ambient_K,
                Pa=Pa,
                num_segments=num_segments,
                inner_material=inner["material"],
                middle_material=middle["material"],
                outer_material=outer["material"],
                inner_thickness_m=inner["thickness_m"],
                middle_thickness_m=middle["thickness_m"],
                outer_thickness_m=outer["thickness_m"],
                air_gap_inner_m=0.0,
                air_gap_outer_m=0.0,
                **kwargs,
            )

        raise ValueError(
            f"Vent '{vent_key}' has unsupported simulator_type={sim_type!r} (expected 'single' or 'triple')"
        )

    def create_triple_wall_chimney_part(self,
                                       inner_diameter_ft: float,
                                       height_ft: float,
                                       inner_material: str = "stainless_316",
                                       middle_material: str = "ceramic_fiber",
                                       outer_material: str = "stainless_430",
                                       **kwargs) -> Dict[str, Any]:
        """
        Create a triple wall ChimneyPart dictionary using feet and material names.

        Args:
            inner_diameter_ft: Inner diameter in feet
            height_ft: Height in feet
            inner_material: Inner wall material name
            middle_material: Insulation material name
            outer_material: Outer wall material name
            **kwargs: Additional parameters (z_ft, T_ambient_K, air_gap_inner_m, etc.)

        Returns:
            Dictionary ready for Chimney(parts=[...])
        """
        return self.create_chimney_part_dict(
            pipe_type='triple',
            inner_diameter_ft=inner_diameter_ft,
            height_ft=height_ft,
            inner_material=inner_material,
            middle_material=middle_material,
            outer_material=outer_material,
            **kwargs
        )