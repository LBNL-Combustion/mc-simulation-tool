"""
Demonstrate the chimney geometry summary sanity check.
"""

def test_simple_chimney():
    """Test a simple L-shaped chimney."""
    from flued_module import Chimney
    
    print("\n" + "="*70)
    print("TEST 1: Simple L-shaped chimney")
    print("="*70)
    
    FT = 0.3048
    IN = 0.0254
    
    parts = [
        {
            "type": "single",
            "inner_diameter_m": 4 * IN,
            "height_m": 6 * FT,
            "orientation": "vertical",
            "num_segments": 3,
        },
        {
            "type": "single",
            "inner_diameter_m": 4 * IN,
            "height_m": 2 * FT,
            "orientation": "horizontal",
            "num_segments": 2,
        },
    ]
    
    elbows = [
        {
            "between_segments": [0, 1],
            "angle_deg": 90,
            "R_over_D": 1.5,
            "elbow_type": "long_radius",
            "smoothness": "commercial_steel",
        },
    ]
    
    chimney = Chimney(
        parts=parts,
        elbows=elbows,
        elbow_method="K_sum",
        dt=1.0,
        total_time=10
    )
    
    return chimney

def test_complex_chimney():
    """Test a complex chimney with multiple elbows and wall types."""
    from flued_module import Chimney
    
    print("\n" + "="*70)
    print("TEST 2: Complex multi-segment chimney")
    print("="*70)
    
    FT = 0.3048
    IN = 0.0254
    
    parts = [
        {
            "type": "single",
            "inner_diameter_m": 6 * IN,
            "height_m": 4 * FT,
            "orientation": "vertical",
            "num_segments": 4,
        },
        {
            "type": "single",
            "inner_diameter_m": 6 * IN,
            "height_m": 3 * FT,
            "orientation": "horizontal",
            "num_segments": 3,
        },
        {
            "type": "triple",
            "inner_diameter_m": 6 * IN,
            "height_m": 8 * FT,
            "orientation": "vertical",
            "num_segments": 6,
            # Triple wall properties (simplified)
            "thickness_inner_m": 0.001,
            "thickness_middle_m": 0.001,
            "thickness_outer_m": 0.001,
            "air_gap_inner_m": 0.01,
            "air_gap_outer_m": 0.01,
        },
        {
            "type": "single",
            "inner_diameter_m": 6 * IN,
            "height_m": 1.5 * FT,
            "orientation": "horizontal",
            "num_segments": 2,
        },
    ]
    
    elbows = [
        {
            "between_segments": [0, 1],
            "angle_deg": 90,
            "R_over_D": 1.5,
            "elbow_type": "long_radius",
            "smoothness": "commercial_steel",
        },
        {
            "between_segments": [1, 2],
            "angle_deg": 90,
            "R_over_D": 1.0,
            "elbow_type": "short_radius",
            "smoothness": "commercial_steel",
        },
        {
            "between_segments": [2, 3],
            "angle_deg": 45,
            "R_over_D": 2.5,
            "elbow_type": "long_radius",
            "smoothness": "smooth",
            "vane_factor": 0.6,  # Has guide vanes
        },
    ]
    
    chimney = Chimney(
        parts=parts,
        elbows=elbows,
        elbow_method="equivalent_length",
        dt=0.5,
        total_time=10
    )
    
    return chimney

def test_diameter_change():
    """Test a chimney with changing diameters."""
    from flued_module import Chimney
    
    print("\n" + "="*70)
    print("TEST 3: Chimney with diameter changes")
    print("="*70)
    
    FT = 0.3048
    IN = 0.0254
    
    parts = [
        {
            "type": "single",
            "inner_diameter_m": 4 * IN,  # 4 inch
            "height_m": 5 * FT,
            "orientation": "vertical",
            "num_segments": 3,
        },
        {
            "type": "single",
            "inner_diameter_m": 6 * IN,  # 6 inch (expansion)
            "height_m": 3 * FT,
            "orientation": "vertical",
            "num_segments": 2,
        },
        {
            "type": "single",
            "inner_diameter_m": 5 * IN,  # 5 inch (reduction)
            "height_m": 4 * FT,
            "orientation": "vertical",
            "num_segments": 3,
        },
    ]
    
    elbows = []  # No elbows, just straight with diameter changes
    
    chimney = Chimney(
        parts=parts,
        elbows=elbows,
        dt=1.0,
        total_time=10
    )
    
    return chimney

def test_no_elbows():
    """Test a straight chimney with no elbows."""
    from flued_module import Chimney
    
    print("\n" + "="*70)
    print("TEST 4: Straight chimney (no elbows)")
    print("="*70)
    
    FT = 0.3048
    IN = 0.0254
    
    parts = [
        {
            "type": "single",
            "inner_diameter_m": 4 * IN,
            "height_m": 10 * FT,
            "orientation": "vertical",
            "num_segments": 5,
        },
    ]
    
    chimney = Chimney(
        parts=parts,
        # No elbows specified
        dt=1.0,
        total_time=10
    )
    
    return chimney

def test_mitered_elbow():
    """Test chimney with mitered elbow."""
    from flued_module import Chimney
    
    print("\n" + "="*70)
    print("TEST 5: Chimney with mitered elbow")
    print("="*70)
    
    FT = 0.3048
    IN = 0.0254
    
    parts = [
        {
            "type": "single",
            "inner_diameter_m": 5 * IN,
            "height_m": 4 * FT,
            "orientation": "vertical",
            "num_segments": 2,
        },
        {
            "type": "single",
            "inner_diameter_m": 5 * IN,
            "height_m": 2.5 * FT,
            "orientation": "horizontal",
            "num_segments": 2,
        },
    ]
    
    elbows = [
        {
            "between_segments": [0, 1],
            "angle_deg": 90,
            "R_over_D": 1.0,
            "elbow_type": "mitered",
            "smoothness": "galvanized",
        },
    ]
    
    chimney = Chimney(
        parts=parts,
        elbows=elbows,
        elbow_method="K_sum",
        dt=1.0,
        total_time=10
    )
    
    return chimney

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# CHIMNEY GEOMETRY SUMMARY DEMONSTRATION")
    print("#"*70)
    
    # Run all tests
    test_simple_chimney()
    test_complex_chimney()
    test_diameter_change()
    test_no_elbows()
    test_mitered_elbow()
    
    print("\n" + "#"*70)
    print("# All geometry summaries displayed successfully!")
    print("#"*70)