"""
Test elbow geometry validation to ensure configurations make physical sense.
"""

def test_geometry_validation():
    """Test various invalid elbow configurations."""
    from flued_module import Chimney
    
    # Basic valid parts configuration
    parts = [
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 1.0, 
         "orientation": "vertical", "num_segments": 2},
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 0.5,
         "orientation": "horizontal", "num_segments": 1},
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 1.0,
         "orientation": "vertical", "num_segments": 2},
    ]
    
    print("Testing elbow geometry validation...")
    print("=" * 60)
    
    # Test 1: Too many elbows
    print("\nTest 1: Too many elbows (4 elbows for 3 parts)")
    try:
        elbows_too_many = [
            {"between_segments": [0, 1], "angle_deg": 90},
            {"between_segments": [1, 2], "angle_deg": 90},
            {"between_segments": [0, 1], "angle_deg": 45},  # Duplicate junction
            {"between_segments": [2, 3], "angle_deg": 90},  # Part 3 doesn't exist
        ]
        chimney = Chimney(parts=parts, elbows=elbows_too_many, dt=1.0, total_time=10)
        print("  ✗ Should have failed but didn't!")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")
    
    # Test 2: Invalid part indices
    print("\nTest 2: Invalid part index (part 5 doesn't exist)")
    try:
        elbows_invalid_index = [
            {"between_segments": [0, 1], "angle_deg": 90},
            {"between_segments": [4, 5], "angle_deg": 90},  # Parts don't exist
        ]
        chimney = Chimney(parts=parts, elbows=elbows_invalid_index, dt=1.0, total_time=10)
        print("  ✗ Should have failed but didn't!")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")
    
    # Test 3: Non-adjacent parts
    print("\nTest 3: Non-adjacent parts (elbow between parts 0 and 2)")
    try:
        elbows_non_adjacent = [
            {"between_segments": [0, 2], "angle_deg": 90},  # Skip part 1
        ]
        chimney = Chimney(parts=parts, elbows=elbows_non_adjacent, dt=1.0, total_time=10)
        print("  ✗ Should have failed but didn't!")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")
    
    # Test 4: Duplicate junction
    print("\nTest 4: Duplicate elbow at same junction")
    try:
        elbows_duplicate = [
            {"between_segments": [0, 1], "angle_deg": 90},
            {"between_segments": [0, 1], "angle_deg": 45},  # Same junction
        ]
        chimney = Chimney(parts=parts, elbows=elbows_duplicate, dt=1.0, total_time=10)
        print("  ✗ Should have failed but didn't!")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")
    
    # Test 5: Missing between_segments
    print("\nTest 5: Missing between_segments specification")
    try:
        elbows_missing = [
            {"angle_deg": 90},  # No between_segments
        ]
        chimney = Chimney(parts=parts, elbows=elbows_missing, dt=1.0, total_time=10)
        print("  ✗ Should have failed but didn't!")
    except ValueError as e:
        print(f"  ✓ Correctly caught error: {e}")
    
    # Test 6: Valid configuration (should work)
    print("\nTest 6: Valid configuration (2 elbows for 3 parts)")
    try:
        elbows_valid = [
            {"between_segments": [0, 1], "angle_deg": 90, "elbow_type": "long_radius"},
            {"between_segments": [1, 2], "angle_deg": 90, "elbow_type": "short_radius"},
        ]
        chimney = Chimney(parts=parts, elbows=elbows_valid, dt=1.0, total_time=10)
        print("  ✓ Valid configuration accepted")
        print(f"    Created {len(chimney.segment_elbows)} elbow mappings")
    except ValueError as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test 7: Diameter mismatch warning
    print("\nTest 7: Diameter mismatch warning")
    parts_diam_mismatch = [
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 1.0,
         "orientation": "vertical", "num_segments": 2},
        {"type": "single", "inner_diameter_m": 0.15, "height_m": 0.5,  # 50% larger
         "orientation": "horizontal", "num_segments": 1},
    ]
    try:
        elbows_diam = [
            {"between_segments": [0, 1], "angle_deg": 90},
        ]
        print("  Creating elbow between 100mm and 150mm parts...")
        chimney = Chimney(parts=parts_diam_mismatch, elbows=elbows_diam, dt=1.0, total_time=10)
        print("  ✓ Warning should have been printed above")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test 8: Horizontal-to-horizontal warning
    print("\nTest 8: Horizontal-to-horizontal connection")
    parts_both_horiz = [
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 1.0,
         "orientation": "horizontal", "num_segments": 2},
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 0.5,
         "orientation": "horizontal", "num_segments": 1},
    ]
    try:
        elbows_horiz = [
            {"between_segments": [0, 1], "angle_deg": 45},  # Angle in horizontal plane
        ]
        print("  Creating elbow between two horizontal sections...")
        chimney = Chimney(parts=parts_both_horiz, elbows=elbows_horiz, dt=1.0, total_time=10)
        print("  ✓ Warning should have been printed above")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("Validation tests complete!")

def demonstrate_error_messages():
    """Show clear error messages for common mistakes."""
    from flued_module import Chimney
    
    print("\n" + "=" * 60)
    print("DEMONSTRATING ERROR MESSAGES")
    print("=" * 60)
    
    parts = [
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 1.0, "num_segments": 2},
        {"type": "single", "inner_diameter_m": 0.1, "height_m": 1.0, "num_segments": 2},
    ]
    
    # Example 1: User tries to add 3 elbows to 2 parts
    print("\nExample: User adds too many elbows")
    print("Code: 2 parts, 3 elbows")
    try:
        elbows = [
            {"between_segments": [0, 1], "angle_deg": 90},
            {"between_segments": [1, 2], "angle_deg": 90},
            {"between_segments": [2, 3], "angle_deg": 90},
        ]
        chimney = Chimney(parts=parts, elbows=elbows)
    except ValueError as e:
        print(f"Error message:\n  {e}")
    
    # Example 2: User forgets to specify between_segments
    print("\nExample: User forgets between_segments")
    print("Code: elbow without location")
    try:
        elbows = [
            {"angle_deg": 90, "elbow_type": "long_radius"},
        ]
        chimney = Chimney(parts=parts, elbows=elbows)
    except ValueError as e:
        print(f"Error message:\n  {e}")
    
    # Example 3: User uses wrong indices
    print("\nExample: User uses 1-based indexing instead of 0-based")
    print("Code: between_segments: [1, 2] for 2 parts (should be [0, 1])")
    try:
        elbows = [
            {"between_segments": [1, 2], "angle_deg": 90},
        ]
        chimney = Chimney(parts=parts, elbows=elbows)
    except ValueError as e:
        print(f"Error message:\n  {e}")

if __name__ == "__main__":
    test_geometry_validation()
    demonstrate_error_messages()