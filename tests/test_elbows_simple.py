"""
Simple test of elbow calculations without external dependencies.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test that modules can be imported and basic calculations work."""
    print("Testing elbow loss calculations...")
    
    # Test imports
    try:
        from flued_module.flow.elbows import (
            ElbowSpec, relative_roughness, churchill_f, base_K
        )
        print("✓ Module imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test relative roughness
    try:
        rel_eps = relative_roughness("commercial_steel", 0.1)
        assert 0.0001 < rel_eps < 0.001
        print(f"✓ Relative roughness: ε/D = {rel_eps:.6f}")
    except Exception as e:
        print(f"✗ Relative roughness test failed: {e}")
        return False
    
    # Test Churchill friction factor
    try:
        f = churchill_f(Re=10000, rel_eps=0.0001)
        assert 0.01 < f < 0.1
        print(f"✓ Churchill friction factor: f = {f:.6f}")
    except Exception as e:
        print(f"✗ Churchill friction test failed: {e}")
        return False
    
    # Test base K calculation
    try:
        spec = ElbowSpec(
            angle_deg=90,
            diameter_m=0.1,
            R_over_D=1.5,
            elbow_type="long_radius",
            smoothness="commercial_steel"
        )
        K = base_K(spec)
        assert 0.1 < K < 2.0
        print(f"✓ Base K coefficient: K = {K:.3f}")
    except Exception as e:
        print(f"✗ Base K test failed: {e}")
        return False
    
    # Test elbow type inference
    try:
        from flued_module.flow.elbows_infer import estimate_elbow_type_from_geometry
        elbow_type, R_over_D = estimate_elbow_type_from_geometry(angle_deg=90)
        print(f"✓ Elbow type inference: type={elbow_type}, R/D={R_over_D}")
    except Exception as e:
        print(f"✗ Elbow inference test failed: {e}")
        return False
    
    print("\nAll basic tests passed!")
    return True

def test_integration_with_heat_transfer():
    """Test integration with comprehensive heat transfer module."""
    print("\nTesting integration with heat transfer module...")
    
    try:
        from flued_module.comprehensive_heat_transfer import ComprehensiveHeatTransfer
        ht = ComprehensiveHeatTransfer()
        
        # Test backward compatibility shim
        elbows = ht._create_default_elbow_from_fittings(has_fittings=True, diameter=0.15)
        if elbows and len(elbows) == 1:
            print(f"✓ Backward compatibility shim works: created {len(elbows)} default elbow")
        else:
            print("✗ Backward compatibility shim failed")
            return False
            
    except ImportError as e:
        print(f"⚠ Could not test integration (missing dependencies): {e}")
        return None
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
    
    print("Integration tests passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("ELBOW LOSS CALCULATION TEST SUITE")
    print("=" * 60)
    
    # Run basic tests
    basic_ok = test_basic_functionality()
    
    # Run integration tests if possible
    integration_ok = test_integration_with_heat_transfer()
    
    print("\n" + "=" * 60)
    if basic_ok:
        if integration_ok is None:
            print("RESULT: Basic tests PASSED, integration tests SKIPPED")
        elif integration_ok:
            print("RESULT: All tests PASSED")
        else:
            print("RESULT: Basic tests PASSED, integration tests FAILED")
    else:
        print("RESULT: Tests FAILED")
    print("=" * 60)
    
    return basic_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)