"""
Master Test Runner for Backdraft Implementation
Runs all backdraft test scenarios and summarizes results.

Test Suite:
1. Normal Draft Validation - ensures backward compatibility
2. Wind-Induced Backdraft - adverse wind pressure
3. Backdraft → Recovery Transition - dynamic scenario
4. Cold Start Backdraft - inverted buoyancy

Usage:
    python run_all_backdraft_tests.py
"""

import subprocess
import sys
import time

tests = [
    ("Test 1: Normal Draft Validation", "test_backdraft_normal_draft.py"),
    ("Test 2: Wind-Induced Backdraft", "test_backdraft_wind_induced.py"),
    ("Test 3: Backdraft → Recovery", "test_backdraft_transition.py"),
    ("Test 4: Cold Start Backdraft", "test_backdraft_cold_start.py"),
]

print("="*80)
print("BACKDRAFT IMPLEMENTATION TEST SUITE")
print("="*80)
print(f"Running {len(tests)} test scenarios...")
print()

results = []
start_time = time.time()

for i, (name, script) in enumerate(tests, 1):
    print(f"\n{'='*80}")
    print(f"Running {name} ({i}/{len(tests)})")
    print(f"{'='*80}\n")

    try:
        # Run test script
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=600  # 10 minute timeout per test
        )

        if result.returncode == 0:
            results.append((name, "✓ PASS"))
        else:
            results.append((name, "✗ FAIL"))

    except subprocess.TimeoutExpired:
        results.append((name, "⏱ TIMEOUT"))
        print(f"\n⚠️  Test timed out after 10 minutes")

    except Exception as e:
        results.append((name, f"✗ ERROR: {str(e)}"))
        print(f"\n⚠️  Test error: {e}")

    print(f"\n{'='*80}\n")
    time.sleep(1)  # Brief pause between tests

elapsed_time = time.time() - start_time

# Print summary
print("\n" + "="*80)
print("TEST SUITE SUMMARY")
print("="*80)

for name, status in results:
    print(f"{status:15s} | {name}")

passed = sum(1 for _, status in results if "PASS" in status)
failed = sum(1 for _, status in results if "FAIL" in status or "ERROR" in status)
timeout = sum(1 for _, status in results if "TIMEOUT" in status)

print("\n" + "-"*80)
print(f"Total: {len(tests)} tests")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Timeout: {timeout}")
print(f"Time: {elapsed_time:.1f} seconds")
print("="*80)

# Exit with error code if any tests failed
if failed > 0 or timeout > 0:
    print("\n⚠️  Some tests did not pass!")
    sys.exit(1)
else:
    print("\n✓ All tests passed!")
    sys.exit(0)
