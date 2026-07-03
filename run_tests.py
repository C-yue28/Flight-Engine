"""
Test runner for 6DOF flight simulation engine
Formatting of output done with AI
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    parent_dir = Path(__file__).resolve().parent
    sys.path.append(str(parent_dir))
    
    print("=" * 70)
    print("6DOF Flight Simulation Engine - Test Suite")
    print("=" * 70)
    print()
    
    # Test files
    test_files = [
        'tests/test_dynamics.py',
        'tests/test_aerodynamics.py'
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        print(f"\nRunning {test_file}...")
        print("-" * 70)
        
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_file, '-v'],
            cwd=str(parent_dir),
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"{test_file} PASSED")
        else:
            print(f"{test_file} FAILED")
            total_failed += 1
    
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if total_failed == 0:
        print("All tests PASSED")
        return 0
    else:
        print(f"{total_failed} test file(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
