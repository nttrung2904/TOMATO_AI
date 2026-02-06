"""
Quick Test Script - Check if CV features work correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tomato'))

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    try:
        import cv2
        import numpy as np
        import tensorflow as tf
        from flask import Flask
        print("✓ Core libraries imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_app_config():
    """Test if app configuration is correct"""
    print("\nTesting app configuration...")
    try:
        from tomato.app import app, MODELS, DEFAULT_MODEL, DEFAULT_PIPELINE, PIPELINES
        
        print(f"✓ MODELS: {MODELS}")
        print(f"✓ DEFAULT_MODEL: {DEFAULT_MODEL}")
        print(f"✓ DEFAULT_PIPELINE: {DEFAULT_PIPELINE}")
        print(f"✓ PIPELINES: {list(PIPELINES.keys())}")
        
        # Test pipeline access
        for key in PIPELINES.keys():
            fn, desc = PIPELINES[key]
            print(f"  - {key}: {desc}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cv_functions():
    """Test CV utility functions"""
    print("\nTesting CV functions...")
    try:
        from tomato.utils import (
            generate_gradcam, 
            overlay_heatmap_on_image,
            enhance_image_quality,
            check_image_quality,
            detect_leaf_region
        )
        print("✓ All CV functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ CV function import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_routes():
    """Test if CV routes are registered"""
    print("\nTesting routes...")
    try:
        from tomato.app import app
        
        cv_routes = [
            '/api/gradcam',
            '/api/enhance_image',
            '/api/check_quality',
            '/api/detect_leaf',
            '/webcam',
            '/api/webcam_predict'
        ]
        
        all_routes = [str(rule) for rule in app.url_map.iter_rules()]
        
        for route in cv_routes:
            if route in all_routes:
                print(f"✓ Route {route} registered")
            else:
                print(f"✗ Route {route} NOT found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Route test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("CV FEATURES TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_config,
        test_cv_functions,
        test_routes
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ All tests passed! CV features are ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
