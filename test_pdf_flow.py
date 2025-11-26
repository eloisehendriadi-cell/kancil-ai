#!/usr/bin/env python
"""
Quick test to verify PDF extraction and workspace seeding work.
Run with: python test_pdf_flow.py
"""
import os
import sys
import json

# Ensure we can import from the workspace
sys.path.insert(0, os.path.dirname(__file__))

# Set required env var
os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "test-key")

def test_imports():
    """Test that all modules import correctly."""
    print("[1] Testing imports...")
    try:
        import workspace
        import notes_app
        import quiz_app
        import llm_client
        print("    ✓ All modules import successfully")
        return True
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        return False

def test_app_creation():
    """Test that the Flask app creates successfully."""
    print("[2] Testing Flask app creation...")
    try:
        from app import create_app
        app = create_app()
        print("    ✓ Flask app created successfully")
        return app
    except Exception as e:
        print(f"    ✗ App creation failed: {e}")
        return None

def test_routes(app):
    """Test that required routes exist."""
    print("[3] Testing routes...")
    routes = [str(r) for r in app.url_map.iter_rules()]
    
    required_routes = [
        "/notes/extract_pdf",
        "/workspace/seed",
    ]
    
    missing = []
    for route in required_routes:
        if not any(route in r for r in routes):
            missing.append(route)
    
    if missing:
        print(f"    ✗ Missing routes: {missing}")
        return False
    
    print(f"    ✓ All required routes found")
    return True

def test_llm_config():
    """Test LLM configuration."""
    print("[4] Testing LLM configuration...")
    try:
        from llm_config import get_llm_config, validate_config, LLM_PROVIDER
        config = get_llm_config()
        print(f"    ✓ LLM Config loaded: provider={config['provider']}, model={config['model']}")
        
        # Note: Don't validate now since we might not have a real API key
        return True
    except Exception as e:
        print(f"    ✗ LLM config failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PDF Submission Flow - Diagnostic Test")
    print("=" * 60)
    
    if not test_imports():
        return False
    
    app = test_app_creation()
    if not app:
        return False
    
    if not test_routes(app):
        return False
    
    if not test_llm_config():
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nThe app should work. If you're still seeing 404 errors:")
    print("1. Check your browser's Network tab for the actual failing request URL")
    print("2. Make sure Flask is running with: python app.py")
    print("3. Check that you have a valid LLM_API_KEY set in your .env file")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
