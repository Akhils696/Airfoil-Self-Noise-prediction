"""
Test script to verify XAI environment setup
"""
import sys
import importlib

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'shap', 'matplotlib', 'openpyxl', 'xlrd'
    ]
    
    print("Checking dependencies...")
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_xai.txt")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def check_files():
    """Check if required files exist"""
    import os
    
    required_files = ['Preprocessed_Data.xls', 'hybrid_model.pkl']
    print("\nChecking required files...")
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✓ {file} ({size/1024:.1f} KB)")
        else:
            print(f"✗ {file} (MISSING)")
            missing.append(file)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        print("Please ensure these files are in the working directory.")
        return False
    else:
        print("\nAll required files are present!")
        return True

if __name__ == '__main__':
    print("=" * 50)
    print("XAI ENVIRONMENT CHECK")
    print("=" * 50)
    
    deps_ok = check_dependencies()
    files_ok = check_files()
    
    print("\n" + "=" * 50)
    if deps_ok and files_ok:
        print("✓ Environment is ready!")
        print("Run: python xai_hybrid_regression.py")
    else:
        print("✗ Environment needs setup!")
        if not deps_ok:
            print("1. Install missing dependencies")
        if not files_ok:
            print("2. Ensure data and model files are present")
    print("=" * 50)