#!/usr/bin/env python3
"""
Docker validation script for Adobe Hackathon Round 1A
Verifies that the solution meets all Docker requirements
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_dockerfile():
    """Check if Dockerfile meets requirements"""
    print("🔍 Checking Dockerfile...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    content = dockerfile_path.read_text()
    
    # Check platform specification
    if "--platform=linux/amd64" not in content:
        print("❌ Missing --platform=linux/amd64 specification")
        return False
    
    print("✅ Dockerfile has AMD64 platform specification")
    return True

def check_requirements():
    """Check if requirements.txt is reasonable size"""
    print("🔍 Checking requirements...")
    
    req_path = Path("requirements.txt")
    if not req_path.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Basic size check (requirements should be reasonable)
    content = req_path.read_text()
    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
    
    if len(lines) > 20:
        print(f"⚠️  Many dependencies ({len(lines)} packages)")
    else:
        print(f"✅ Reasonable dependencies ({len(lines)} packages)")
    
    return True

def check_main_script():
    """Check if main.py has Docker support"""
    print("🔍 Checking main.py Docker compatibility...")
    
    main_path = Path("main.py")
    if not main_path.exists():
        print("❌ main.py not found")
        return False
    
    content = main_path.read_text()
    
    # Check for Docker-specific behavior
    if "/app/input" not in content or "/app/output" not in content:
        print("❌ Missing Docker path handling")
        return False
    
    if "--batch" not in content:
        print("❌ Missing batch processing support")
        return False
    
    print("✅ main.py has Docker support")
    return True

def test_batch_processing():
    """Test batch processing functionality"""
    print("🔍 Testing batch processing...")
    
    if not Path("input").exists():
        print("⚠️  No input directory found, skipping batch test")
        return True
    
    # Count PDF files
    pdf_files = list(Path("input").glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found, skipping batch test")
        return True
    
    print(f"📁 Found {len(pdf_files)} PDF files for testing")
    
    # Test with a timeout to ensure it doesn't hang
    try:
        result = subprocess.run([
            sys.executable, "main.py", "input", "output", "--batch"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Batch processing test successful")
            return True
        else:
            print(f"❌ Batch processing failed: {result.stderr}")
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ Batch processing timed out")
        return False
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False

def check_output_format():
    """Check if output files are in correct format"""
    print("🔍 Checking output format...")
    
    output_dir = Path("output")
    if not output_dir.exists():
        print("⚠️  No output directory found")
        return True
    
    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        print("⚠️  No JSON output files found")
        return True
    
    # Validate JSON structure
    for json_file in json_files[:3]:  # Check first 3 files
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Check required structure
            if "title" not in data or "outline" not in data:
                print(f"❌ Invalid JSON structure in {json_file}")
                return False
            
            if not isinstance(data["outline"], list):
                print(f"❌ Outline must be a list in {json_file}")
                return False
            
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {json_file}")
            return False
    
    print(f"✅ Output format validation passed ({len(json_files)} files)")
    return True

def main():
    """Main validation function"""
    print("=" * 50)
    print("🐳 Docker Validation for Adobe Hackathon Round 1A")
    print("=" * 50)
    
    checks = [
        check_dockerfile,
        check_requirements,
        check_main_script,
        test_batch_processing,
        check_output_format
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Validation Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Ready for Docker deployment.")
        print("\nTo build and run:")
        print("docker build --platform linux/amd64 -t mysolutionname:identifier .")
        print("docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:identifier")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
