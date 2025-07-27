#!/usr/bin/env python3
"""
Test script for the adaptive PDF extraction system.
Tests the system on all sample files to verify it works without hardcoding.
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.outline_extractor.outline_builder import OutlineBuilder
from src.utils.document_analyzer import AdaptiveDocumentAnalyzer

def test_adaptive_extraction():
    """Test the adaptive extraction system on all sample files."""
    
    # Input and output directories
    input_dir = Path("input")
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in input directory")
        return
    
    print(f"Testing adaptive system on {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    print()
    
    # Initialize components
    parser = PyMuPDFParser()
    outline_builder = OutlineBuilder()
    
    results = {}
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        start_time = time.time()
        
        try:
            # Parse PDF
            print(f"  Parsing PDF...")
            text_blocks = parser.parse_pdf(str(pdf_file))
            print(f"  Extracted {len(text_blocks)} text blocks")
            
            # Build outline with adaptive analysis
            print(f"  Building outline with adaptive analysis...")
            outline = outline_builder.build_outline(text_blocks)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Output file
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(outline, f, indent=2, ensure_ascii=False)
            
            # Store test results
            results[pdf_file.name] = {
                'success': True,
                'processing_time': processing_time,
                'title': outline.get('title', 'Not found'),
                'heading_count': len(outline.get('headings', [])),
                'output_file': str(output_file)
            }
            
            print(f"  ✅ Success: {processing_time:.2f}s")
            print(f"     Title: {outline.get('title', 'Not found')}")
            print(f"     Headings: {len(outline.get('headings', []))}")
            print()
            
        except Exception as e:
            processing_time = time.time() - start_time
            results[pdf_file.name] = {
                'success': False,
                'processing_time': processing_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            print(f"  ❌ Failed: {processing_time:.2f}s")
            print(f"     Error: {e}")
            print()
    
    # Print summary
    print("=" * 60)
    print("ADAPTIVE SYSTEM TEST SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Files processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print()
    
    if successful > 0:
        avg_time = sum(r['processing_time'] for r in results.values() if r['success']) / successful
        print(f"Average processing time: {avg_time:.2f}s")
        print()
        
        print("Successful extractions:")
        for filename, result in results.items():
            if result['success']:
                print(f"  {filename}:")
                print(f"    Title: {result['title']}")
                print(f"    Headings: {result['heading_count']}")
                print(f"    Time: {result['processing_time']:.2f}s")
    
    if total - successful > 0:
        print()
        print("Failed extractions:")
        for filename, result in results.items():
            if not result['success']:
                print(f"  {filename}: {result['error']}")
    
    print()
    print("Test completed! Check test_output/ directory for JSON results.")
    
    return results

if __name__ == "__main__":
    test_adaptive_extraction()
