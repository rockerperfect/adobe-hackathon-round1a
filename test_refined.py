#!/usr/bin/env python3
"""Test the refined outline extraction system without hardcoded patterns."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.outline_extractor.heading_detector import HeadingDetector  
from src.outline_extractor.outline_builder import UniversalOutlineBuilder
import json

def test_refined_system():
    """Test the system after removing hardcoded patterns."""
    
    heading_detector = HeadingDetector()
    builder = UniversalOutlineBuilder()
    
    # Test with each file
    test_files = ['file01.pdf', 'file02.pdf', 'file03.pdf', 'file04.pdf', 'file05.pdf']
    
    for pdf_file in test_files:
        print(f"\n=== Testing {pdf_file} ===")
        
        try:
            # Create parser for this specific file
            parser = PyMuPDFParser(f'input/{pdf_file}')
            
            # Parse PDF to get text blocks and metadata
            parsing_result = parser.parse_pdf(f'input/{pdf_file}')
            text_blocks = parsing_result['text_blocks']
            document_metadata = parsing_result['metadata']
            
            print(f"Extracted {len(text_blocks)} text blocks")
            
            # Detect headings
            detected_headings = heading_detector.detect_headings(text_blocks)
            print(f"Detected {len(detected_headings)} potential headings")
            
            # Build outline structure
            outline_structure = builder.build_outline(
                detected_headings=detected_headings,
                document_metadata=document_metadata,
                document_path=f'input/{pdf_file}',
                text_blocks=text_blocks
            )
            
            # Display results
            title = outline_structure.get('title', 'Not found')
            outline = outline_structure.get('outline', [])
            
            print(f"Title: {title}")
            print(f"Outline entries: {len(outline)}")
            
            # Show first few outline entries
            for i, entry in enumerate(outline[:5]):
                level = entry.get('level', '?')
                text = entry.get('text', '?')
                page = entry.get('page', '?')
                print(f"  {level}: {text[:60]}... (page {page})")
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_refined_system()
