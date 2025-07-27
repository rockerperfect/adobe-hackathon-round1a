#!/usr/bin/env python3
"""Test multi-line title detection specifically."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.outline_extractor.heading_detector import HeadingDetector  
from src.outline_extractor.outline_builder import UniversalOutlineBuilder
import json

def test_multiline_title_detection():
    """Test specifically for multi-line title detection like 'Foundation Level Extensions'."""
    
    heading_detector = HeadingDetector(debug_mode=True)
    builder = UniversalOutlineBuilder()
    
    # Test file02 which should have "Foundation Level Extensions"
    pdf_file = 'file02.pdf'
    print(f"=== Testing multi-line title detection in {pdf_file} ===")
    
    try:
        # Create parser for this specific file
        parser = PyMuPDFParser(f'input/{pdf_file}')
        
        # Parse PDF to get text blocks and metadata
        parsing_result = parser.parse_pdf(f'input/{pdf_file}')
        text_blocks = parsing_result['text_blocks']
        document_metadata = parsing_result['metadata']
        
        print(f"Total text blocks: {len(text_blocks)}")
        
        # Show first page text blocks to see fragmentation
        first_page_blocks = [b for b in text_blocks if getattr(b, 'page_number', getattr(b, 'page', 1)) == 1]
        print(f"First page blocks: {len(first_page_blocks)}")
        
        # Look for title-like blocks
        print("\nLarge font blocks on first page (potential title fragments):")
        large_font_blocks = [b for b in first_page_blocks if b.font_size >= 14]
        for i, block in enumerate(large_font_blocks[:10]):
            print(f"  Block {i}: '{block.text}' (font: {block.font_size}, pos: {block.x:.1f}, {block.y:.1f})")
        
        # Detect headings
        detected_headings = heading_detector.detect_headings(text_blocks)
        print(f"\nDetected {len(detected_headings)} potential headings")
        
        # Show headings that might be the title
        print("\nPotential title headings:")
        for heading in detected_headings[:5]:
            print(f"  '{heading.text}' (level: {heading.level}, conf: {heading.confidence:.2f}, font: {heading.font_size})")
        
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
        
        print(f"\nFinal extracted title: '{title}'")
        print(f"Total outline entries: {len(outline)}")
        
        # Check if we successfully merged multi-line titles
        multiline_indicators = [
            'Foundation Level Extensions' in title,
            'Foundation' in title and 'Level' in title and 'Extensions' in title,
            len(title.split()) >= 3
        ]
        
        if any(multiline_indicators):
            print("✅ Successfully detected multi-line title!")
        else:
            print("❌ May have missed multi-line title components")
            
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multiline_title_detection()
