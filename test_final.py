#!/usr/bin/env python3

import os
import json
from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.outline_extractor.outline_builder import OutlineBuilder
from src.outline_extractor.heading_detector import HeadingDetector

def test_file(filename):
    print(f'\n=== Testing {filename} ===')
    
    parser = PyMuPDFParser(f'input/{filename}')
    outline_builder = OutlineBuilder()
    heading_detector = HeadingDetector()

    # Extract and process
    document = parser.parse_pdf(f'input/{filename}')
    text_blocks = document['text_blocks']
    detected_headings = heading_detector.detect_headings(text_blocks)
    outline = outline_builder.build_outline(detected_headings, text_blocks=text_blocks)

    print(f'Title: "{outline["title"]}"')
    print(f'Number of outline entries: {len(outline["outline"])}')
    
    if len(outline["outline"]) == 0:
        print('Empty outline (likely a form document)')
    else:
        print('\nOutline entries:')
        for i, entry in enumerate(outline['outline']):
            print(f'{i+1:2d}. {entry["text"]} (Level: {entry["level"]})')
    
    # Save to file for comparison
    output_filename = f'output/{filename.replace(".pdf", ".json")}'
    os.makedirs('output', exist_ok=True)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(outline, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to: {output_filename}')

# Test both files
test_file('file01.pdf')
test_file('file02.pdf')
