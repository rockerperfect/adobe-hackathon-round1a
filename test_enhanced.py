#!/usr/bin/env python3

import os
from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.outline_extractor.outline_builder import OutlineBuilder
from src.outline_extractor.heading_detector import HeadingDetector

# Process file02.pdf
input_file = 'input/file02.pdf'

parser = PyMuPDFParser(input_file)
outline_builder = OutlineBuilder()
heading_detector = HeadingDetector()

print('Extracting text from PDF...')
document = parser.parse_pdf(input_file)
text_blocks = document['text_blocks']

print(f'Document keys: {list(document.keys())}')
print(f'Number of text blocks: {len(text_blocks)}')

print('Detecting headings...')
detected_headings = heading_detector.detect_headings(text_blocks)
print(f'Number of detected headings: {len(detected_headings)}')

print('Building outline...')
try:
    outline = outline_builder.build_outline(detected_headings, text_blocks=text_blocks)
    print(f'Title: {outline["title"]}')
    print(f'Number of outline entries: {len(outline["outline"])}')
    print('\nOutline entries:')
    for i, entry in enumerate(outline['outline']):
        print(f'{i+1}. {entry["text"]} (Level: {entry["level"]})')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
