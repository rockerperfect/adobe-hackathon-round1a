"""
Unit tests for outline extraction components.

This module contains comprehensive unit tests for the outline extraction
functionality including heading detection and outline building components.
Tests validate core functionality against Adobe Hackathon requirements.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pdf_parser.base_parser import TextBlock
from outline_extractor.heading_detector import HeadingDetector, DetectedHeading
from outline_extractor.outline_builder import OutlineBuilder


class TestHeadingDetector(unittest.TestCase):
    """
    Test suite for HeadingDetector class.
    
    Validates the rule-based heading detection functionality including
    font analysis, pattern matching, and multilingual support.
    """
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        self.detector = HeadingDetector()
        
        # Create sample text blocks for testing
        self.sample_blocks = [
            TextBlock(
                text="Document Title",
                bbox=(50, 700, 500, 720),
                font_name="Arial-Bold",
                font_size=18.0,
                page_number=1
            ),
            TextBlock(
                text="1. Introduction",
                bbox=(50, 650, 200, 670),
                font_name="Arial-Bold", 
                font_size=14.0,
                page_number=1
            ),
            TextBlock(
                text="This is regular body text that should not be detected as a heading.",
                bbox=(50, 600, 450, 620),
                font_name="Arial",
                font_size=12.0,
                page_number=1
            ),
            TextBlock(
                text="1.1 Subsection",
                bbox=(70, 550, 250, 570),
                font_name="Arial-Bold",
                font_size=12.0,
                page_number=1
            ),
            TextBlock(
                text="2. Methodology",
                bbox=(50, 500, 200, 520),
                font_name="Arial-Bold",
                font_size=14.0,
                page_number=1
            )
        ]
    
    def test_heading_detector_initialization(self):
        """Test proper initialization of HeadingDetector."""
        detector = HeadingDetector()
        self.assertIsInstance(detector, HeadingDetector)
        self.assertIsNotNone(detector.patterns)
        self.assertIsNotNone(detector.multilingual_patterns)
    
    def test_font_based_detection(self):
        """Test heading detection based on font properties."""
        headings = self.detector.detect_headings(self.sample_blocks)
        
        # Should detect headings based on font size and weight
        self.assertGreater(len(headings), 0)
        
        # Verify that the document title is detected
        title_headings = [h for h in headings if h.text == "Document Title"]
        self.assertEqual(len(title_headings), 1)
        self.assertEqual(title_headings[0].level, 1)
    
    def test_pattern_based_detection(self):
        """Test heading detection based on text patterns."""
        headings = self.detector.detect_headings(self.sample_blocks)
        
        # Should detect numbered headings
        numbered_headings = [h for h in headings if h.text.startswith("1.")]
        self.assertGreater(len(numbered_headings), 0)
        
        # Verify hierarchical structure
        intro_heading = next((h for h in headings if h.text == "1. Introduction"), None)
        self.assertIsNotNone(intro_heading)
        
        subsection_heading = next((h for h in headings if h.text == "1.1 Subsection"), None)
        self.assertIsNotNone(subsection_heading)
        self.assertGreater(subsection_heading.level, intro_heading.level)
    
    def test_confidence_scoring(self):
        """Test confidence scoring for detected headings."""
        headings = self.detector.detect_headings(self.sample_blocks)
        
        for heading in headings:
            self.assertIsInstance(heading.confidence, float)
            self.assertGreaterEqual(heading.confidence, 0.0)
            self.assertLessEqual(heading.confidence, 1.0)
        
        # Title should have high confidence
        title_heading = next((h for h in headings if h.text == "Document Title"), None)
        if title_heading:
            self.assertGreater(title_heading.confidence, 0.7)
    
    def test_multilingual_pattern_detection(self):
        """Test detection of headings with multilingual patterns."""
        # Test Japanese patterns
        japanese_blocks = [
            TextBlock(
                text="第1章 はじめに",
                bbox=(50, 700, 200, 720),
                font_name="MS-Gothic",
                font_size=16.0,
                page_number=1
            ),
            TextBlock(
                text="1.1 概要",
                bbox=(70, 650, 150, 670),
                font_name="MS-Gothic",
                font_size=14.0,
                page_number=1
            )
        ]
        
        headings = self.detector.detect_headings(japanese_blocks)
        self.assertGreater(len(headings), 0)
        
        # Should detect Japanese chapter pattern
        chapter_heading = next((h for h in headings if "第1章" in h.text), None)
        self.assertIsNotNone(chapter_heading)
    
    def test_edge_cases(self):
        """Test handling of edge cases in heading detection."""
        # Test empty input
        empty_headings = self.detector.detect_headings([])
        self.assertEqual(len(empty_headings), 0)
        
        # Test single block
        single_block = [self.sample_blocks[0]]
        single_headings = self.detector.detect_headings(single_block)
        self.assertGreaterEqual(len(single_headings), 0)
        
        # Test blocks with no clear headings
        body_blocks = [
            TextBlock(
                text="Regular paragraph text without any heading indicators.",
                bbox=(50, 600, 450, 620),
                font_name="Arial",
                font_size=12.0,
                page_number=1
            )
        ]
        body_headings = self.detector.detect_headings(body_blocks)
        self.assertEqual(len(body_headings), 0)


class TestOutlineBuilder(unittest.TestCase):
    """
    Test suite for OutlineBuilder class.
    
    Validates the outline building functionality including hierarchical
    structure creation and JSON schema compliance.
    """
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        self.builder = OutlineBuilder()
        
        # Create sample detected headings
        self.sample_headings = [
            DetectedHeading(
                text="Document Title",
                level=1,
                page_number=1,
                bbox=(50, 700, 500, 720),
                confidence=0.95
            ),
            DetectedHeading(
                text="1. Introduction",
                level=2,
                page_number=1,
                bbox=(50, 650, 200, 670),
                confidence=0.90
            ),
            DetectedHeading(
                text="1.1 Overview",
                level=3,
                page_number=1,
                bbox=(70, 600, 180, 620),
                confidence=0.85
            ),
            DetectedHeading(
                text="1.2 Scope",
                level=3,
                page_number=2,
                bbox=(70, 550, 150, 570),
                confidence=0.88
            ),
            DetectedHeading(
                text="2. Methodology",
                level=2,
                page_number=2,
                bbox=(50, 500, 200, 520),
                confidence=0.92
            )
        ]
    
    def test_outline_builder_initialization(self):
        """Test proper initialization of OutlineBuilder."""
        builder = OutlineBuilder()
        self.assertIsInstance(builder, OutlineBuilder)
    
    def test_basic_outline_construction(self):
        """Test basic outline construction from detected headings."""
        outline = self.builder.build_outline(self.sample_headings)
        
        # Verify outline structure
        self.assertIsInstance(outline, dict)
        self.assertIn("outline", outline)
        self.assertIn("metadata", outline)
        
        # Check outline content
        outline_items = outline["outline"]
        self.assertGreater(len(outline_items), 0)
        
        # Verify hierarchical structure
        root_item = outline_items[0]
        self.assertEqual(root_item["title"], "Document Title")
        self.assertEqual(root_item["level"], 1)
        self.assertEqual(root_item["page"], 1)
    
    def test_hierarchical_structure(self):
        """Test proper hierarchical structure creation."""
        outline = self.builder.build_outline(self.sample_headings)
        outline_items = outline["outline"]
        
        # Find introduction section
        intro_item = next((item for item in outline_items 
                          if item["title"] == "1. Introduction"), None)
        self.assertIsNotNone(intro_item)
        
        # Check if it has children
        if "children" in intro_item and intro_item["children"]:
            children = intro_item["children"]
            self.assertGreater(len(children), 0)
            
            # Verify child levels are higher
            for child in children:
                self.assertGreater(child["level"], intro_item["level"])
    
    def test_json_schema_compliance(self):
        """Test that generated outline complies with JSON schema."""
        outline = self.builder.build_outline(self.sample_headings)
        
        # Required top-level fields
        self.assertIn("outline", outline)
        self.assertIn("metadata", outline)
        
        # Metadata structure
        metadata = outline["metadata"]
        self.assertIn("total_pages", metadata)
        self.assertIn("total_headings", metadata)
        self.assertIn("extraction_confidence", metadata)
        
        # Outline items structure
        for item in outline["outline"]:
            self.assertIn("title", item)
            self.assertIn("level", item)
            self.assertIn("page", item)
            self.assertIn("confidence", item)
            
            # Optional bbox field
            if "bbox" in item:
                self.assertIsInstance(item["bbox"], list)
                self.assertEqual(len(item["bbox"]), 4)
    
    def test_confidence_aggregation(self):
        """Test confidence score aggregation in metadata."""
        outline = self.builder.build_outline(self.sample_headings)
        metadata = outline["metadata"]
        
        self.assertIn("extraction_confidence", metadata)
        confidence = metadata["extraction_confidence"]
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_page_number_tracking(self):
        """Test proper page number tracking in outline."""
        outline = self.builder.build_outline(self.sample_headings)
        metadata = outline["metadata"]
        
        self.assertIn("total_pages", metadata)
        total_pages = metadata["total_pages"]
        
        # Should detect 2 pages from sample data
        self.assertEqual(total_pages, 2)
    
    def test_empty_headings_handling(self):
        """Test handling of empty headings list."""
        outline = self.builder.build_outline([])
        
        self.assertIn("outline", outline)
        self.assertIn("metadata", outline)
        
        # Empty outline
        self.assertEqual(len(outline["outline"]), 0)
        
        # Metadata should reflect empty state
        metadata = outline["metadata"]
        self.assertEqual(metadata["total_headings"], 0)
        self.assertEqual(metadata["total_pages"], 0)
    
    def test_single_heading_outline(self):
        """Test outline construction with single heading."""
        single_heading = [self.sample_headings[0]]
        outline = self.builder.build_outline(single_heading)
        
        self.assertEqual(len(outline["outline"]), 1)
        self.assertEqual(outline["metadata"]["total_headings"], 1)
    
    def test_outline_serialization(self):
        """Test that outline can be properly serialized to JSON."""
        outline = self.builder.build_outline(self.sample_headings)
        
        # Should be able to serialize to JSON without errors
        try:
            json_str = json.dumps(outline, indent=2)
            self.assertIsInstance(json_str, str)
            
            # Should be able to deserialize back
            parsed_outline = json.loads(json_str)
            self.assertEqual(outline, parsed_outline)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Outline serialization failed: {e}")


class TestIntegratedOutlineExtraction(unittest.TestCase):
    """
    Integration tests for the complete outline extraction pipeline.
    
    Tests the coordination between HeadingDetector and OutlineBuilder
    to ensure proper end-to-end functionality.
    """
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.detector = HeadingDetector()
        self.builder = OutlineBuilder()
        
        # Create comprehensive test document
        self.document_blocks = [
            TextBlock(
                text="Annual Report 2024",
                bbox=(50, 750, 400, 770),
                font_name="Arial-Bold",
                font_size=20.0,
                page_number=1
            ),
            TextBlock(
                text="Executive Summary",
                bbox=(50, 700, 250, 720),
                font_name="Arial-Bold",
                font_size=16.0,
                page_number=1
            ),
            TextBlock(
                text="This report presents our findings for the fiscal year 2024.",
                bbox=(50, 650, 450, 670),
                font_name="Arial",
                font_size=12.0,
                page_number=1
            ),
            TextBlock(
                text="1. Financial Performance",
                bbox=(50, 600, 300, 620),
                font_name="Arial-Bold",
                font_size=14.0,
                page_number=1
            ),
            TextBlock(
                text="1.1 Revenue Analysis",
                bbox=(70, 550, 280, 570),
                font_name="Arial-Bold",
                font_size=12.0,
                page_number=1
            ),
            TextBlock(
                text="Revenue increased by 15% compared to previous year.",
                bbox=(70, 500, 400, 520),
                font_name="Arial",
                font_size=12.0,
                page_number=1
            ),
            TextBlock(
                text="1.2 Cost Structure",
                bbox=(70, 450, 250, 470),
                font_name="Arial-Bold",
                font_size=12.0,
                page_number=2
            ),
            TextBlock(
                text="2. Market Analysis",
                bbox=(50, 400, 250, 420),
                font_name="Arial-Bold",
                font_size=14.0,
                page_number=2
            ),
            TextBlock(
                text="3. Future Outlook",
                bbox=(50, 350, 200, 370),
                font_name="Arial-Bold",
                font_size=14.0,
                page_number=3
            )
        ]
    
    def test_end_to_end_extraction(self):
        """Test complete outline extraction pipeline."""
        # Step 1: Detect headings
        headings = self.detector.detect_headings(self.document_blocks)
        self.assertGreater(len(headings), 0)
        
        # Step 2: Build outline
        outline = self.builder.build_outline(headings)
        
        # Verify complete pipeline results
        self.assertIsInstance(outline, dict)
        self.assertIn("outline", outline)
        self.assertIn("metadata", outline)
        
        # Check that major sections are detected
        outline_items = outline["outline"]
        titles = [item["title"] for item in outline_items]
        
        # Should detect main title
        self.assertIn("Annual Report 2024", titles)
        
        # Should detect numbered sections
        financial_sections = [title for title in titles if "Financial Performance" in title]
        self.assertGreater(len(financial_sections), 0)
    
    def test_hierarchical_integrity(self):
        """Test integrity of hierarchical structure in full pipeline."""
        headings = self.detector.detect_headings(self.document_blocks)
        outline = self.builder.build_outline(headings)
        
        # Verify hierarchical levels are logical
        outline_items = outline["outline"]
        for item in outline_items:
            if "children" in item and item["children"]:
                parent_level = item["level"]
                for child in item["children"]:
                    self.assertGreater(child["level"], parent_level)
    
    def test_performance_requirements(self):
        """Test that extraction meets performance requirements."""
        import time
        
        start_time = time.time()
        
        # Run full pipeline
        headings = self.detector.detect_headings(self.document_blocks)
        outline = self.builder.build_outline(headings)
        
        processing_time = time.time() - start_time
        
        # Should complete well within Adobe Hackathon constraints
        self.assertLess(processing_time, 1.0)
        
        # Verify output quality
        self.assertGreater(len(outline["outline"]), 0)
        self.assertGreater(outline["metadata"]["total_headings"], 0)
    
    def test_multilingual_integration(self):
        """Test integrated pipeline with multilingual content."""
        # Add multilingual content
        multilingual_blocks = self.document_blocks + [
            TextBlock(
                text="付録A 技術仕様",
                bbox=(50, 300, 200, 320),
                font_name="MS-Gothic",
                font_size=14.0,
                page_number=4
            ),
            TextBlock(
                text="A.1 システム要件",
                bbox=(70, 250, 250, 270),
                font_name="MS-Gothic",
                font_size=12.0,
                page_number=4
            )
        ]
        
        headings = self.detector.detect_headings(multilingual_blocks)
        outline = self.builder.build_outline(headings)
        
        # Should handle multilingual content
        self.assertGreater(len(outline["outline"]), len(self.document_blocks) // 2)
        
        # Check for Japanese content in outline
        titles = [item["title"] for item in outline["outline"]]
        japanese_titles = [title for title in titles if "付録" in title or "技術仕様" in title]
        self.assertGreater(len(japanese_titles), 0)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
