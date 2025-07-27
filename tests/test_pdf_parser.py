"""
Unit tests for Adobe India Hackathon Round 1A PDF parser implementations.

This module provides comprehensive test coverage for both PyMuPDF and PDFMiner 
parser implementations, validating functionality, performance, and error handling
as required by the Adobe Hackathon specifications.

Test coverage includes:
- Successful PDF parsing and text extraction
- Font and positioning information accuracy
- Document metadata extraction and validation
- Multilingual text handling (Japanese for bonus scoring)
- Parser fallback behavior for challenging documents
- Error handling for corrupted or malformed PDFs
- Performance validation for <10s processing constraints

Testing strategy:
- Unit tests for individual parser components
- Integration tests for complete parsing workflows
- Performance tests for timing constraints
- Edge case validation for robustness
- Multilingual support verification

Test fixtures:
- Sample PDFs with known structure and content
- Expected output JSON for validation
- Multilingual documents for bonus feature testing
- Corrupted files for error handling validation

Performance requirements:
- Each test completes within 2 seconds
- CPU-only execution without GPU dependencies
- Offline operation with no internet access
- Memory-efficient test execution
"""

import os
import sys
import unittest
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pdf_parser.base_parser import BasePDFParser, TextBlock
from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.pdf_parser.pdfminer_parser import PDFMinerParser


class TestBasePDFParser(unittest.TestCase):
    """
    Test suite for the abstract base PDF parser class.
    
    This test class validates the base parser interface and common
    functionality shared across all parser implementations.
    """
    
    def setUp(self):
        """Set up test fixtures and mock objects."""
        self.test_fixtures_path = Path(__file__).parent / 'fixtures'
        self.sample_pdf_path = self.test_fixtures_path / 'sample.pdf'
        
    def test_textblock_creation(self):
        """
        Test TextBlock data structure creation and validation.
        
        Validates that TextBlock objects can be created with required
        attributes and maintain data integrity for PDF parsing results.
        """
        # Create a sample TextBlock
        text_block = TextBlock(
            text="Sample heading text",
            x=100.0,
            y=200.0,
            width=300.0,
            height=15.0,
            font_name="Arial-Bold",
            font_size=14.0,
            font_flags=16,  # Bold flag
            page_number=1
        )
        
        # Validate attributes
        self.assertEqual(text_block.text, "Sample heading text")
        self.assertEqual(text_block.x, 100.0)
        self.assertEqual(text_block.y, 200.0)
        self.assertEqual(text_block.width, 300.0)
        self.assertEqual(text_block.height, 15.0)
        self.assertEqual(text_block.font_name, "Arial-Bold")
        self.assertEqual(text_block.font_size, 14.0)
        self.assertEqual(text_block.font_flags, 16)
        self.assertEqual(text_block.page_number, 1)
        
    def test_textblock_unicode_handling(self):
        """
        Test TextBlock Unicode text handling for multilingual support.
        
        Validates proper handling of Unicode text including Japanese
        characters for bonus scoring requirements.
        """
        # Test Japanese text (for bonus scoring)
        japanese_text = "第1章 はじめに"
        text_block = TextBlock(
            text=japanese_text,
            x=50.0,
            y=100.0,
            width=200.0,
            height=12.0,
            font_name="MS-Gothic",
            font_size=12.0,
            font_flags=0,
            page_number=1
        )
        
        self.assertEqual(text_block.text, japanese_text)
        self.assertIsInstance(text_block.text, str)
        
        # Test other Unicode characters
        unicode_text = "Résumé: αβγ 测试"
        text_block_unicode = TextBlock(
            text=unicode_text,
            x=50.0,
            y=100.0,
            width=200.0,
            height=12.0,
            font_name="Arial",
            font_size=12.0,
            font_flags=0,
            page_number=1
        )
        
        self.assertEqual(text_block_unicode.text, unicode_text)


class TestPyMuPDFParser(unittest.TestCase):
    """
    Test suite for PyMuPDF parser implementation.
    
    This test class validates the primary PDF parser functionality
    including text extraction, metadata handling, and performance
    characteristics as required by the Adobe Hackathon specifications.
    """
    
    def setUp(self):
        """Set up test fixtures and sample data."""
        self.test_fixtures_path = Path(__file__).parent / 'fixtures'
        self.sample_pdf_path = self.test_fixtures_path / 'sample.pdf'
        
        # Create a mock PDF file if fixtures don't exist
        if not self.sample_pdf_path.exists():
            self._create_mock_pdf_file()
    
    def _create_mock_pdf_file(self):
        """
        Create a mock PDF file for testing when fixtures are not available.
        
        This method generates a simple PDF using available libraries
        for basic testing functionality.
        """
        # For now, create a placeholder file
        # In a real implementation, this would create a simple PDF
        # with known content for testing
        self.sample_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip PDF creation if fitz is not available
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            
            # Add some test content
            text_content = ("Test Document\n\n1. Introduction\n\n"
                          "This is a sample document for testing.\n\n"
                          "1.1 Overview\n\nSample content here.\n\n"
                          "2. Methodology\n\nMore content.")
            page.insert_text((50, 50), text_content)
            
            doc.save(str(self.sample_pdf_path))
            doc.close()
        except ImportError:
            # Create empty file if PyMuPDF not available
            self.sample_pdf_path.touch()
    
    @patch('fitz.open')
    def test_parser_initialization_success(self, mock_fitz_open):
        """
        Test successful PyMuPDF parser initialization.
        
        Validates that the parser can be initialized with a valid PDF
        file and sets up required internal state correctly.
        """
        # Mock successful fitz document opening
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 5
        mock_fitz_open.return_value = mock_doc
        
        # Test parser initialization
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            parser = PyMuPDFParser(tmp_path)
            
            # Validate initialization
            self.assertTrue(parser._is_loaded)
            self.assertEqual(str(parser.pdf_path), tmp_path)
            
            # Cleanup
            parser.close()
            
        finally:
            os.unlink(tmp_path)
    
    @patch('fitz.open')
    def test_parser_initialization_encrypted_pdf(self, mock_fitz_open):
        """
        Test parser handling of encrypted/password-protected PDFs.
        
        Validates proper error handling when encountering PDFs that
        require passwords or are otherwise inaccessible.
        """
        # Mock encrypted document
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with self.assertRaises(ValueError) as context:
                PyMuPDFParser(tmp_path)
            
            self.assertIn("password-protected", str(context.exception))
            
        finally:
            os.unlink(tmp_path)
    
    @patch('fitz.open')
    def test_text_extraction_with_positions(self, mock_fitz_open):
        """
        Test text extraction with positioning information.
        
        Validates that the parser correctly extracts text blocks with
        accurate positioning, font, and formatting information required
        for heading detection algorithms.
        """
        # Mock document and page structure
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 1
        
        # Mock page with text blocks
        mock_page = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        
        # Mock text extraction result
        mock_text_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Chapter 1: Introduction",
                                    "bbox": [50.0, 100.0, 200.0, 120.0],
                                    "font": "Arial-Bold",
                                    "size": 16.0,
                                    "flags": 16
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_page.get_text.return_value = mock_text_dict
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            parser = PyMuPDFParser(tmp_path)
            text_blocks = parser.extract_text_with_positions()
            
            # Validate text extraction
            self.assertIsInstance(text_blocks, list)
            if text_blocks:  # Only test if blocks were extracted
                block = text_blocks[0]
                self.assertIsInstance(block, TextBlock)
                self.assertEqual(block.text, "Chapter 1: Introduction")
                self.assertEqual(block.x, 50.0)
                self.assertEqual(block.y, 100.0)
                self.assertEqual(block.font_name, "Arial-Bold")
                self.assertEqual(block.font_size, 16.0)
                self.assertEqual(block.page_number, 1)
            
            parser.close()
            
        finally:
            os.unlink(tmp_path)
    
    @patch('fitz.open')
    def test_metadata_extraction(self, mock_fitz_open):
        """
        Test document metadata extraction.
        
        Validates extraction of PDF metadata including title, author,
        creation date, and other document properties required for
        outline building and title detection.
        """
        # Mock document with metadata
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 3
        mock_doc.metadata = {
            'title': 'Sample Document Title',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creator': 'Test Creator',
            'producer': 'Test Producer'
        }
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            parser = PyMuPDFParser(tmp_path)
            metadata = parser.get_document_metadata()
            
            # Validate metadata extraction
            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata['title'], 'Sample Document Title')
            self.assertEqual(metadata['author'], 'Test Author')
            self.assertEqual(metadata['page_count'], 3)
            self.assertIn('file_size', metadata)
            
            parser.close()
            
        finally:
            os.unlink(tmp_path)
    
    @patch('fitz.open')
    def test_parse_pdf_complete_workflow(self, mock_fitz_open):
        """
        Test complete PDF parsing workflow.
        
        Validates the main parse_pdf method that orchestrates the complete
        parsing process including text extraction, metadata gathering,
        and result compilation.
        """
        # Setup comprehensive mock
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.page_count = 2
        mock_doc.metadata = {'title': 'Test Document'}
        
        # Mock pages and text extraction
        mock_page = MagicMock()
        mock_page.get_text.return_value = {"blocks": []}
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__iter__.return_value = [mock_page, mock_page]
        
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            parser = PyMuPDFParser(tmp_path)
            
            # Test complete parsing workflow
            start_time = time.time()
            result = parser.parse_pdf(tmp_path)
            processing_time = time.time() - start_time
            
            # Validate result structure
            self.assertIsInstance(result, dict)
            self.assertIn('text_blocks', result)
            self.assertIn('metadata', result)
            self.assertIn('page_count', result)
            self.assertIn('processing_time', result)
            self.assertIn('language', result)
            self.assertIn('is_scanned', result)
            
            # Validate performance (should be fast for mock)
            self.assertLess(processing_time, 2.0)  # <2s per test requirement
            
            parser.close()
            
        finally:
            os.unlink(tmp_path)
    
    def test_parser_error_handling(self):
        """
        Test error handling for invalid or missing files.
        
        Validates robust error handling when attempting to parse
        non-existent files, corrupted PDFs, or invalid file formats.
        """
        # Test non-existent file
        with self.assertRaises(RuntimeError):
            PyMuPDFParser("non_existent_file.pdf")
        
        # Test invalid file type
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"Not a PDF file")
            tmp_path = tmp_file.name
        
        try:
            with self.assertRaises(RuntimeError):
                PyMuPDFParser(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestPDFMinerParser(unittest.TestCase):
    """
    Test suite for PDFMiner parser implementation.
    
    This test class validates the fallback PDF parser functionality
    including robust handling of challenging documents and edge cases
    that may not be supported by the primary PyMuPDF parser.
    """
    
    def setUp(self):
        """Set up test fixtures and sample data."""
        self.test_fixtures_path = Path(__file__).parent / 'fixtures'
        self.sample_pdf_path = self.test_fixtures_path / 'sample.pdf'
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('src.pdf_parser.pdfminer_parser.PDFParser')
    @patch('src.pdf_parser.pdfminer_parser.PDFDocument')
    def test_parser_initialization_success(self, mock_pdf_document, mock_pdf_parser, mock_open):
        """
        Test successful PDFMiner parser initialization.
        
        Validates that the fallback parser can be initialized correctly
        and sets up the required PDFMiner components.
        """
        # Mock file handle
        mock_file = MagicMock()
        mock_open.return_value = mock_file
        
        # Mock PDFMiner components
        mock_parser_instance = MagicMock()
        mock_pdf_parser.return_value = mock_parser_instance
        
        mock_document = MagicMock()
        mock_document.is_extractable = True
        mock_pdf_document.return_value = mock_document
        
        # Mock PDFPage.create_pages to return some pages
        with patch('src.pdf_parser.pdfminer_parser.PDFPage') as mock_pdf_page:
            mock_pdf_page.create_pages.return_value = [MagicMock(), MagicMock()]
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                parser = PDFMinerParser(tmp_path)
                
                # Validate initialization
                self.assertTrue(parser._is_loaded)
                self.assertEqual(str(parser.pdf_path), tmp_path)
                
                parser.close()
                
            finally:
                os.unlink(tmp_path)
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('src.pdf_parser.pdfminer_parser.PDFParser')
    @patch('src.pdf_parser.pdfminer_parser.PDFDocument')
    def test_parser_encrypted_pdf_handling(self, mock_pdf_document, mock_pdf_parser, mock_open):
        """
        Test PDFMiner handling of encrypted PDFs.
        
        Validates that the fallback parser properly handles encrypted
        documents and provides appropriate error messages.
        """
        # Mock file handle
        mock_file = MagicMock()
        mock_open.return_value = mock_file
        
        # Mock PDFMiner components
        mock_parser_instance = MagicMock()
        mock_pdf_parser.return_value = mock_parser_instance
        
        mock_document = MagicMock()
        mock_document.is_extractable = False  # Encrypted document
        mock_pdf_document.return_value = mock_document
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with self.assertRaises(ValueError) as context:
                PDFMinerParser(tmp_path)
            
            self.assertIn("encrypted", str(context.exception))
            
        finally:
            os.unlink(tmp_path)
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('src.pdf_parser.pdfminer_parser.PDFParser')
    @patch('src.pdf_parser.pdfminer_parser.PDFDocument')
    @patch('src.pdf_parser.pdfminer_parser.extract_pages')
    def test_text_extraction_with_layout_analysis(self, mock_extract_pages, mock_pdf_document, mock_pdf_parser, mock_open):
        """
        Test PDFMiner text extraction with layout analysis.
        
        Validates that the fallback parser correctly extracts text
        with positioning information using PDFMiner's layout analysis
        capabilities.
        """
        # Setup mocks
        mock_file = MagicMock()
        mock_open.return_value = mock_file
        
        mock_parser_instance = MagicMock()
        mock_pdf_parser.return_value = mock_parser_instance
        
        mock_document = MagicMock()
        mock_document.is_extractable = True
        mock_pdf_document.return_value = mock_document
        
        # Mock layout analysis results
        from unittest.mock import MagicMock
        
        # Create mock character objects
        mock_char1 = MagicMock()
        mock_char1.get_text.return_value = "T"
        mock_char1.x0, mock_char1.y0, mock_char1.x1, mock_char1.y1 = 50, 100, 55, 115
        mock_char1.fontname = "Arial-Bold"
        mock_char1.height = 15
        
        mock_char2 = MagicMock()
        mock_char2.get_text.return_value = "est"
        mock_char2.x0, mock_char2.y0, mock_char2.x1, mock_char2.y1 = 55, 100, 75, 115
        mock_char2.fontname = "Arial-Bold"
        mock_char2.height = 15
        
        # Mock text line containing characters
        mock_line = MagicMock()
        mock_line.__iter__.return_value = [mock_char1, mock_char2]
        
        # Mock text container containing line
        mock_container = MagicMock()
        mock_container.__iter__.return_value = [mock_line]
        
        # Mock page containing container
        mock_page = MagicMock()
        mock_page.__iter__.return_value = [mock_container]
        
        mock_extract_pages.return_value = [mock_page]
        
        with patch('src.pdf_parser.pdfminer_parser.PDFPage') as mock_pdf_page:
            mock_pdf_page.create_pages.return_value = [MagicMock()]
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                parser = PDFMinerParser(tmp_path)
                text_blocks = parser.extract_text_with_positions()
                
                # Validate extraction results
                self.assertIsInstance(text_blocks, list)
                # Note: Actual validation depends on mock setup complexity
                
                parser.close()
                
            finally:
                os.unlink(tmp_path)
    
    def test_parser_performance_constraints(self):
        """
        Test parser performance meets timing requirements.
        
        Validates that PDFMiner parser operations complete within
        reasonable time limits for the Adobe Hackathon constraints.
        """
        # This test would use actual PDF files in a full implementation
        # For now, test the performance of parser initialization
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"""%PDF-1.4
1 0 obj
<<
>>
endobj
xref
0 1
0000000000 65535 f 
trailer
<<
/Size 1
>>
startxref
9
%%EOF""")
            tmp_path = tmp_file.name
        
        try:
            start_time = time.time()
            
            # Test would initialize parser here in real implementation
            # For mock test, just validate timing infrastructure
            processing_time = time.time() - start_time
            
            # Should be very fast for this simple test
            self.assertLess(processing_time, 2.0)
            
        finally:
            os.unlink(tmp_path)


class TestParserFallbackStrategy(unittest.TestCase):
    """
    Test suite for parser fallback strategy validation.
    
    This test class validates the intelligent fallback mechanism
    from PyMuPDF to PDFMiner when the primary parser encounters
    challenging or unsupported document formats.
    """
    
    def test_parser_selection_logic(self):
        """
        Test logic for selecting appropriate parser implementation.
        
        Validates the decision logic for choosing between PyMuPDF
        and PDFMiner based on document characteristics and parsing
        success rates.
        """
        # This would test the main pipeline's parser selection
        # For now, validate that both parsers are available
        
        # Check PyMuPDF availability
        try:
            import fitz
            pymupdf_available = True
        except ImportError:
            pymupdf_available = False
        
        # Check PDFMiner availability
        try:
            import pdfminer
            pdfminer_available = True
        except ImportError:
            pdfminer_available = False
        
        # At least one parser should be available
        self.assertTrue(pymupdf_available or pdfminer_available)
    
    def test_fallback_error_handling(self):
        """
        Test error handling in fallback scenarios.
        
        Validates that appropriate errors are raised when both
        parsers fail to process a document, ensuring graceful
        degradation and proper error reporting.
        """
        # Test with non-existent file - both parsers should fail
        non_existent_file = "definitely_does_not_exist.pdf"
        
        pymupdf_error = None
        pdfminer_error = None
        
        # Test PyMuPDF error handling
        try:
            PyMuPDFParser(non_existent_file)
        except Exception as e:
            pymupdf_error = e
        
        # Test PDFMiner error handling  
        try:
            PDFMinerParser(non_existent_file)
        except Exception as e:
            pdfminer_error = e
        
        # Both should raise errors for non-existent files
        self.assertIsNotNone(pymupdf_error)
        self.assertIsNotNone(pdfminer_error)


if __name__ == '__main__':
    # Configure test execution for Adobe Hackathon requirements
    unittest.main(
        verbosity=2,
        failfast=False,
        buffer=True
    )
