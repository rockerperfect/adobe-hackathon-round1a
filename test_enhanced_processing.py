"""
Test script for enhanced OCR capabilities and generalized PDF processing.

This script validates the complete OCR integration and tests the system's
ability to handle various types of PDFs including image-based documents.
"""

import logging
import time
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_parser.pymupdf_parser import PyMuPDFParser
from src.pdf_parser.pdfminer_parser import PDFMinerParser
from src.outline_extractor.heading_detector import HeadingDetector
from src.outline_extractor.outline_builder import OutlineBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_ocr_capabilities.log')
    ]
)


def test_enhanced_pdf_processing():
    """Test enhanced PDF processing with OCR capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced PDF processing tests")
    
    # Test files
    test_files = [
        "input/file01.pdf",
        "input/file02.pdf", 
        "input/file03.pdf",
        "input/file04.pdf",
        "input/file05.pdf"
    ]
    
    # Initialize components
    heading_detector = HeadingDetector()
    outline_builder = OutlineBuilder()
    
    results = {}
    
    for pdf_file in test_files:
        if not Path(pdf_file).exists():
            logger.warning(f"Test file not found: {pdf_file}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing enhanced processing for: {pdf_file}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Test PyMuPDF parser with OCR
            logger.info("Testing PyMuPDF parser with OCR integration...")
            pymupdf_parser = PyMuPDFParser(pdf_file)
            parsing_result = pymupdf_parser.parse_pdf(pdf_file)
            
            logger.info(f"PyMuPDF Results:")
            logger.info(f"  - Text blocks extracted: {len(parsing_result['text_blocks'])}")
            logger.info(f"  - Is scanned document: {parsing_result['is_scanned']}")
            logger.info(f"  - Processing time: {parsing_result['processing_time']:.2f}s")
            logger.info(f"  - Language detected: {parsing_result.get('language', 'unknown')}")
            
            # Check for OCR-enhanced blocks
            ocr_blocks = []
            try:
                ocr_blocks = [
                    block for block in parsing_result['text_blocks'] 
                    if hasattr(block, 'metadata') and block.metadata and 
                    block.metadata.get('source') == 'ocr'
                ]
            except Exception:
                ocr_blocks = []
            
            logger.info(f"  - OCR-enhanced blocks: {len(ocr_blocks)}")
            
            if ocr_blocks:
                avg_confidence = sum(
                    block.metadata.get('confidence', 0) for block in ocr_blocks
                ) / len(ocr_blocks)
                logger.info(f"  - Average OCR confidence: {avg_confidence:.1f}%")
            
            # Test heading detection with mixed content
            logger.info("\nTesting enhanced heading detection...")
            detected_headings = heading_detector.detect_headings(parsing_result['text_blocks'])
            
            logger.info(f"Heading Detection Results:")
            logger.info(f"  - Total headings detected: {len(detected_headings)}")
            
            # Analyze heading sources
            native_headings = 0
            ocr_headings = 0
            
            try:
                for h in detected_headings:
                    if hasattr(h, 'source_blocks') and h.source_blocks:
                        has_ocr = any(
                            hasattr(block, 'metadata') and block.metadata and 
                            block.metadata.get('source') == 'ocr'
                            for block in h.source_blocks
                        )
                        if has_ocr:
                            ocr_headings += 1
                        else:
                            native_headings += 1
                    else:
                        native_headings += 1  # Default assumption
            except Exception:
                native_headings = len(detected_headings)
                ocr_headings = 0
            
            logger.info(f"  - Native text headings: {native_headings}")
            logger.info(f"  - OCR-derived headings: {ocr_headings}")
            
            # Display detected headings
            logger.info("\nDetected Headings:")
            for i, heading in enumerate(detected_headings[:10]):  # Show first 10
                try:
                    source = "Native"  # Default
                    if hasattr(heading, 'source_blocks') and heading.source_blocks:
                        has_ocr = any(
                            hasattr(block, 'metadata') and block.metadata and 
                            block.metadata.get('source') == 'ocr'
                            for block in heading.source_blocks
                        )
                        source = "OCR" if has_ocr else "Native"
                    logger.info(f"  {i+1}. [{source}] {heading.level}: {heading.text[:80]}")
                except Exception:
                    logger.info(f"  {i+1}. [Unknown] {heading.level}: {heading.text[:80]}")
            
            # Test outline building
            logger.info("\nTesting outline construction...")
            outline_structure = outline_builder.build_outline(
                detected_headings=detected_headings,
                document_metadata=parsing_result['metadata'],
                document_path=pdf_file,
                text_blocks=parsing_result['text_blocks']
            )
            
            logger.info(f"Outline Construction Results:")
            logger.info(f"  - Document title: {outline_structure.get('title', 'Unknown')}")
            logger.info(f"  - Outline entries: {len(outline_structure.get('outline', []))}")
            
            # Display outline structure
            logger.info("\nOutline Structure:")
            for i, entry in enumerate(outline_structure.get('outline', [])[:10]):
                logger.info(f"  {entry['level']}: {entry['text'][:60]} (page {entry['page']})")
            
            total_time = time.time() - start_time
            
            results[pdf_file] = {
                'processing_time': total_time,
                'text_blocks': len(parsing_result['text_blocks']),
                'ocr_blocks': len(ocr_blocks),
                'is_scanned': parsing_result['is_scanned'],
                'headings_detected': len(detected_headings),
                'outline_entries': len(outline_structure.get('outline', [])),
                'title_extracted': outline_structure.get('title', 'Unknown'),
                'success': True
            }
            
            logger.info(f"\nTotal processing time: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            results[pdf_file] = {
                'processing_time': time.time() - start_time,
                'error': str(e),
                'success': False
            }
        
        finally:
            # Cleanup
            try:
                pymupdf_parser.cleanup()
            except:
                pass
    
    # Summary report
    logger.info(f"\n{'='*80}")
    logger.info("ENHANCED PDF PROCESSING TEST SUMMARY")
    logger.info(f"{'='*80}")
    
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    logger.info(f"Tests completed: {total_tests}")
    logger.info(f"Successful: {successful_tests}")
    logger.info(f"Failed: {total_tests - successful_tests}")
    
    if successful_tests > 0:
        avg_time = sum(r['processing_time'] for r in results.values() if r.get('success')) / successful_tests
        total_blocks = sum(r.get('text_blocks', 0) for r in results.values() if r.get('success'))
        total_ocr_blocks = sum(r.get('ocr_blocks', 0) for r in results.values() if r.get('success'))
        total_headings = sum(r.get('headings_detected', 0) for r in results.values() if r.get('success'))
        
        logger.info(f"\nPerformance Summary:")
        logger.info(f"  - Average processing time: {avg_time:.2f}s")
        logger.info(f"  - Total text blocks processed: {total_blocks}")
        logger.info(f"  - Total OCR blocks: {total_ocr_blocks}")
        logger.info(f"  - OCR enhancement ratio: {(total_ocr_blocks/total_blocks)*100:.1f}%" if total_blocks > 0 else "  - OCR enhancement ratio: 0.0%")
        logger.info(f"  - Total headings detected: {total_headings}")
        
        scanned_docs = sum(1 for r in results.values() if r.get('is_scanned', False))
        logger.info(f"  - Scanned documents detected: {scanned_docs}")
    
    logger.info(f"\nDetailed Results:")
    for pdf_file, result in results.items():
        if result.get('success'):
            logger.info(f"  {Path(pdf_file).name}:")
            logger.info(f"    - Time: {result['processing_time']:.2f}s")
            logger.info(f"    - Text blocks: {result['text_blocks']}")
            logger.info(f"    - OCR blocks: {result['ocr_blocks']}")
            logger.info(f"    - Scanned: {result['is_scanned']}")
            logger.info(f"    - Headings: {result['headings_detected']}")
            logger.info(f"    - Title: {result['title_extracted'][:50]}")
        else:
            logger.info(f"  {Path(pdf_file).name}: FAILED - {result.get('error', 'Unknown error')}")
    
    return results


def test_ocr_specific_features():
    """Test OCR-specific functionality."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("TESTING OCR-SPECIFIC FEATURES")
    logger.info("="*60)
    
    try:
        from src.utils.ocr_processor import PDFOCRProcessor, ImagePreprocessor, OCREngine
        
        logger.info("✓ OCR processor modules imported successfully")
        
        # Test image preprocessor
        preprocessor = ImagePreprocessor()
        logger.info("✓ Image preprocessor initialized")
        
        # Test OCR engine
        ocr_engine = OCREngine()
        logger.info("✓ OCR engine initialized")
        
        # Test PDF OCR processor
        pdf_ocr = PDFOCRProcessor()
        logger.info("✓ PDF OCR processor initialized")
        
        logger.info("\n✓ All OCR components available and functional")
        
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {str(e)}")
        logger.info("This is expected if pytesseract/opencv not installed")
        logger.info("System will fall back to standard text extraction")
        
    except Exception as e:
        logger.error(f"Error testing OCR features: {str(e)}")


if __name__ == "__main__":
    print("Enhanced PDF Processing Test Suite")
    print("=" * 50)
    
    # Test OCR availability
    test_ocr_specific_features()
    
    # Test enhanced processing
    results = test_enhanced_pdf_processing()
    
    print(f"\nTest completed. Check 'test_ocr_capabilities.log' for detailed results.")
