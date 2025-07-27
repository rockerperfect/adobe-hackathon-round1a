"""
Adaptive Document Analyzer for Robust PDF Processing

This module provides document-level statistical analysis to compute adaptive thresholds
for title extraction and heading detection, ensuring consistent results across all PDFs
without hardcoding specific patterns or content.

Key Features:
- Adaptive font size clustering for hierarchical level assignment
- Spatial layout analysis for compound title reconstruction  
- Statistical distribution analysis for robust threshold computation
- Generic content filtering based on structural properties
"""

import logging
import math
import statistics
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter

from src.pdf_parser.base_parser import TextBlock


@dataclass
class DocumentStatistics:
    """Statistical analysis of document properties."""
    font_size_distribution: Dict[float, int]
    font_size_clusters: List[Tuple[float, float]]  # (center, threshold)
    line_spacing_stats: Dict[str, float]
    x_margin_stats: Dict[str, float]
    y_position_stats: Dict[str, float]
    content_density: float
    first_page_stats: Dict[str, Any]
    
    
@dataclass
class AdaptiveThresholds:
    """Adaptive thresholds computed from document analysis."""
    title_font_threshold: float
    heading_font_thresholds: Dict[str, float]  # H1, H2, H3
    spatial_grouping_tolerance: float
    line_spacing_tolerance: float
    noise_filter_thresholds: Dict[str, float]


class AdaptiveDocumentAnalyzer:
    """
    Analyzes document structure to compute adaptive thresholds and parameters
    for robust title extraction and heading detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_document(self, text_blocks: List[TextBlock]) -> Tuple[DocumentStatistics, AdaptiveThresholds]:
        """
        Analyze document structure and compute adaptive thresholds.
        
        Args:
            text_blocks: All text blocks from the document
            
        Returns:
            Tuple of (document statistics, adaptive thresholds)
        """
        if not text_blocks:
            return self._get_default_stats_and_thresholds()
            
        stats = self._compute_document_statistics(text_blocks)
        thresholds = self._compute_adaptive_thresholds(stats, text_blocks)
        
        return stats, thresholds
    
    def _compute_document_statistics(self, text_blocks: List[TextBlock]) -> DocumentStatistics:
        """Compute comprehensive document statistics."""
        
        # Font size analysis
        font_sizes = [block.font_size for block in text_blocks]
        font_size_counts = Counter(font_sizes)
        
        # Spatial analysis
        line_heights = []
        x_positions = [block.x for block in text_blocks]
        y_positions = [block.y for block in text_blocks]
        
        # Calculate line spacing
        sorted_by_y = sorted(text_blocks, key=lambda b: (b.page_number, -b.y))
        for i in range(1, len(sorted_by_y)):
            prev_block = sorted_by_y[i-1]
            curr_block = sorted_by_y[i]
            if prev_block.page_number == curr_block.page_number:
                spacing = abs(prev_block.y - curr_block.y)
                if spacing > 0:  # Avoid division by zero
                    line_heights.append(spacing)
        
        # First page analysis
        first_page_blocks = [b for b in text_blocks if self._get_page_number(b) == 1]
        first_page_fonts = [b.font_size for b in first_page_blocks] if first_page_blocks else font_sizes
        
        # Font clustering using statistical methods
        font_clusters = self._cluster_font_sizes(font_size_counts)
        
        return DocumentStatistics(
            font_size_distribution=dict(font_size_counts),
            font_size_clusters=font_clusters,
            line_spacing_stats={
                'mean': statistics.mean(line_heights) if line_heights else 14.0,
                'median': statistics.median(line_heights) if line_heights else 14.0,
                'std': statistics.stdev(line_heights) if len(line_heights) > 1 else 2.0
            },
            x_margin_stats={
                'mean': statistics.mean(x_positions),
                'median': statistics.median(x_positions),
                'std': statistics.stdev(x_positions) if len(x_positions) > 1 else 10.0
            },
            y_position_stats={
                'mean': statistics.mean(y_positions),
                'median': statistics.median(y_positions),
                'std': statistics.stdev(y_positions) if len(y_positions) > 1 else 50.0
            },
            content_density=len(text_blocks) / max(1, len(set((b.page_number for b in text_blocks)))),
            first_page_stats={
                'max_font': max(first_page_fonts),
                'font_variety': len(set(first_page_fonts)),
                'block_count': len(first_page_blocks)
            }
        )
    
    def _cluster_font_sizes(self, font_size_counts: Counter) -> List[Tuple[float, float]]:
        """
        Cluster font sizes using statistical analysis to identify natural breaks
        for hierarchical level assignment.
        """
        if not font_size_counts:
            return [(12.0, 1.0)]
            
        sizes = list(font_size_counts.keys())
        sizes.sort(reverse=True)  # Largest first
        
        if len(sizes) <= 2:
            return [(size, 0.5) for size in sizes]
        
        # Find natural breaks using gaps in font sizes
        gaps = []
        for i in range(len(sizes) - 1):
            gap = sizes[i] - sizes[i + 1]
            gaps.append((gap, i))
        
        # Sort gaps by size
        gaps.sort(reverse=True)
        
        # Take the largest gaps as cluster boundaries
        cluster_boundaries = []
        for gap_size, position in gaps[:min(3, len(gaps))]:
            if gap_size > 1.0:  # Only significant gaps
                cluster_boundaries.append(position)
        
        cluster_boundaries.sort()
        
        # Create clusters
        clusters = []
        start_idx = 0
        for boundary in cluster_boundaries:
            if boundary > start_idx:
                cluster_sizes = sizes[start_idx:boundary + 1]
                center = statistics.mean(cluster_sizes)
                threshold = min(cluster_sizes) - 0.5
                clusters.append((center, threshold))
                start_idx = boundary + 1
        
        # Add final cluster
        if start_idx < len(sizes):
            cluster_sizes = sizes[start_idx:]
            center = statistics.mean(cluster_sizes)
            threshold = min(cluster_sizes) - 0.5
            clusters.append((center, threshold))
        
        return clusters if clusters else [(sizes[0], sizes[0] - 0.5)]
    
    def _compute_adaptive_thresholds(self, stats: DocumentStatistics, text_blocks: List[TextBlock]) -> AdaptiveThresholds:
        """Compute adaptive thresholds based on document statistics."""
        
        # Title font threshold based on first page analysis
        max_first_page_font = stats.first_page_stats['max_font']
        title_font_threshold = max_first_page_font * 0.85  # 85% of max font
        
        # Heading font thresholds based on font clusters
        heading_thresholds = {}
        if len(stats.font_size_clusters) >= 1:
            heading_thresholds['H1'] = stats.font_size_clusters[0][1]
        if len(stats.font_size_clusters) >= 2:
            heading_thresholds['H2'] = stats.font_size_clusters[1][1]
        if len(stats.font_size_clusters) >= 3:
            heading_thresholds['H3'] = stats.font_size_clusters[2][1]
        else:
            # Fall back to statistical percentiles
            all_fonts = [b.font_size for b in text_blocks]
            heading_thresholds.setdefault('H1', statistics.quantiles(all_fonts, n=4)[2])  # 75th percentile
            heading_thresholds.setdefault('H2', statistics.median(all_fonts))
            heading_thresholds.setdefault('H3', statistics.quantiles(all_fonts, n=4)[0])  # 25th percentile
        
        # Spatial grouping tolerance based on line spacing
        line_spacing_mean = stats.line_spacing_stats['mean']
        spatial_tolerance = line_spacing_mean * 1.5  # 1.5x average line spacing
        
        # Noise filtering thresholds
        noise_thresholds = {
            'min_text_length': 3,
            'max_special_char_ratio': 0.6,
            'min_alpha_ratio': 0.3,
            'max_digit_ratio': 0.8
        }
        
        return AdaptiveThresholds(
            title_font_threshold=title_font_threshold,
            heading_font_thresholds=heading_thresholds,
            spatial_grouping_tolerance=spatial_tolerance,
            line_spacing_tolerance=line_spacing_mean * 0.5,
            noise_filter_thresholds=noise_thresholds
        )
    
    def _get_default_stats_and_thresholds(self) -> Tuple[DocumentStatistics, AdaptiveThresholds]:
        """Return default statistics and thresholds for empty documents."""
        stats = DocumentStatistics(
            font_size_distribution={12.0: 1},
            font_size_clusters=[(12.0, 11.5)],
            line_spacing_stats={'mean': 14.0, 'median': 14.0, 'std': 2.0},
            x_margin_stats={'mean': 72.0, 'median': 72.0, 'std': 10.0},
            y_position_stats={'mean': 400.0, 'median': 400.0, 'std': 50.0},
            content_density=1.0,
            first_page_stats={'max_font': 12.0, 'font_variety': 1, 'block_count': 0}
        )
        
        thresholds = AdaptiveThresholds(
            title_font_threshold=12.0,
            heading_font_thresholds={'H1': 12.0, 'H2': 11.0, 'H3': 10.0},
            spatial_grouping_tolerance=14.0,
            line_spacing_tolerance=7.0,
            noise_filter_thresholds={
                'min_text_length': 3,
                'max_special_char_ratio': 0.6,
                'min_alpha_ratio': 0.3,
                'max_digit_ratio': 0.8
            }
        )
        
        return stats, thresholds
    
    def _get_page_number(self, block: TextBlock) -> int:
        """Extract page number from text block."""
        return getattr(block, 'page_number', getattr(block, 'page', 1))
    
    def analyze_mixed_content_document(self, text_blocks: List[TextBlock]) -> Tuple[DocumentStatistics, AdaptiveThresholds]:
        """
        Enhanced analysis for documents with mixed OCR and native text content.
        
        This method provides specialized analysis for documents that contain both
        native PDF text and OCR-derived text, adjusting thresholds to handle
        the different characteristics of each text source.
        
        Args:
            text_blocks: List of text blocks from both native PDF parsing and OCR
            
        Returns:
            Tuple of enhanced document statistics and adaptive thresholds
        """
        if not text_blocks:
            return self._get_default_stats_and_thresholds()
        
        # Separate OCR and native text blocks
        ocr_blocks = [block for block in text_blocks if self._is_ocr_block(block)]
        native_blocks = [block for block in text_blocks if not self._is_ocr_block(block)]
        
        self.logger.info(f"Analyzing document with {len(native_blocks)} native blocks and {len(ocr_blocks)} OCR blocks")
        
        # Analyze each type separately
        ocr_stats = self._analyze_ocr_blocks(ocr_blocks) if ocr_blocks else None
        native_stats = self.analyze_document(native_blocks) if native_blocks else None
        
        # Merge statistics intelligently
        merged_stats, merged_thresholds = self._merge_mixed_content_analysis(
            native_stats, ocr_stats, text_blocks
        )
        
        return merged_stats, merged_thresholds
    
    def _is_ocr_block(self, block: TextBlock) -> bool:
        """Determine if a text block originated from OCR processing."""
        # Check for OCR indicators
        if hasattr(block, 'metadata') and block.metadata:
            return block.metadata.get('source') == 'ocr'
        
        # Check font name for OCR indicators
        font_name = getattr(block, 'font_name', '')
        if 'OCR' in font_name or 'ocr' in font_name:
            return True
        
        # Check for OCR-specific confidence scores
        if hasattr(block, 'confidence'):
            return True
        
        return False
    
    def _analyze_ocr_blocks(self, ocr_blocks: List[TextBlock]) -> Tuple[DocumentStatistics, AdaptiveThresholds]:
        """
        Specialized analysis for OCR-derived text blocks.
        
        OCR text requires different statistical analysis due to:
        - Variable font size estimation from image analysis
        - Potential spatial positioning errors
        - Different confidence characteristics
        """
        if not ocr_blocks:
            return self._get_default_stats_and_thresholds()
        
        # Font size analysis with OCR confidence weighting
        font_sizes = []
        confidence_weights = []
        
        for block in ocr_blocks:
            font_size = getattr(block, 'font_size', 12.0)
            confidence = getattr(block, 'confidence', 80.0)
            
            # Weight font sizes by OCR confidence
            weight = confidence / 100.0
            font_sizes.append(font_size)
            confidence_weights.append(weight)
        
        # Weighted font size distribution
        font_distribution = defaultdict(float)
        for size, weight in zip(font_sizes, confidence_weights):
            font_distribution[size] += weight
        
        # Enhanced spatial analysis for OCR positioning
        spatial_stats = self._analyze_ocr_spatial_layout(ocr_blocks)
        
        # Confidence-based clustering
        font_clusters = self._compute_confidence_weighted_clusters(font_sizes, confidence_weights)
        
        stats = DocumentStatistics(
            font_size_distribution=dict(font_distribution),
            font_size_clusters=font_clusters,
            line_spacing_stats=spatial_stats['line_spacing'],
            x_margin_stats=spatial_stats['x_margins'],
            y_position_stats=spatial_stats['y_positions'],
            content_density=spatial_stats['density'],
            first_page_stats=self._analyze_first_page_ocr(ocr_blocks)
        )
        
        # OCR-specific thresholds
        thresholds = self._compute_ocr_adaptive_thresholds(stats, ocr_blocks)
        
        return stats, thresholds
    
    def _analyze_ocr_spatial_layout(self, ocr_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze spatial layout characteristics specific to OCR text."""
        if not ocr_blocks:
            return self._get_default_spatial_stats()
        
        # Collect spatial data with confidence weighting
        line_spacings = []
        x_margins = []
        y_positions = []
        confidences = []
        
        for block in ocr_blocks:
            confidence = getattr(block, 'confidence', 80.0) / 100.0
            confidences.append(confidence)
            
            y_positions.append(block.bbox[1] * confidence)  # Weighted by confidence
            x_margins.append(block.bbox[0] * confidence)
            
            # Estimate line spacing from block height
            block_height = block.bbox[3] - block.bbox[1]
            line_spacings.append(block_height * confidence)
        
        # Compute weighted statistics
        total_weight = sum(confidences)
        
        return {
            'line_spacing': {
                'mean': sum(line_spacings) / total_weight if total_weight > 0 else 14.0,
                'median': statistics.median(line_spacings) if line_spacings else 14.0,
                'std': statistics.stdev(line_spacings) if len(line_spacings) > 1 else 2.0
            },
            'x_margins': {
                'mean': sum(x_margins) / total_weight if total_weight > 0 else 72.0,
                'median': statistics.median(x_margins) if x_margins else 72.0,
                'std': statistics.stdev(x_margins) if len(x_margins) > 1 else 10.0
            },
            'y_positions': {
                'mean': sum(y_positions) / total_weight if total_weight > 0 else 400.0,
                'median': statistics.median(y_positions) if y_positions else 400.0,
                'std': statistics.stdev(y_positions) if len(y_positions) > 1 else 50.0
            },
            'density': len(ocr_blocks) / max(1, len(set(self._get_page_number(block) for block in ocr_blocks)))
        }
    
    def _compute_confidence_weighted_clusters(self, font_sizes: List[float], 
                                            confidence_weights: List[float]) -> List[Tuple[float, float]]:
        """Compute font size clusters weighted by OCR confidence scores."""
        if not font_sizes:
            return [(12.0, 11.5)]
        
        # Create weighted font size pairs
        weighted_sizes = []
        for size, weight in zip(font_sizes, confidence_weights):
            # Replicate sizes based on confidence weight
            count = max(1, int(weight * 10))  # Scale confidence to count
            weighted_sizes.extend([size] * count)
        
        # Cluster weighted sizes
        if len(set(weighted_sizes)) <= 3:
            # Few unique sizes - create simple clusters
            unique_sizes = sorted(set(weighted_sizes))
            clusters = []
            for size in unique_sizes:
                threshold = size * 0.95  # 5% tolerance
                clusters.append((size, threshold))
            return clusters
        
        # Use quantile-based clustering for more sizes
        sorted_sizes = sorted(weighted_sizes)
        q25, q50, q75 = statistics.quantiles(sorted_sizes, n=4)
        
        return [
            (q75, q75 * 0.9),    # Large fonts (titles/major headings)
            (q50, q50 * 0.9),    # Medium fonts (subheadings)
            (q25, q25 * 0.9)     # Small fonts (body text)
        ]
    
    def _analyze_first_page_ocr(self, ocr_blocks: List[TextBlock]) -> Dict[str, Any]:
        """Analyze first page characteristics for OCR blocks."""
        first_page_blocks = [block for block in ocr_blocks if self._get_page_number(block) == 1]
        
        if not first_page_blocks:
            return {'max_font': 12.0, 'font_variety': 1, 'block_count': 0}
        
        font_sizes = [getattr(block, 'font_size', 12.0) for block in first_page_blocks]
        confidences = [getattr(block, 'confidence', 80.0) for block in first_page_blocks]
        
        # Weight max font by confidence
        weighted_max_font = max(
            size * (conf / 100.0) for size, conf in zip(font_sizes, confidences)
        )
        
        return {
            'max_font': weighted_max_font,
            'font_variety': len(set(font_sizes)),
            'block_count': len(first_page_blocks),
            'avg_confidence': statistics.mean(confidences) if confidences else 80.0
        }
    
    def _compute_ocr_adaptive_thresholds(self, stats: DocumentStatistics, 
                                       ocr_blocks: List[TextBlock]) -> AdaptiveThresholds:
        """Compute adaptive thresholds specifically for OCR content."""
        # Base thresholds on font clusters with OCR adjustments
        clusters = stats.font_size_clusters
        
        if clusters:
            # Use highest confidence cluster for title threshold
            title_threshold = clusters[0][0] * 0.85  # More lenient for OCR
            
            heading_thresholds = {}
            if len(clusters) >= 1:
                heading_thresholds['H1'] = clusters[0][0] * 0.8
            if len(clusters) >= 2:
                heading_thresholds['H2'] = clusters[1][0] * 0.8
            if len(clusters) >= 3:
                heading_thresholds['H3'] = clusters[2][0] * 0.8
        else:
            title_threshold = 12.0
            heading_thresholds = {'H1': 12.0, 'H2': 11.0, 'H3': 10.0}
        
        # More tolerant spatial grouping for OCR positioning errors
        spatial_tolerance = stats.line_spacing_stats['mean'] * 2.0
        
        # OCR-specific noise filtering
        noise_thresholds = {
            'min_text_length': 2,  # More lenient for OCR
            'max_special_char_ratio': 0.7,  # OCR may introduce artifacts
            'min_alpha_ratio': 0.2,  # More lenient character ratio
            'max_digit_ratio': 0.9,  # Allow more digits
            'min_confidence': 50.0   # OCR confidence threshold
        }
        
        return AdaptiveThresholds(
            title_font_threshold=title_threshold,
            heading_font_thresholds=heading_thresholds,
            spatial_grouping_tolerance=spatial_tolerance,
            line_spacing_tolerance=stats.line_spacing_stats['mean'] * 0.8,
            noise_filter_thresholds=noise_thresholds
        )
    
    def _merge_mixed_content_analysis(self, native_analysis: Optional[Tuple], 
                                    ocr_analysis: Optional[Tuple], 
                                    all_blocks: List[TextBlock]) -> Tuple[DocumentStatistics, AdaptiveThresholds]:
        """Merge analysis results from native and OCR content intelligently."""
        if not native_analysis and not ocr_analysis:
            return self._get_default_stats_and_thresholds()
        
        if not native_analysis:
            return ocr_analysis
        
        if not ocr_analysis:
            return native_analysis
        
        native_stats, native_thresholds = native_analysis
        ocr_stats, ocr_thresholds = ocr_analysis
        
        # Count blocks for weighting
        native_count = len([b for b in all_blocks if not self._is_ocr_block(b)])
        ocr_count = len([b for b in all_blocks if self._is_ocr_block(b)])
        total_count = native_count + ocr_count
        
        native_weight = native_count / total_count if total_count > 0 else 0.5
        ocr_weight = ocr_count / total_count if total_count > 0 else 0.5
        
        # Merge font distributions
        merged_font_dist = {}
        for size, count in native_stats.font_size_distribution.items():
            merged_font_dist[size] = count * native_weight
        
        for size, count in ocr_stats.font_size_distribution.items():
            merged_font_dist[size] = merged_font_dist.get(size, 0) + count * ocr_weight
        
        # Merge thresholds with preference for native text accuracy
        title_threshold = (
            native_thresholds.title_font_threshold * 0.7 +
            ocr_thresholds.title_font_threshold * 0.3
        )
        
        merged_heading_thresholds = {}
        for level in ['H1', 'H2', 'H3']:
            native_val = native_thresholds.heading_font_thresholds.get(level, 12.0)
            ocr_val = ocr_thresholds.heading_font_thresholds.get(level, 12.0)
            merged_heading_thresholds[level] = native_val * 0.7 + ocr_val * 0.3
        
        # Create merged statistics
        merged_stats = DocumentStatistics(
            font_size_distribution=merged_font_dist,
            font_size_clusters=native_stats.font_size_clusters + ocr_stats.font_size_clusters,
            line_spacing_stats=self._merge_stats_dict(
                native_stats.line_spacing_stats, ocr_stats.line_spacing_stats, native_weight
            ),
            x_margin_stats=self._merge_stats_dict(
                native_stats.x_margin_stats, ocr_stats.x_margin_stats, native_weight
            ),
            y_position_stats=self._merge_stats_dict(
                native_stats.y_position_stats, ocr_stats.y_position_stats, native_weight
            ),
            content_density=(native_stats.content_density * native_weight + 
                           ocr_stats.content_density * ocr_weight),
            first_page_stats=native_stats.first_page_stats  # Prefer native for first page
        )
        
        # Create merged thresholds
        merged_thresholds = AdaptiveThresholds(
            title_font_threshold=title_threshold,
            heading_font_thresholds=merged_heading_thresholds,
            spatial_grouping_tolerance=max(
                native_thresholds.spatial_grouping_tolerance,
                ocr_thresholds.spatial_grouping_tolerance
            ),  # Use more tolerant value
            line_spacing_tolerance=max(
                native_thresholds.line_spacing_tolerance,
                ocr_thresholds.line_spacing_tolerance
            ),
            noise_filter_thresholds=self._merge_noise_thresholds(
                native_thresholds.noise_filter_thresholds,
                ocr_thresholds.noise_filter_thresholds
            )
        )
        
        return merged_stats, merged_thresholds
    
    def _merge_stats_dict(self, native_dict: Dict[str, float], 
                         ocr_dict: Dict[str, float], native_weight: float) -> Dict[str, float]:
        """Merge statistical dictionaries with weighting."""
        merged = {}
        for key in native_dict:
            native_val = native_dict[key]
            ocr_val = ocr_dict.get(key, native_val)
            merged[key] = native_val * native_weight + ocr_val * (1 - native_weight)
        return merged
    
    def _merge_noise_thresholds(self, native_thresholds: Dict[str, float], 
                              ocr_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Merge noise filtering thresholds, preferring more lenient values for OCR."""
        merged = native_thresholds.copy()
        
        # Use more lenient values from OCR thresholds where appropriate
        if 'min_text_length' in ocr_thresholds:
            merged['min_text_length'] = min(
                native_thresholds.get('min_text_length', 3),
                ocr_thresholds['min_text_length']
            )
        
        if 'max_special_char_ratio' in ocr_thresholds:
            merged['max_special_char_ratio'] = max(
                native_thresholds.get('max_special_char_ratio', 0.6),
                ocr_thresholds['max_special_char_ratio']
            )
        
        # Add OCR-specific thresholds
        if 'min_confidence' in ocr_thresholds:
            merged['min_confidence'] = ocr_thresholds['min_confidence']
        
        return merged
    
    def _get_default_spatial_stats(self) -> Dict[str, Any]:
        """Return default spatial statistics."""
        return {
            'line_spacing': {'mean': 14.0, 'median': 14.0, 'std': 2.0},
            'x_margins': {'mean': 72.0, 'median': 72.0, 'std': 10.0},
            'y_positions': {'mean': 400.0, 'median': 400.0, 'std': 50.0},
            'density': 1.0
        }
