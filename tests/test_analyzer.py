"""
Unit tests for ClauseWiseAnalyzer
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from clausewise_implementation import ClauseWiseAnalyzer
except ImportError as e:
    print(f"Warning: Could not import ClauseWiseAnalyzer: {e}")
    ClauseWiseAnalyzer = None


class TestClauseWiseAnalyzer(unittest.TestCase):
    """Test cases for ClauseWiseAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        if ClauseWiseAnalyzer is None:
            self.skipTest("ClauseWiseAnalyzer not available")

        self.analyzer = ClauseWiseAnalyzer()
        self.sample_text = """
        This Non-Disclosure Agreement is entered into between
        TechCorp Inc. and John Smith. The receiving party agrees
        to maintain confidentiality of all disclosed information.
        """

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.max_file_size, 10 * 1024 * 1024)
        self.assertIn('.pdf', self.analyzer.supported_formats)
        self.assertIn('.docx', self.analyzer.supported_formats)
        self.assertIn('.txt', self.analyzer.supported_formats)

    def test_extract_named_entities(self):
        """Test named entity extraction."""
        entities = self.analyzer.extract_named_entities(self.sample_text)

        self.assertIsInstance(entities, list)

        # Check entity structure if entities found
        if entities:
            entity = entities[0]
            self.assertIn('text', entity)
            self.assertIn('label', entity)

    def test_extract_clauses(self):
        """Test clause extraction."""
        clauses = self.analyzer.extract_clauses(self.sample_text)

        self.assertIsInstance(clauses, list)

        # Check clause structure if clauses found
        if clauses:
            clause = clauses[0]
            self.assertIn('id', clause)
            self.assertIn('title', clause)
            self.assertIn('text', clause)
            self.assertIn('category', clause)
            self.assertIn('importance', clause)

    def test_classify_document(self):
        """Test document classification."""
        classification = self.analyzer.classify_document(self.sample_text)

        self.assertIsInstance(classification, dict)
        self.assertIn('predictions', classification)
        self.assertIn('keyIndicators', classification)

        # Check predictions structure
        predictions = classification['predictions']
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        if predictions:
            prediction = predictions[0]
            self.assertIn('type', prediction)
            self.assertIn('confidence', prediction)

    def test_simplify_clause(self):
        """Test clause simplification."""
        clause = "The party of the first part shall maintain confidentiality."
        simplified = self.analyzer.simplify_clause(clause)

        self.assertIsInstance(simplified, str)
        self.assertGreater(len(simplified), 0)


if __name__ == '__main__':
    unittest.main()
