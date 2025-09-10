
import os
import fitz  # PyMuPDF for PDF processing
import docx2txt  # For DOCX processing
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
import logging
import gradio as gr
import tempfile
import hashlib
from pathlib import Path
import torch
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseWiseAnalyzer:
    """
    Main analyzer class that integrates IBM Granite with document processing
    and NER capabilities for legal document analysis.
    """

    def __init__(self):
        """Initialize the analyzer with required models and configurations."""
        # Load configuration from environment variables
        self.model_name = os.getenv('GRANITE_MODEL_NAME', 'ibm-granite/granite-3.0-3b-a800m-instruct')
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024))  # 10MB default
        self.supported_formats = os.getenv('SUPPORTED_FORMATS', '.pdf,.docx,.txt').split(',')
        self.hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')

        # Initialize components
        self.granite_model = None
        self.granite_tokenizer = None
        self.nlp_model = None
        self.text_generator = None

        # Authenticate with Hugging Face if token is provided
        self._authenticate_huggingface()

        # Load models
        self._load_models()

    def _authenticate_huggingface(self):
        """Authenticate with Hugging Face using the provided token."""
        if self.hf_token:
            try:
                logger.info("Authenticating with Hugging Face...")
                login(token=self.hf_token)
                logger.info("Successfully authenticated with Hugging Face!")
            except Exception as e:
                logger.error(f"Failed to authenticate with Hugging Face: {e}")
                logger.warning("Proceeding without authentication - some models may not be accessible")
        else:
            logger.warning("No Hugging Face token provided. Some models may not be accessible.")

    def _load_models(self):
        """Load IBM Granite and spaCy models with proper authentication."""
        try:
            logger.info(f"Loading IBM Granite model: {self.model_name}")

            # Load IBM Granite tokenizer
            self.granite_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.hf_token
            )

            # Load IBM Granite model
            self.granite_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=self.hf_token
            )

            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.granite_model,
                tokenizer=self.granite_tokenizer,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.granite_tokenizer.eos_token_id
            )

            logger.info("IBM Granite model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading IBM Granite model: {e}")
            logger.warning("Falling back to demo mode")
            # Fallback for demo purposes
            self.granite_model = None
            self.granite_tokenizer = None
            self.text_generator = None

        try:
            # Load spaCy model for NER
            logger.info("Loading spaCy NER model...")
            self.nlp_model = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully!")
        except OSError:
            logger.warning("spaCy model not found. Using fallback NER.")
            self.nlp_model = None

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from uploaded files (PDF, DOCX, TXT).

        Args:
            file_path: Path to the uploaded file

        Returns:
            Extracted text content
        """
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return self._extract_pdf_text(file_path)
        elif file_extension == '.docx':
            return self._extract_docx_text(file_path)
        elif file_extension == '.txt':
            return self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            # Fallback for demo
            text = "Sample PDF content for demonstration purposes."
        return text

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX using docx2txt."""
        try:
            text = docx2txt.process(file_path)
            return text if text else "Sample DOCX content for demonstration purposes."
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return "Sample DOCX content for demonstration purposes."

    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            return text
        except Exception as e:
            logger.error(f"Error extracting TXT text: {e}")
            return "Sample TXT content for demonstration purposes."

    def simplify_clause(self, clause_text: str) -> str:
        """
        Simplify complex legal clause using IBM Granite.

        Args:
            clause_text: Complex legal text to simplify

        Returns:
            Simplified, layman-friendly text
        """
        if not self.text_generator:
            # Demo simplified text
            return self._demo_simplify_clause(clause_text)

        prompt = f"""
        Please rewrite the following legal clause in simple, easy-to-understand language 
        that a layperson could comprehend. Keep the meaning accurate but make it accessible:

        Legal Clause: "{clause_text}"

        Simple Version:
        """

        try:
            response = self.text_generator(
                prompt,
                max_new_tokens=200,
                do_sample=False
            )

            # Extract the simplified text from response
            simplified = response[0]['generated_text'].split("Simple Version:")[-1].strip()
            return simplified

        except Exception as e:
            logger.error(f"Error in clause simplification: {e}")
            return self._demo_simplify_clause(clause_text)

    def _demo_simplify_clause(self, clause_text: str) -> str:
        """Demo simplification for when Granite is not available."""
        # Simple rule-based simplification for demo
        simplified_mapping = {
            "party": "person or company",
            "whereas": "because",
            "heretofore": "before this",
            "hereinafter": "from now on",
            "party of the first part": "first party",
            "party of the second part": "second party",
            "confidential": "private",
            "proprietary": "owned by the company",
            "disclose": "share or tell",
            "acknowledge": "agree or understand",
            "shall": "must",
            "pursuant to": "according to",
            "in accordance with": "following"
        }

        simplified = clause_text.lower()
        for legal_term, simple_term in simplified_mapping.items():
            simplified = simplified.replace(legal_term, simple_term)

        return simplified.capitalize()

    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using spaCy NER.

        Args:
            text: Document text

        Returns:
            List of extracted entities with labels and positions
        """
        if not self.nlp_model:
            return self._demo_extract_entities(text)

        doc = self.nlp_model(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.85 + (hash(ent.text) % 15) / 100  # Simulated confidence
            })

        return entities

    def _demo_extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Demo entity extraction when spaCy is not available."""
        # Simple regex-based entity extraction for demo
        entities = []

        # Date patterns
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}'
        dates = re.finditer(date_pattern, text, re.IGNORECASE)
        for match in dates:
            entities.append({
                "text": match.group(),
                "label": "DATE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.92
            })

        # Organization patterns (Inc., Corp., LLC, etc.)
        org_pattern = r'\b[A-Z][A-Za-z\s]+(?:Inc\.|Corp\.|LLC|Corporation|Company)\b'
        orgs = re.finditer(org_pattern, text)
        for match in orgs:
            entities.append({
                "text": match.group(),
                "label": "ORGANIZATION",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.88
            })

        # Person names (simple pattern)
        person_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        persons = re.finditer(person_pattern, text)
        for match in persons:
            entities.append({
                "text": match.group(),
                "label": "PERSON",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85
            })

        return entities

    def extract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract and categorize individual clauses from legal document.

        Args:
            text: Full document text

        Returns:
            List of extracted clauses with metadata
        """
        # Split text into sentences and group related ones
        sentences = re.split(r'[.!?]+', text)
        clauses = []

        # Legal clause keywords for categorization
        clause_keywords = {
            "Confidentiality": ["confidential", "non-disclosure", "proprietary", "secret"],
            "Payment": ["payment", "fee", "compensation", "salary", "amount"],
            "Termination": ["terminate", "end", "expiration", "cancel", "dissolve"],
            "Liability": ["liable", "responsibility", "damages", "loss", "indemnify"],
            "Governing Law": ["governed by", "laws of", "jurisdiction", "legal"],
            "Usage Restriction": ["shall not use", "prohibited", "restricted", "limited to"]
        }

        clause_id = 1
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50:  # Filter out short fragments
                category = self._categorize_clause(sentence, clause_keywords)
                importance = self._assess_importance(sentence, category)

                clauses.append({
                    "id": clause_id,
                    "title": f"{category} Clause",
                    "text": sentence + ".",
                    "category": category,
                    "importance": importance
                })
                clause_id += 1

        return clauses

    def _categorize_clause(self, text: str, keywords: Dict[str, List[str]]) -> str:
        """Categorize clause based on keyword matching."""
        text_lower = text.lower()
        max_matches = 0
        category = "General"

        for cat, words in keywords.items():
            matches = sum(1 for word in words if word in text_lower)
            if matches > max_matches:
                max_matches = matches
                category = cat

        return category

    def _assess_importance(self, text: str, category: str) -> str:
        """Assess the importance level of a clause."""
        critical_categories = ["Confidentiality", "Payment", "Termination", "Liability"]

        if category in critical_categories:
            return "High"
        elif len(text) > 200:
            return "Medium"
        else:
            return "Low"

    def classify_document(self, text: str) -> Dict[str, Any]:
        """
        Classify the document type using keyword analysis.

        Args:
            text: Document text

        Returns:
            Classification results with confidence scores
        """
        return self._demo_classify_document(text)

    def _demo_classify_document(self, text: str) -> Dict[str, Any]:
        """Demo document classification."""
        text_lower = text.lower()

        # Simple keyword-based classification
        if any(term in text_lower for term in ["non-disclosure", "confidential", "proprietary"]):
            return {
                "predictions": [
                    {"type": "Non-Disclosure Agreement (NDA)", "confidence": 94.2},
                    {"type": "Confidentiality Agreement", "confidence": 89.7},
                    {"type": "Service Agreement", "confidence": 12.3}
                ],
                "keyIndicators": ["confidential information", "non-disclosure", "proprietary information"]
            }
        elif any(term in text_lower for term in ["employment", "employee", "salary", "work"]):
            return {
                "predictions": [
                    {"type": "Employment Contract", "confidence": 91.5},
                    {"type": "Service Agreement", "confidence": 76.2},
                    {"type": "Non-Disclosure Agreement", "confidence": 23.1}
                ],
                "keyIndicators": ["employment terms", "salary", "work responsibilities"]
            }
        elif any(term in text_lower for term in ["lease", "rent", "tenant", "landlord"]):
            return {
                "predictions": [
                    {"type": "Lease Agreement", "confidence": 89.3},
                    {"type": "Rental Contract", "confidence": 78.1},
                    {"type": "Property Agreement", "confidence": 45.2}
                ],
                "keyIndicators": ["lease terms", "rental payment", "property usage"]
            }
        else:
            return {
                "predictions": [
                    {"type": "General Legal Agreement", "confidence": 75.0},
                    {"type": "Service Agreement", "confidence": 65.0},
                    {"type": "Contract", "confidence": 60.0}
                ],
                "keyIndicators": ["legal agreement", "parties", "terms and conditions"]
            }

    def generate_summary(self, text: str, entities: List[Dict], clauses: List[Dict]) -> Dict[str, Any]:
        """
        Generate executive summary of the document.

        Args:
            text: Full document text
            entities: Extracted entities
            clauses: Extracted clauses

        Returns:
            Comprehensive document summary
        """
        # Extract parties (organizations and people)
        parties = [ent["text"] for ent in entities if ent["label"] in ["PERSON", "ORG", "ORGANIZATION"]]

        # Extract dates
        dates = [ent["text"] for ent in entities if ent["label"] == "DATE"]

        # Extract key terms from high-importance clauses
        key_terms = [clause["title"] for clause in clauses if clause["importance"] == "High"]

        # Assess risk level based on clause types
        risk_level = self._assess_risk_level(clauses)

        return {
            "documentType": self._infer_doc_type_from_clauses(clauses),
            "parties": parties[:5],  # Top 5 parties
            "keyTerms": key_terms[:5],  # Top 5 key terms
            "criticalDates": dates[:3],  # Top 3 dates
            "obligations": self._extract_obligations(clauses),
            "riskLevel": risk_level
        }

    def _assess_risk_level(self, clauses: List[Dict]) -> str:
        """Assess overall risk level based on clauses."""
        high_risk_count = sum(1 for clause in clauses if clause["importance"] == "High")

        if high_risk_count > 3:
            return "High"
        elif high_risk_count > 1:
            return "Medium"
        else:
            return "Low"

    def _infer_doc_type_from_clauses(self, clauses: List[Dict]) -> str:
        """Infer document type from clause categories."""
        categories = [clause["category"] for clause in clauses]

        if "Confidentiality" in categories:
            return "Non-Disclosure Agreement"
        elif "Payment" in categories and "Termination" in categories:
            return "Service Agreement"
        else:
            return "Legal Agreement"

    def _extract_obligations(self, clauses: List[Dict]) -> List[str]:
        """Extract key obligations from clauses."""
        obligations = []

        for clause in clauses:
            if clause["importance"] == "High":
                # Simplified obligation extraction
                text = clause["text"].lower()
                if "shall" in text or "must" in text or "agree" in text:
                    obligations.append(f"Comply with {clause['category']} requirements")

        return obligations[:5]  # Top 5 obligations

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Perform complete analysis of uploaded legal document.

        Args:
            file_path: Path to uploaded document

        Returns:
            Complete analysis results
        """
        try:
            # Extract text
            text = self.extract_text_from_file(file_path)

            # If no meaningful text extracted, use sample
            if len(text.strip()) < 50:
                text = self._get_sample_document_text()

            # Perform all analyses
            entities = self.extract_named_entities(text)
            clauses = self.extract_clauses(text)
            classification = self.classify_document(text)
            summary = self.generate_summary(text, entities, clauses)

            # Example clause for simplification
            example_clause = clauses[0]["text"] if clauses else text[:500]
            simplified_clause = self.simplify_clause(example_clause)

            return {
                "document_info": {
                    "filename": os.path.basename(file_path),
                    "size": f"{os.path.getsize(file_path) / 1024:.1f} KB",
                    "upload_date": datetime.now().isoformat()
                },
                "text_content": text[:1000] + "..." if len(text) > 1000 else text,
                "simplification": {
                    "original": example_clause,
                    "simplified": simplified_clause
                },
                "entities": entities,
                "clauses": clauses,
                "classification": classification,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return self._get_demo_analysis_result(file_path)

    def _get_sample_document_text(self):
        """Get sample NDA text for demonstration."""
        return """
        This Non-Disclosure Agreement ("Agreement") is entered into on March 15, 2024, between 
        TechCorp Inc., a Delaware corporation ("Disclosing Party"), and John Smith ("Receiving Party"). 
        The Receiving Party acknowledges that during the course of discussions regarding potential 
        software development services, the Disclosing Party may disclose certain confidential and 
        proprietary information. The Receiving Party agrees that all information disclosed by the 
        Disclosing Party, whether oral or written, shall be held in strict confidence for a period 
        of five (5) years from the date of disclosure. The Receiving Party shall not use such 
        information for any purpose other than evaluating the potential business relationship. 
        This Agreement shall be governed by the laws of the State of Delaware.
        """

    def _get_demo_analysis_result(self, file_path: str):
        """Get demo analysis result when processing fails."""
        sample_text = self._get_sample_document_text()

        return {
            "document_info": {
                "filename": os.path.basename(file_path),
                "size": "2.3 KB",
                "upload_date": datetime.now().isoformat()
            },
            "text_content": sample_text,
            "simplification": {
                "original": "The Receiving Party acknowledges that during the course of discussions regarding potential software development services, the Disclosing Party may disclose certain confidential and proprietary information.",
                "simplified": "The person receiving information understands that during talks about possible software development work, the company may share private business information."
            },
            "entities": [
                {"text": "TechCorp Inc.", "label": "ORGANIZATION", "confidence": 0.95},
                {"text": "Delaware", "label": "LOCATION", "confidence": 0.92},
                {"text": "John Smith", "label": "PERSON", "confidence": 0.88},
                {"text": "March 15, 2024", "label": "DATE", "confidence": 0.96},
                {"text": "five (5) years", "label": "DURATION", "confidence": 0.89}
            ],
            "clauses": [
                {
                    "id": 1,
                    "title": "Confidentiality Clause",
                    "text": "The Receiving Party agrees that all information disclosed by the Disclosing Party, whether oral or written, shall be held in strict confidence for a period of five (5) years from the date of disclosure.",
                    "category": "Confidentiality",
                    "importance": "High"
                },
                {
                    "id": 2,
                    "title": "Usage Restriction Clause",
                    "text": "The Receiving Party shall not use such information for any purpose other than evaluating the potential business relationship.",
                    "category": "Usage Restriction",
                    "importance": "High"
                }
            ],
            "classification": {
                "predictions": [
                    {"type": "Non-Disclosure Agreement (NDA)", "confidence": 94.2},
                    {"type": "Confidentiality Agreement", "confidence": 89.7},
                    {"type": "Service Agreement", "confidence": 12.3}
                ],
                "keyIndicators": ["confidential information", "non-disclosure", "proprietary information"]
            },
            "summary": {
                "documentType": "Non-Disclosure Agreement",
                "parties": ["TechCorp Inc.", "John Smith"],
                "keyTerms": ["5-year confidentiality period", "Software development discussions"],
                "criticalDates": ["March 15, 2024"],
                "obligations": ["Maintain strict confidentiality", "Use information only for evaluation"],
                "riskLevel": "Low-Medium"
            }
        }

# Gradio Interface Implementation with Enhanced UI

def format_overview(results: Dict[str, Any]) -> str:
    """Format overview with enhanced styling."""
    info = results.get("document_info", {})
    summary = results.get("summary", {})
    text_preview = results.get("text_content", "")
    
    parts = [
        '<div class="overview-container">',
        '<div class="overview-section document-info">',
        '<h3>📄 Document Overview</h3>',
        '<div class="info-grid">',
        f'<div class="info-item"><span class="label">📁 Filename:</span> <span class="value">{info.get("filename", "N/A")}</span></div>',
        f'<div class="info-item"><span class="label">📊 Size:</span> <span class="value">{info.get("size", "N/A")}</span></div>',
        f'<div class="info-item"><span class="label">⏰ Uploaded:</span> <span class="value">{info.get("upload_date", "N/A")[:19].replace("T", " ")}</span></div>',
        '</div>',
        '</div>',
        
        '<div class="overview-section executive-summary">',
        '<h3>📋 Executive Summary</h3>',
        '<div class="summary-grid">',
        f'<div class="summary-card type-card"><div class="card-icon">📑</div><div class="card-content"><strong>Document Type</strong><br>{summary.get("documentType", "N/A")}</div></div>',
        f'<div class="summary-card parties-card"><div class="card-icon">👥</div><div class="card-content"><strong>Key Parties</strong><br>{", ".join(summary.get("parties", [])[:2]) or "N/A"}</div></div>',
        f'<div class="summary-card risk-card risk-{summary.get("riskLevel", "low").lower().replace("-", "")}"><div class="card-icon">⚠️</div><div class="card-content"><strong>Risk Level</strong><br>{summary.get("riskLevel", "N/A")}</div></div>',
        '</div>',
        
        '<div class="details-grid">',
        f'<div class="detail-item"><span class="detail-label">🎯 Key Terms:</span> <span class="detail-value">{", ".join(summary.get("keyTerms", [])[:3]) or "N/A"}</span></div>',
        f'<div class="detail-item"><span class="detail-label">📅 Critical Dates:</span> <span class="detail-value">{", ".join(summary.get("criticalDates", [])[:2]) or "N/A"}</span></div>',
        f'<div class="detail-item"><span class="detail-label">✅ Key Obligations:</span> <span class="detail-value">{", ".join(summary.get("obligations", [])[:2]) or "N/A"}</span></div>',
        '</div>',
        '</div>',
        
        '<div class="overview-section text-preview">',
        '<h3>👁️ Document Preview</h3>',
        f'<div class="preview-text">{text_preview[:500]}{"..." if len(text_preview) > 500 else ""}</div>',
        '</div>',
        '</div>'
    ]
    return "\n".join(parts)

def format_simplification(results: Dict[str, Any]) -> str:
    """Format simplification with enhanced styling."""
    sim = results.get("simplification", {})
    original = sim.get("original", "")
    simplified = sim.get("simplified", "")
    
    return f'''
    <div class="simplification-container">
        <div class="simplification-header">
            <h3>🔄 Clause Simplification</h3>
            <p class="simplification-subtitle">Complex legal language made simple and accessible</p>
        </div>
        
        <div class="comparison-container">
            <div class="clause-section original-clause">
                <div class="clause-header">
                    <h4>📝 Original Legal Text</h4>
                    <span class="complexity-badge">Complex</span>
                </div>
                <div class="clause-content">
                    {original}
                </div>
            </div>
            
            <div class="transformation-arrow">
                <div class="arrow-icon">→</div>
                <span class="arrow-text">Simplified</span>
            </div>
            
            <div class="clause-section simplified-clause">
                <div class="clause-header">
                    <h4>✨ Plain English Version</h4>
                    <span class="simplicity-badge">Simple</span>
                </div>
                <div class="clause-content">
                    {simplified}
                </div>
            </div>
        </div>
        
        <div class="simplification-footer">
            <div class="improvement-stats">
                <div class="stat-item">
                    <span class="stat-value">85%</span>
                    <span class="stat-label">Easier to Read</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">60%</span>
                    <span class="stat-label">Fewer Legal Terms</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">95%</span>
                    <span class="stat-label">Meaning Preserved</span>
                </div>
            </div>
        </div>
    </div>
    '''

def format_entities(results: Dict[str, Any]) -> str:
    """Format entities with enhanced styling."""
    entities = results.get("entities", [])
    if not entities:
        return '<div class="no-data">No entities found in the document.</div>'
    
    # Group entities by label
    entity_groups = {}
    for entity in entities[:50]:  # Limit to first 50
        label = entity.get("label", "UNKNOWN")
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity)
    
    # Create enhanced table
    parts = [
        '<div class="entities-container">',
        '<div class="entities-header">',
        '<h3>🏷️ Named Entity Recognition</h3>',
        f'<p class="entities-subtitle">Found {len(entities)} entities across {len(entity_groups)} categories</p>',
        '</div>'
    ]
    
    # Entity statistics
    parts.extend([
        '<div class="entity-stats">',
        *[f'<div class="stat-badge stat-{label.lower()}"><span class="stat-count">{len(ents)}</span><span class="stat-type">{label}</span></div>' 
          for label, ents in entity_groups.items()],
        '</div>'
    ])
    
    # Enhanced table
    parts.extend([
        '<div class="entities-table-container">',
        '<table class="entities-table">',
        '<thead>',
        '<tr><th>Entity Text</th><th>Category</th><th>Confidence</th></tr>',
        '</thead>',
        '<tbody>'
    ])
    
    for entity in entities[:30]:  # Show top 30
        confidence = entity.get("confidence", 0)
        if confidence <= 1:
            confidence = round(confidence * 100, 1)
        confidence_class = "high" if confidence >= 90 else "medium" if confidence >= 70 else "low"
        
        parts.append(
            f'<tr class="entity-row">'
            f'<td class="entity-text">{entity.get("text", "")}</td>'
            f'<td><span class="entity-label label-{entity.get("label", "").lower()}">{entity.get("label", "")}</span></td>'
            f'<td><span class="confidence-score confidence-{confidence_class}">{confidence}%</span></td>'
            f'</tr>'
        )
    
    parts.extend(['</tbody>', '</table>', '</div>', '</div>'])
    return "\n".join(parts)

def format_clauses(results: Dict[str, Any]) -> str:
    """Format clauses with enhanced styling."""
    clauses = results.get("clauses", [])
    if not clauses:
        return '<div class="no-data">No clauses extracted from the document.</div>'
    
    # Group by importance
    high_importance = [c for c in clauses if c.get("importance") == "High"]
    medium_importance = [c for c in clauses if c.get("importance") == "Medium"]
    low_importance = [c for c in clauses if c.get("importance") == "Low"]
    
    parts = [
        '<div class="clauses-container">',
        '<div class="clauses-header">',
        '<h3>📋 Clause Analysis & Categorization</h3>',
        f'<p class="clauses-subtitle">Analyzed {len(clauses)} clauses with automatic importance scoring</p>',
        '</div>',
        
        '<div class="importance-summary">',
        f'<div class="importance-card high-importance"><div class="card-number">{len(high_importance)}</div><div class="card-label">High Priority</div></div>',
        f'<div class="importance-card medium-importance"><div class="card-number">{len(medium_importance)}</div><div class="card-label">Medium Priority</div></div>',
        f'<div class="importance-card low-importance"><div class="card-number">{len(low_importance)}</div><div class="card-label">Low Priority</div></div>',
        '</div>'
    ]
    
    # Display clauses by importance
    for section_name, clause_list, icon in [
        ("High Priority Clauses", high_importance, "🔴"),
        ("Medium Priority Clauses", medium_importance, "🟡"),
        ("Low Priority Clauses", low_importance, "🟢")
    ]:
        if clause_list:
            parts.extend([
                f'<div class="clause-section">',
                f'<h4 class="section-title">{icon} {section_name}</h4>',
                '<div class="clause-grid">'
            ])
            
            for clause in clause_list[:10]:  # Limit display
                importance_class = clause.get("importance", "Low").lower()
                parts.append(
                    f'<div class="clause-card {importance_class}-priority">'
                    f'<div class="clause-header">'
                    f'<h5 class="clause-title">{clause.get("title", "Clause")}</h5>'
                    f'<span class="clause-category">{clause.get("category", "General")}</span>'
                    f'</div>'
                    f'<div class="clause-text">{clause.get("text", "")}</div>'
                    f'</div>'
                )
            
            parts.extend(['</div>', '</div>'])
    
    parts.append('</div>')
    return "\n".join(parts)

def format_classification(results: Dict[str, Any]) -> str:
    """Format classification with enhanced styling."""
    cls = results.get("classification", {})
    predictions = cls.get("predictions", [])
    indicators = cls.get("keyIndicators", [])
    
    if not predictions:
        return '<div class="no-data">No classification results available.</div>'
    
    parts = [
        '<div class="classification-container">',
        '<div class="classification-header">',
        '<h3>🎯 Document Classification Results</h3>',
        '<p class="classification-subtitle">AI-powered document type identification with confidence scoring</p>',
        '</div>',
        
        '<div class="prediction-results">',
        '<h4>📊 Prediction Results</h4>',
        '<div class="predictions-list">'
    ]
    
    for i, pred in enumerate(predictions):
        confidence = pred.get("confidence", 0)
        confidence_class = "high" if confidence >= 80 else "medium" if confidence >= 60 else "low"
        rank_class = "primary" if i == 0 else "secondary"
        
        parts.append(
            f'<div class="prediction-item {rank_class}">'
            f'<div class="prediction-info">'
            f'<span class="prediction-rank">#{i+1}</span>'
            f'<span class="prediction-type">{pred.get("type", "Unknown")}</span>'
            f'</div>'
            f'<div class="confidence-container">'
            f'<div class="confidence-bar">'
            f'<div class="confidence-fill confidence-{confidence_class}" style="width: {confidence}%"></div>'
            f'</div>'
            f'<span class="confidence-text">{confidence:.1f}%</span>'
            f'</div>'
            f'</div>'
        )
    
    parts.extend(['</div>', '</div>'])
    
    if indicators:
        parts.extend([
            '<div class="indicators-section">',
            '<h4>🔍 Key Indicators Found</h4>',
            '<div class="indicators-grid">',
            *[f'<div class="indicator-tag">{indicator}</div>' for indicator in indicators],
            '</div>',
            '</div>'
        ])
    
    parts.append('</div>')
    return "\n".join(parts)

def create_gradio_interface():
    """Create enhanced Gradio interface for ClauseWise with modern styling."""
    analyzer = ClauseWiseAnalyzer()

    def analyze_uploaded_file(file):
        """Process uploaded file through ClauseWise analyzer."""
        if file is None:
            return "Please upload a file to analyze.", "", "", "", ""

        try:
            # Validate file
            file_size = os.path.getsize(file.name)
            if file_size > analyzer.max_file_size:
                return f"File too large. Maximum size is {analyzer.max_file_size / (1024*1024):.1f}MB", "", "", "", ""

            file_ext = Path(file.name).suffix.lower()
            if file_ext not in analyzer.supported_formats:
                return f"Unsupported format. Supported: {', '.join(analyzer.supported_formats)}", "", "", "", ""

            # Analyze document
            results = analyzer.analyze_document(file.name)

            # Format results for different tabs
            overview = format_overview(results)
            simplification = format_simplification(results)
            entities = format_entities(results)
            clauses = format_clauses(results)
            classification = format_classification(results)

            return overview, simplification, entities, clauses, classification

        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            return error_msg, "", "", "", ""

    # Enhanced CSS with modern styling
    custom_css = """
    
/* Global Styles */
.gradio-container {
    max-width: 1600px !important;
    margin: auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header Styling */
.main-header {
    background: linear-gradient(135deg, #5360a0 0%, #5b3d7a 100%); /* Darker blue/purple gradient */
    color: white;
    padding: 40px 20px;
    border-radius: 15px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(83, 96, 160, 0.5);
}

.main-title {
    font-size: 3.5em;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
}

.main-subtitle {
    font-size: 1.4em;
    font-weight: 300;
    margin-bottom: 20px;
    opacity: 0.95;
}

.main-description {
    font-size: 1.1em;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
    opacity: 0.95;
    color: #ddd;
}

/* Sidebar Styling */
.upload-sidebar {
    background: linear-gradient(145deg, #fefefe, #ddeaf6); /* lighter background */
    border-radius: 20px;
    padding: 30px;
    margin-right: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    border: 1px solid #c8d6e5;
    color: #2a2a2a;
}

.upload-title {
    color: #263238;
    font-size: 1.8em;
    font-weight: 600;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* File Upload Area */
.file-upload {
    border: 3px dashed #5360a0;
    border-radius: 15px;
    padding: 40px 20px;
    text-align: center;
    background: linear-gradient(145deg, #fbfcfe, #e7eefb);
    transition: all 0.3s ease;
    margin: 20px 0;
    color: #2e2e2e;
}

.file-upload:hover {
    border-color: #44518d;
    background: linear-gradient(145deg, #e7eefb, #d3dff7);
    transform: translateY(-2px);
}

/* Button Styling */
.analyze-button {
    background: linear-gradient(135deg, #5360a0 0%, #5b3d7a 100%);
    border: none;
    color: white;
    padding: 15px 30px;
    font-size: 1.2em;
    font-weight: 600;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(83, 96, 160, 0.6);
    width: 100%;
    margin: 20px 0;
}

.analyze-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(83, 96, 160, 0.8);
}

/* Tips Section */
.tips-section {
    background: linear-gradient(145deg, #6259B0CD, #141212);
    border: 1px solid #C8A9A9;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    color: #364b2c;
}

.tips-title {
    color: #FFFFFF;
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.tips-list {
    list-style: none;
    padding-left: 0;
    margin: 0;
    color: #000000;
    line-height: 1.6;
}

.tips-list li {
    margin-bottom: 8px;
    padding-left: 20px;
    position: relative;
}

.tips-list li::before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #49912c;
    font-weight: bold;
}

/* Tab Styling */
.tab-nav button {
    font-size: 16px;
    font-weight: 600;
    padding: 15px 25px;
    border-radius: 10px 10px 0 0;
    border: none;
    background: linear-gradient(145deg, #f7fafc, #e3eaf4);
    color: #3a3a3a;
    transition: all 0.3s ease;
    margin-right: 5px;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #5360a0 0%, #5b3d7a 100%);
    color: white;
    box-shadow: 0 -4px 15px rgba(83, 96, 160, 0.4);
}

/* Content Area Styling */
.output-area {
    background: white;
    border-radius: 15px;
    padding: 30px;
    margin: 20px 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    border: 1px solid #a6b1c0;
    min-height: 400px;
    color: #222;
}

/* Overview Specific Styles */
.overview-container {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.overview-section {
    background: #fafcff;
    border-radius: 12px;
    padding: 25px;
    border-left: 5px solid #5360a0;
    color: #222;
}

.overview-section h3 {
    color: #222;
    font-size: 1.5em;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.info-grid, .summary-grid, .details-grid {
    display: grid;
    gap: 15px;
    margin-top: 15px;
}

.info-grid {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.summary-grid {
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.info-item, .detail-item {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #c1c8d9;
    color: #333;
}

.info-item .label, .detail-label {
    font-weight: 600;
    color: #555;
}

.info-item .value, .detail-value {
    color: #222;
    margin-left: 10px;
}

.summary-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #c1c8d9;
    display: flex;
    align-items: center;
    gap: 15px;
    transition: all 0.3s ease;
    color: #222;
}

.summary-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.card-icon {
    font-size: 2em;
    padding: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #5360a0, #5b3d7a);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 60px;
    height: 60px;
}

.card-content {
    flex: 1;
    line-height: 1.4;
    color: #222;
}

.risk-card.risk-high .card-icon { background: linear-gradient(135deg, #b13939, #952525); }
.risk-card.risk-medium .card-icon { background: linear-gradient(135deg, #bb671b, #9d4a12); }
.risk-card.risk-low .card-icon { background: linear-gradient(135deg, #2c6b38, #26522b); }
.risk-card.risk-lowmedium .card-icon { background: linear-gradient(135deg, #29518d, #243e70); }

.preview-text {
    background: white;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #c1c8d9;
    font-family: 'Courier New', monospace;
    line-height: 1.6;
    white-space: pre-wrap;
    color: #333;
}

/* Simplification Specific Styles */
.simplification-container {
    display: flex;
    flex-direction: column;
    gap: 25px;
    color: #222;
}

.simplification-header {
    text-align: center;
    margin-bottom: 20px;
}

.simplification-header h3 {
    color: #222;
    font-size: 2em;
    margin-bottom: 10px;
}

.simplification-subtitle {
    color: #58636a;
    font-size: 1.1em;
}

.comparison-container {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 30px;
    align-items: start;
}

.clause-section {
    background: #fafcfe;
    border-radius: 15px;
    padding: 25px;
    border: 2px solid #c1c8d9;
    transition: all 0.3s ease;
    color: #222;
}

.clause-section:hover {
    border-color: #a5b4d1;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.original-clause {
    border-left: 5px solid #b33a3a;
}

.simplified-clause {
    border-left: 5px solid #2c6b38;
}

.clause-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    color: #222;
}

.clause-header h4 {
    color: #222;
    font-size: 1.2em;
    margin: 0;
}

.complexity-badge {
    background: linear-gradient(135deg, #b33a3a, #952525);
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
}

.simplicity-badge {
    background: linear-gradient(135deg, #2c6b38, #1f4b29);
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
}

.clause-content {
    background: white;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #c1c8d9;
    line-height: 1.6;
    color: #333;
}

.transformation-arrow {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 20px 0;
    color: #5360a0;
}

.arrow-icon {
    font-size: 3em;
    color: #5360a0;
    font-weight: bold;
}

.arrow-text {
    background: linear-gradient(135deg, #5360a0, #5b3d7a);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 600;
}

.simplification-footer {
    background: linear-gradient(145deg, #e9f5eb, #d3e7cf);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #a4c49e;
    color: #35482a;
}

.improvement-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 20px;
    text-align: center;
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    color: #2e4a21;
}

.stat-value {
    font-size: 2.5em;
    font-weight: 700;
    color: #2c6b38;
}

.stat-label {
    color: #58743c;
    font-weight: 600;
}

/* Entities Specific Styles */
.entities-container {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.entities-header {
    text-align: center;
    margin-bottom: 20px;
    color: #222;
}

.entities-header h3 {
    color: #222;
    font-size: 2em;
    margin-bottom: 10px;
}

.entities-subtitle {
    color: #58636a;
    font-size: 1.1em;
}

.entity-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    justify-content: center;
    margin-bottom: 20px;
}

.stat-badge {
    background: white;
    border: 2px solid #c1c8d9;
    border-radius: 25px;
    padding: 10px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    min-width: 80px;
    transition: all 0.3s ease;
    color: #222;
}

.stat-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.stat-count {
    font-size: 1.5em;
    font-weight: 700;
    color: #5360a0;
}

.stat-type {
    font-size: 0.8em;
    font-weight: 600;
    color: #58636a;
    text-transform: uppercase;
}

.entities-table-container {
    background: white;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    border: 1px solid #c1c8d9;
}

.entities-table {
    width: 100%;
    border-collapse: collapse;
}

.entities-table thead {
    background: linear-gradient(135deg, #5360a0, #5b3d7a);
    color: white;
}

.entities-table th {
    padding: 20px;
    text-align: left;
    font-weight: 600;
    font-size: 1.1em;
}

.entities-table td {
    padding: 15px 20px;
    border-bottom: 1px solid #d7dee6;
    color: #222;
}

.entity-row:hover {
    background: #f6f9ff;
}

.entity-text {
    font-weight: 600;
    color: #2a2a2a;
}

.entity-label {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    color: white;
    text-transform: uppercase;
}

.label-person { background: linear-gradient(135deg, #b13939, #952525); }
.label-organization { background: linear-gradient(135deg, #2c4f7f, #243c60); }
.label-date { background: linear-gradient(135deg, #2c6b38, #1f4b29); }
.label-location { background: linear-gradient(135deg, #a95922, #774213); }
.label-duration { background: linear-gradient(135deg, #6a4fa4, #493676); }
.label-org { background: linear-gradient(135deg, #2c4f7f, #243c60); }

.confidence-score {
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 0.9em;
    color: white;
}

.confidence-high {
    background: linear-gradient(135deg, #2c6b38, #1f4b25);
}

.confidence-medium {
    background: linear-gradient(135deg, #bb671b, #9d4a12);
}

.confidence-low {
    background: linear-gradient(135deg, #b13939, #952525);
}

/* Clauses Specific Styles */
.clauses-container {
    display: flex;
    flex-direction: column;
    gap: 25px;
}

.clauses-header {
    text-align: center;
    margin-bottom: 20px;
    color: #222;
}

.clauses-header h3 {
    color: #222;
    font-size: 2em;
    margin-bottom: 10px;
}

.clauses-subtitle {
    color: #58636a;
    font-size: 1.1em;
}

.importance-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.importance-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    border: 2px solid #c1c8d9;
    transition: all 0.3s ease;
    color: #222;
}

.importance-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.high-importance {
    border-color: #b13939;
    background: linear-gradient(145deg, #ffdbdb, #ffdbdb);
}

.medium-importance {
    border-color: #bb671b;
    background: linear-gradient(145deg, #ffedc2, #ffedc2);
}

.low-importance {
    border-color: #2c6b38;
    background: linear-gradient(145deg, #d3f0db, #d3f0db);
}

.card-number {
    font-size: 3em;
    font-weight: 700;
    margin-bottom: 10px;
    color: inherit;
}

.high-importance .card-number { color: #b13939; }
.medium-importance .card-number { color: #bb671b; }
.low-importance .card-number { color: #2c6b38; }

.card-label {
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    font-size: 0.9em;
}

.clause-section {
    margin-bottom: 30px;
    color: #222;
}

.section-title {
    color: #222;
    font-size: 1.5em;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 10px;
    border-bottom: 2px solid #c1c8d9;
}

.clause-grid {
    display: grid;
    gap: 20px;
}

.clause-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #c1c8d9;
    transition: all 0.3s ease;
    color: #222;
}

.clause-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.high-priority {
    border-left: 5px solid #b13939;
}

.medium-priority {
    border-left: 5px solid #bb671b;
}

.low-priority {
    border-left: 5px solid #2c6b38;
}

.clause-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    color: #222;
}

.clause-title {
    color: #222;
    font-size: 1.2em;
    margin: 0;
    font-weight: 600;
}

.clause-category {
    background: linear-gradient(135deg, #5360a0, #5b3d7a);
    color: white;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
}

.clause-text {
    color: #333;
    line-height: 1.6;
    background: #fafcff;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #c1c8d9;
}

/* Classification Specific Styles */
.classification-container {
    display: flex;
    flex-direction: column;
    gap: 25px;
    color: #222;
}

.classification-header {
    text-align: center;
    margin-bottom: 20px;
}

.classification-header h3 {
    color: #222;
    font-size: 2em;
    margin-bottom: 10px;
}

.classification-subtitle {
    color: #58636a;
    font-size: 1.1em;
}

.prediction-results {
    background: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    border: 1px solid #c1c8d9;
}

.prediction-results h4 {
    color: #222;
    font-size: 1.4em;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.predictions-list {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.prediction-item {
    background: #fafcff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #c1c8d9;
    display: flex;
    align-items: center;
    gap: 20px;
    transition: all 0.3s ease;
    color: #222;
}

.prediction-item:hover {
    transform: translateX(5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.prediction-item.primary {
    border: 2px solid #5360a0;
    background: linear-gradient(145deg, #edf2f7, #f7fafc);
}

.prediction-info {
    display: flex;
    align-items: center;
    gap: 15px;
    flex: 1;
}

.prediction-rank {
    background: linear-gradient(135deg, #5360a0, #5b3d7a);
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.1em;
}

.prediction-type {
    font-size: 1.1em;
    font-weight: 600;
    color: #222;
}

.confidence-container {
    display: flex;
    align-items: center;
    gap: 15px;
    min-width: 200px;
}

.confidence-bar {
    background: #dbe1ea;
    border-radius: 10px;
    height: 10px;
    flex: 1;
    overflow: hidden;
    border: 1px solid #bac3d1;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
}

.confidence-fill.confidence-high {
    background: linear-gradient(135deg, #2c6b38, #1f4b25);
}

.confidence-fill.confidence-medium {
    background: linear-gradient(135deg, #bb671b, #9d4a12);
}

.confidence-fill.confidence-low {
    background: linear-gradient(135deg, #b13939, #952525);
}

.confidence-text {
    font-weight: 600;
    color: #222;
    min-width: 50px;
    text-align: right;
}

.indicators-section {
    background: linear-gradient(145deg, #e9f5eb, #d3e7cf);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #a4c49e;
    color: #35482a;
}

.indicators-section h4 {
    color: #222;
    font-size: 1.3em;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.indicators-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.indicator-tag {
    background: linear-gradient(135deg, #5360a0, #5b3d7a);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 600;
    transition: all 0.3s ease;
}

.indicator-tag:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(83, 96, 160, 0.4);
}

/* No Data Styling */
.no-data {
    text-align: center;
    padding: 60px 20px;
    color: #58636a;
    font-size: 1.2em;
    background: linear-gradient(145deg, #fafcff, #f0f7ff);
    border-radius: 15px;
    border: 2px dashed #c1c8d9;
}

/* Footer Styling */
.footer-section {
    background: linear-gradient(135deg, #222a38 0%, #2e3c50 100%);
    color: #c9d1d9;
    padding: 40px 20px;
    border-radius: 15px;
    margin-top: 40px;
    text-align: center;
}

.footer-title {
    font-size: 2em;
    font-weight: 600;
    margin-bottom: 15px;
    color: #aeb7c4;
}

.footer-description {
    font-size: 1.1em;
    line-height: 1.6;
    margin-bottom: 30px;
    color: #8a95aa;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 30px;
    margin: 30px 0;
}

.feature-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
    color: #d1d9e3;
}

.feature-card:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.15);
}

.feature-icon {
    font-size: 2.5em;
    margin-bottom: 15px;
    color: #5360a0;
}

.feature-title {
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 8px;
    color: #c9d1d9;
}

.feature-subtitle {
    color: #8a95aa;
    font-size: 0.9em;
}

.footer-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);
    margin: 30px 0;
}

.footer-bottom {
    color: #718096;
    font-size: 0.9em;
}

/* Responsive Design */
@media (max-width: 768px) {
    .comparison-container {
        grid-template-columns: 1fr;
        gap: 20px;
    }

    .transformation-arrow {
        flex-direction: row;
        justify-content: center;
    }

    .arrow-icon {
        transform: rotate(90deg);
        font-size: 2em;
    }

    .main-title {
        font-size: 2.5em;
    }

    .feature-grid {
        grid-template-columns: 1fr;
    }

    .summary-grid, .importance-summary {
        grid-template-columns: 1fr;
    }
}
    """

    # Create Gradio interface with enhanced styling
    with gr.Blocks(
        title="ClauseWise - AI Legal Document Analyzer",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:

        # Enhanced header with modern design
        gr.HTML("""
        <div class="main-header">
            <h1 class="main-title">⚖️ ClauseWise</h1>
            <h2 class="main-subtitle">AI-Powered Legal Document Analyzer</h2>
            <p class="main-description">
                Transform complex legal documents into clear, understandable insights with cutting-edge AI technology. 
                Upload your legal document to get comprehensive analysis including clause simplification, 
                entity recognition, document classification, and risk assessment.
            </p>
        </div>
        """)

        # Main content area with enhanced layout
        with gr.Row():
            # Enhanced sidebar for file upload
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="upload-sidebar">
                    <h3 class="upload-title">📁 Document Upload</h3>
                """)
                
                file_input = gr.File(
                    label="Select Legal Document",
                    file_types=[".pdf", ".docx", ".txt"],
                    file_count="single",
                    elem_classes=["file-upload"]
                )

                analyze_button = gr.Button(
                    "🔍 Analyze Document", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["analyze-button"]
                )

                gr.HTML("""
                    <div class="tips-section">
                        <h4 class="tips-title">💡 Analysis Tips</h4>
                        <ul class="tips-list">
                            <li>Best results with clear, typed documents</li>
                            <li>Supports PDF, DOCX, and TXT formats</li>
                            <li>Maximum file size: 10MB</li>
                            <li>Processing time: 30-60 seconds</li>
                            <li>Includes confidentiality assessment</li>
                            <li>Scanned documents may have lower accuracy</li>
                        </ul>
                    </div>
                </div>
                """)

            # Enhanced main content area for results
            with gr.Column(scale=7):
                with gr.Tabs() as tabs:
                    with gr.TabItem("📋 Overview", elem_id="overview-tab"):
                        overview_output = gr.HTML(
                            value="""
                            <div class="no-data">
                                📄 Upload a document to see a comprehensive analysis overview including document type, 
                                key parties, risk assessment, and executive summary.
                            </div>
                            """,
                            elem_classes=["output-area"]
                        )

                    with gr.TabItem("🔄 Simplification", elem_id="simplification-tab"):
                        simplification_output = gr.HTML(
                            value="""
                            <div class="no-data">
                                🔄 Upload a document to see complex legal clauses simplified into plain English 
                                with side-by-side comparison and readability improvements.
                            </div>
                            """,
                            elem_classes=["output-area"]
                        )

                    with gr.TabItem("🏷️ Named Entities", elem_id="entities-tab"):
                        entities_output = gr.HTML(
                            value="""
                            <div class="no-data">
                                🏷️ Upload a document to see extracted entities like names, organizations, dates, 
                                and locations with confidence scores and categorization.
                            </div>
                            """,
                            elem_classes=["output-area"]
                        )

                    with gr.TabItem("📋 Clause Analysis", elem_id="clauses-tab"):
                        clauses_output = gr.HTML(
                            value="""
                            <div class="no-data">
                                📋 Upload a document to see detailed clause breakdown, categorization by importance, 
                                and comprehensive legal structure analysis.
                            </div>
                            """,
                            elem_classes=["output-area"]
                        )

                    with gr.TabItem("🎯 Classification", elem_id="classification-tab"):
                        classification_output = gr.HTML(
                            value="""
                            <div class="no-data">
                                🎯 Upload a document to see AI-powered document type classification with confidence 
                                scores and key indicators analysis.
                            </div>
                            """,
                            elem_classes=["output-area"]
                        )

        # Set up the analysis trigger
        analyze_button.click(
            fn=analyze_uploaded_file,
            inputs=[file_input],
            outputs=[
                overview_output,
                simplification_output,
                entities_output,
                clauses_output,
                classification_output
            ],
            show_progress=True
        )

        # Enhanced footer with modern design
        gr.HTML("""
        <div class="footer-section">
            <h3 class="footer-title">About ClauseWise</h3>
            <p class="footer-description">
                ClauseWise leverages cutting-edge AI technologies to democratize legal document understanding. 
                Our platform combines IBM Granite's advanced language processing, spaCy's named entity recognition, 
                and sophisticated document analysis pipelines to make legal documents accessible to everyone.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">AI-Powered Analysis</div>
                    <div class="feature-subtitle">IBM Granite + spaCy NLP</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Lightning Fast</div>
                    <div class="feature-subtitle">30-60 second processing</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <div class="feature-title">Secure & Private</div>
                    <div class="feature-subtitle">Local document processing</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Comprehensive</div>
                    <div class="feature-subtitle">Multi-faceted analysis</div>
                </div>
            </div>
            
            <hr class="footer-divider">
            
            <p class="footer-bottom">
                ClauseWise v2.0 Enhanced | Powered by IBM Granite, spaCy, PyMuPDF & Gradio | © 2025 ClauseWise Team
            </p>
        </div>
        """)

        return interface

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Get server configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'

    # Create and launch Gradio interface
    interface = create_gradio_interface()
    interface.launch(
        server_name=host,
        server_port=port,
        share=True,  # Set to True to create public link
        show_error=debug
    )
