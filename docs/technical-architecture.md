# ClauseWise - AI Legal Document Analyzer
## Technical Architecture & Implementation Guide

### System Architecture

ClauseWise follows a modular architecture designed for scalability and maintainability with:

- **Frontend Layer**: Gradio web interface with responsive design
- **Application Layer**: File processing and analysis coordination
- **AI/ML Layer**: IBM Granite integration with spaCy NER
- **Data Layer**: File storage and optional caching

### Core Components

1. **Document Processing Pipeline**
   - Multi-format support (PDF, DOCX, TXT)
   - Robust text extraction using PyMuPDF and python-docx
   - Error handling and fallback mechanisms

2. **AI Analysis Engine**
   - IBM Granite integration for text generation and simplification
   - spaCy NER for entity extraction
   - Custom legal document classification
   - Clause extraction and categorization

3. **Security Implementation**
   - File upload validation and sanitization
   - Size and format restrictions
   - Secure file handling with automatic cleanup

### Technology Stack

- **Python 3.9+**: Core programming language
- **IBM Granite 3.2**: Large language model for text generation
- **spaCy 3.4+**: Named entity recognition and NLP processing
- **PyMuPDF**: PDF text extraction
- **python-docx**: DOCX document processing
- **Gradio**: Web interface framework

### Performance Characteristics

- **Processing Speed**: 2-5 seconds per document
- **Memory Usage**: 2-4GB RAM typical, 8GB+ recommended
- **File Size Limit**: 10MB maximum
- **Supported Formats**: PDF, DOCX, TXT
- **Accuracy**: 90%+ for common legal document types
