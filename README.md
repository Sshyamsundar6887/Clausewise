# ClauseWise - AI-Powered Legal Document Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Interface-Gradio-orange)](https://gradio.app/)
[![IBM Granite](https://img.shields.io/badge/AI-IBM%20Granite-blue)](https://www.ibm.com/granite)

ClauseWise is an innovative AI-powered legal document analyzer that simplifies, decodes, and classifies complex legal texts. Built for lawyers, businesses, and laypersons alike, it uses cutting-edge generative AI to make legal documents more accessible and understandable.

![ClauseWise Demo](docs/images/clausewise-demo.png)

## 🚀 Features

### 📝 **Clause Simplification**
- Automatically rewrites complex legal clauses into simplified, layman-friendly language
- Powered by IBM Granite's advanced language processing capabilities
- Maintains legal accuracy while improving comprehension

### 🏷️ **Named Entity Recognition (NER)**
- Identifies and extracts key legal entities:
  - Parties and organizations
  - Important dates and deadlines
  - Monetary values and terms
  - Legal obligations and restrictions
  - Geographic locations

### 📋 **Clause Extraction & Breakdown**
- Detects and segments individual clauses from lengthy legal documents
- Categorizes clauses by type (Confidentiality, Payment, Termination, etc.)
- Assesses importance and risk level of each clause
- Provides focused analysis for critical document sections

### 🎯 **Document Classification**
- Accurately classifies legal documents into categories:
  - Non-Disclosure Agreements (NDAs)
  - Employment Contracts
  - Service Agreements
  - Lease Agreements
  - Partnership Agreements
  - And more...

### 📄 **Multi-Format Support**
- **PDF**: Advanced text extraction using PyMuPDF
- **DOCX**: Microsoft Word document processing
- **TXT**: Plain text file analysis
- Intelligent handling of complex document layouts

### 🖥️ **User-Friendly Interface**
- Modern Gradio-based web interface
- Intuitive tabbed layout for different analysis modes
- Real-time processing with progress indicators
- Mobile-responsive design

## 🛠️ Technologies Used

- **AI/ML**: IBM Granite 3.2, spaCy NLP, PyTorch, Transformers
- **Document Processing**: PyMuPDF, python-docx, docx2txt
- **Web Interface**: Gradio, HTML5, CSS3, JavaScript
- **Backend**: Python 3.8+, FastAPI (optional)
- **Deployment**: Docker, Heroku, AWS, Google Cloud

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM (optional, for faster processing)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/clausewise-legal-analyzer.git
   cd clausewise-legal-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv clausewise_env

   # Windows
   clausewise_env\Scripts\activate

   # macOS/Linux
   source clausewise_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   # Download spaCy English model
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   python clausewise_implementation.py
   ```

5. **Access the web interface**
   Open your browser and navigate to: http://localhost:7860

## 🚀 Usage

### Web Interface

1. **Upload Document**: Drag and drop or select a legal document (PDF, DOCX, or TXT)
2. **Choose Analysis Mode**: Navigate through different tabs for various analysis types
3. **Review Results**: Examine simplified clauses, extracted entities, and document classification
4. **Export Results**: Copy or download analysis results for further use

### Python API

```python
from clausewise_implementation import ClauseWiseAnalyzer

# Initialize analyzer
analyzer = ClauseWiseAnalyzer()

# Analyze document
results = analyzer.analyze_document("path/to/your/document.pdf")

# Access specific analysis results
entities = results['entities']
clauses = results['clauses']
classification = results['classification']
summary = results['summary']
```

## 🏗️ Project Structure

```
clausewise-legal-analyzer/
├── clausewise_implementation.py  # Main application file
├── requirements.txt              # Python dependencies
├── README.md                    # Project documentation
├── docs/                        # Documentation files
│   ├── deployment-guide.md      # Deployment instructions
│   └── technical-architecture.md # Technical details
├── tests/                       # Unit and integration tests
├── config/                      # Configuration files
├── sample_documents/            # Sample legal documents
└── uploads/                     # File upload directory
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# All tests
python -m pytest tests/ -v
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t clausewise .

# Run container
docker run -p 7860:7860 clausewise
```

### Heroku Deployment

```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main
```

For detailed deployment instructions, see [docs/deployment-guide.md](docs/deployment-guide.md).

## 📊 Performance

- **Document Processing**: 2-5 seconds per document
- **Supported File Size**: Up to 10MB
- **Accuracy**: 90%+ for common legal document types
- **Languages**: English (expandable to other languages)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Granite Team** for providing advanced language models
- **spaCy Team** for excellent NLP libraries
- **Gradio Team** for the intuitive web interface framework
- **Legal AI Research Community** for inspiration and guidance

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/clausewise-legal-analyzer/issues)
- **Email**: support@clausewise.ai
- **Discord**: [ClauseWise Community](https://discord.gg/clausewise)

## 🎯 Roadmap

- [ ] Multi-language support (Spanish, French, German)
- [ ] Advanced contract comparison features
- [ ] Integration with popular legal databases
- [ ] Mobile application development
- [ ] Enterprise API with authentication
- [ ] Real-time collaboration features

---

**ClauseWise** - Making legal documents accessible to everyone through the power of AI.
