

# Systematic Review Tool

This project provides a **Streamlit-based application** for generating structured summaries and systematic reviews from multiple PDF documents. Using Hugging Face's `T5` transformer model, the tool extracts and summarizes key information from uploaded research papers to facilitate efficient review and analysis.

## Features
- **PDF Parsing**: Extract text from uploaded PDFs.
- **Summarization**: Summarize content using a pre-trained Hugging Face `T5` model with GPU support.
- **Structured Review**: Generate a semantic review, including key sections like:
  - Abstract
  - Research Methods
  - Results
- **Aggregation**: Combine multiple summaries into a consolidated systematic review.
- **Downloadable Output**: Export the aggregated review as a JSON file.

## How It Works
1. Upload one or more PDFs via the web interface.
2. The app processes each PDF to:
   - Extract text.
   - Clean and preprocess the content.
   - Generate a structured summary.
3. Multiple summaries are aggregated into a single systematic review.
4. View and download the aggregated review in JSON format.

## Setup and Installation
### Prerequisites
- Python 3.8 or higher
- An NVIDIA GPU (optional but recommended for faster summarization)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run sysReview.py
   ```

## Usage
1. Access the app through your web browser at `http://localhost:8501`.
2. Upload PDF files using the provided interface.
3. Click **Generate Reviews** to process the files.
4. Download the generated systematic review as a JSON file.

## Key Files
- `sysReview.py`: The main application file containing the Streamlit interface and summarization logic.

## Dependencies
- `streamlit`: For building the web interface.
- `transformers`: For the Hugging Face `T5` summarization model.
- `torch`: For GPU acceleration support.
- `PyPDF2`: For PDF text extraction.

## Customization
- Update the summarization model in the `load_summarizer` function to use a different transformer-based model.
- Modify section prompts in the `structured_summary` function for customized review requirements.

