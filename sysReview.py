import os
import json
from pathlib import Path
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import torch

# Initialize Hugging Face Summarization Model with GPU support (if available)
@st.cache_resource
def load_summarizer():
    """
    Caches the summarizer pipeline for improved performance and resource management. 
    Loading the summarizer is computationally expensive, 
    so caching ensures faster subsequent access and reduces redundant computations,
      especially in shared or iterative workflows.
    """
    return pipeline(
        "summarization",
        model="t5-base",
        tokenizer="t5-base",
        device=0 if torch.cuda.is_available() else -1
    )


summarizer = load_summarizer()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading {pdf_path.name}: {e}")

# Function to clean and preprocess text
def clean_text(text):
    """
    Preprocesses the extracted text to enhance summary quality by 
    removing unnecessary whitespace and ensuring consistent formatting. 
    This helps the summarizer generate more accurate and concise results.
    """
    return " ".join(text.split())


# Function to summarize text
def summarize_text(text, prompt, max_length=150, min_length=30):
    try:
        return summarizer(
            f"{prompt}\n{text}", 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False
        )[0]['summary_text']
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {e}")

# Function to remove repeated sentences
def remove_repeated_sentences(summary):
    sentences = set(summary.split('. '))
    return '. '.join(sentences)

# Generate structured semantic review
def structured_summary(text):
    sections = {
        "A: Abstract": "Provide a concise and clear summary of the document's key focus and purpose.",
        "B: Methods": {
            "1. Research Question": "What is the primary research question explicitly stated in the document?",
            "2. Search Strategy": "Summarize the methods used to identify and select studies, such as databases searched and keywords.",
            "3. Inclusion and Exclusion Criteria": {
                "Inclusion": "List the specific criteria used to include studies in this review.",
                "Exclusion": "List the specific criteria used to exclude studies from this review."
            },
            "5. Data Extraction": "Describe the process and tools used for extracting data in this research.",
            "6. Data Synthesis": "Explain the approach and methods used to synthesize the data extracted."
        },
        "D: Results": "Summarize the primary findings and results of this review."
    }
    review = {}
    for section, prompt in sections.items():
        if isinstance(prompt, dict):  # Nested sections
            sub_sections = {}
            for sub_section, sub_prompt in prompt.items():
                if isinstance(sub_prompt, dict):  # Further nested sections
                    nested_sections = {}
                    for nested_section, nested_prompt in sub_prompt.items():
                        result = summarize_text(text, nested_prompt)
                        nested_sections[nested_section] = result
                    sub_sections[sub_section] = nested_sections
                else:
                    result = summarize_text(text, sub_prompt)
                    sub_sections[sub_section] = result
            review[section] = sub_sections
        else:
            result = summarize_text(text, prompt)
            review[section] = result
    return review

# Function to aggregate multiple summaries into a systematic review
def aggregate_reviews(reviews):
    aggregated = {
        "A: Abstract": [],
        "B: Methods": {
            "1. Research Question": [],
            "2. Search Strategy": [],
            "3. Inclusion and Exclusion Criteria": {
                "Inclusion": [],
                "Exclusion": [],
            },
            "5. Data Extraction": [],
            "6. Data Synthesis": [],
        },
        "D: Results": []
    }

    for review in reviews:
        aggregated["A: Abstract"].append(review.get("A: Abstract", ""))
        for key, sub_section in review.get("B: Methods", {}).items():
            if key == "3. Inclusion and Exclusion Criteria":
                aggregated["B: Methods"]["3. Inclusion and Exclusion Criteria"]["Inclusion"].append(
                    sub_section.get("Inclusion", "")
                )
                aggregated["B: Methods"]["3. Inclusion and Exclusion Criteria"]["Exclusion"].append(
                    sub_section.get("Exclusion", "")
                )
            else:
                aggregated["B: Methods"][key].append(sub_section)
        aggregated["D: Results"].append(review.get("D: Results", ""))

    # Concatenate aggregated summaries into single strings and remove repeated sentences
    aggregated["A: Abstract"] = remove_repeated_sentences(" ".join(aggregated["A: Abstract"]))
    for key, sub_section in aggregated["B: Methods"].items():
        if key == "3. Inclusion and Exclusion Criteria":
            aggregated["B: Methods"]["3. Inclusion and Exclusion Criteria"]["Inclusion"] = remove_repeated_sentences(
                " ".join(sub_section["Inclusion"])
            )
            aggregated["B: Methods"]["3. Inclusion and Exclusion Criteria"]["Exclusion"] = remove_repeated_sentences(
                " ".join(sub_section["Exclusion"])
            )
        else:
            aggregated["B: Methods"][key] = remove_repeated_sentences(" ".join(sub_section))
    aggregated["D: Results"] = remove_repeated_sentences(" ".join(aggregated["D: Results"]))

    return aggregated

# Streamlit App
st.title("Systematic Review Tool")
st.write("Upload multiple PDFs to generate structured summaries and a systematic review.")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.button("Generate Reviews"):
    if uploaded_files:
        results = []

        # Process each PDF
        with st.spinner("Processing PDFs..."):
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    st.write(f"Processing ({i+1}/{len(uploaded_files)}): {uploaded_file.name}")
                    text = extract_text_from_pdf(uploaded_file)
                    cleaned_text = clean_text(text)
                    review = structured_summary(cleaned_text)
                    results.append(review)
                    st.success(f"Review completed for {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")

        # Aggregate reviews
        aggregated_review = aggregate_reviews(results)

        # Display and save results
        st.write("## Aggregated Systematic Review")
        st.json(aggregated_review)

        output_path = Path("systematic_review.json")
        with open(output_path, "w") as f:
            json.dump(aggregated_review, f, indent=4)
        st.success(f"Systematic review saved to {output_path.absolute()}")
        st.download_button(
            label="Download Systematic Review JSON",
            data=json.dumps(aggregated_review, indent=4),
            file_name="systematic_review.json",
            mime="application/json"
        )
    else:
        st.error("No PDFs uploaded!")
