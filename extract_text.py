"""
PDF to Markdown Converter

This script extracts text from PDF files and converts it to markdown format.
It processes all PDF files in the 'papers/' directory and saves the markdown 
versions in the 'papers/md/' directory with the same filename but .md extension.

Requirements:
    - pymupdf4llm package for PDF text extraction
    - The 'papers/' directory should exist and contain PDF files
    - The 'papers/md/' directory should exist for output files
"""
import os
import pymupdf4llm

def extract_text_from_pdf(pdf_path, output_path):
    """
    Extract text from a PDF file and save it as markdown.
    
    Args:
        pdf_path (str): Path to the PDF file to be processed
        output_path (str): Path where the extracted markdown will be saved
    
    Returns:
        None: The function writes the extracted text to the specified output path
    """
    # Convert PDF content to markdown format using pymupdf4llm
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # Write the text to the output file in UTF-8 encoding
    import pathlib
    pathlib.Path(output_path).write_bytes(md_text.encode())


if __name__ == "__main__":
    # Define the directory containing PDF files
    pdf_dir = "papers/"
    
    # Create a list of all PDF files in the specified directory
    list_pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # Process each PDF file
    for pdf_path in list_pdfs:
        # Generate output path by changing directory and extension
        output_path = pdf_path.replace('.pdf', '.md')
        output_path = output_path.replace('papers/', 'papers/md/')
        
        # Convert the PDF to markdown and save it
        print(f"Converting {pdf_path} to {output_path}")
        extract_text_from_pdf(pdf_path, output_path)
