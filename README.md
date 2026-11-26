# StudyMate – AI-Powered Study Assistant

## Project Details

**Project Name:** StudyMate – AI-Powered Study Assistant

**Roll Numbers:**
- 23A81A6119
- 23A81A6127

## 1. Project Overview

StudyMate is an AI-powered study assistant that helps students understand and learn from PDF study materials. It uses Retrieval-Augmented Generation (RAG), Google Gemini, LangChain, and FAISS to answer questions, summarize documents, and generate exam preparation questions based on uploaded PDFs.

## 2. Objectives

- Help students understand academic PDF materials.
- Provide answers based on the uploaded document.
- Generate structured summaries for quick revision.
- Create exam preparation questions.
- Provide relevant PDF page references.

## 3. Key Features

- **PDF Upload:** Upload study materials in PDF format.
- **Ask Questions:** Ask questions and receive answers based on relevant PDF content.
- **PDF Summarization:** Generate organized summaries covering key concepts and topics.
- **Exam Question Generation:** Generate five conceptual and analytical questions.
- **Source References:** Display relevant PDF page numbers.
- **Caching:** Reuse previously generated embeddings and FAISS indexes for the same PDF.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| Google Gemini Flash | AI response generation |
| Gemini Embedding 2 | Cloud-based text embeddings |
| LangChain | Document processing and integration |
| FAISS | Vector storage and similarity search |
| PyPDF | PDF text extraction |
| RecursiveCharacterTextSplitter | Text chunking |
| Google Generative AI | Gemini API integration |

## 5. System Workflow

1. The user uploads a PDF document.
2. The application calculates an MD5 hash to identify the document.
3. PyPDF extracts readable text from the PDF pages.
4. Recursive Character Text Splitter divides the extracted text into chunks of 800 characters with 120 characters of overlap.
5. Google Gemini Embedding 2 generates cloud-based embeddings.
6. FAISS stores the embeddings for similarity-based retrieval.
7. The generated FAISS index and document chunks are cached locally.
8. The user selects a task: Ask a Question, Summarize PDF, or Generate Exam Questions.
9. Relevant document context is retrieved for question answering and exam question generation.
10. Google Gemini Flash generates the response.
11. StudyMate displays the generated output and relevant PDF page references.

## 6. Retrieval-Augmented Generation (RAG)

StudyMate uses RAG to provide document-based answers.

- **Document Loading:** Extracts text from uploaded PDFs.
- **Text Splitting:** Divides the extracted text into manageable chunks.
- **Embedding Generation:** Converts text chunks into numerical vector representations using Gemini Embedding 2.
- **Vector Storage:** Stores embeddings using FAISS.
- **Retrieval:** Finds relevant chunks using similarity search.
- **Generation:** Sends the retrieved context to Gemini Flash to generate responses.

For PDF summarization, the application combines all available chunks and sends the content directly to Gemini Flash.

## 7. Caching Mechanism

StudyMate uses MD5 hashing to identify uploaded PDF files and stores their processed FAISS indexes and chunks in a local cache directory.

When the same PDF is uploaded again, the application attempts to load the cached data instead of repeating the complete document-processing and embedding-generation process.

## 8. Application Tasks

- Ask a Question: Provides answers using relevant PDF content.
- Summarize PDF: Creates a structured summary of the uploaded document.
- Generate Exam Questions: Generates five questions for exam preparation.

## 9. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/nagavasanthi2005/StudyMate-AI-Study-Assistant.git                                                         

Navigate to the project folder:

```bash
cd StudyMate-AI-Study-Assistant
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key
```

Run the application:

```bash
streamlit run app.py
```

## 10. Expected Output

The application provides:
- Answers to questions based on uploaded PDF content.
- Structured study summaries.
- Exam preparation questions.
- Relevant PDF page references.

## 11. Conclusion

StudyMate provides an AI-powered learning environment that helps students interact with their academic PDF materials. By combining RAG, cloud-based embeddings, FAISS, and Google Gemini, it supports document-based question answering, summarization, and exam preparation through a simple Streamlit interface.