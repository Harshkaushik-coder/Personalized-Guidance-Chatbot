# Import necessary libraries
from transformers import pipeline  
from langchain_community.vectorstores import Chroma  
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_core.documents import Document  

# Initialize GPT-2 for text generation
generator = pipeline("text-generation", model="gpt2")

# Define the prompt for generating career options
prompt = """
Generate 5 modern career options with the following format:
Career Name: [Career Name]
Description: [Career Description]
Skills: [Skills List]
"""

# Generate text using GPT-2 model
output = generator(prompt, max_length=500, do_sample=True, temperature=0.9)

# Extract and structure career information
text = output[0]["generated_text"]
career_docs = []
career_lines = text.split("\n")

# Parse career details into structured format
for line in career_lines:
    line = line.strip()
    if line.startswith("Career Name:"):
        career_name = line.replace("Career Name:", "").strip()
    elif line.startswith("Description:"):
        description = line.replace("Description:", "").strip()
    elif line.startswith("Skills:"):
        skills = line.replace("Skills:", "").strip()
        career_docs.append(Document(page_content=f"Career Name: {career_name}\nDescription: {description}\nSkills: {skills}"))

# Create embeddings model and vector store for document storage
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=career_docs,
    embedding=embedding_model,
    persist_directory="career_db"
)

# Persist the vector store to disk
vectorstore.persist()
