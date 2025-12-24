from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load PDF
loader = PyPDFLoader("data/funding_sample.pdf")
documents = loader.load()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in vector DB
db = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="vector_db"
)

db.persist()
print("✅ Funding data ingested successfully")
