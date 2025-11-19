from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os,shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

print("Welcome to ChatWithPDF!")

pdf_path = input("Please Load Your PDF File by entering the path: \n")
#pdf_path = "C:\\Users\\amogh\\OneDrive\\Desktop\\ChatWithPDF\\data\\ITPM_Unit_6.pptx.pdf"

if not os.path.exists(pdf_path):
    print("❌ File not found!")
    exit()

if os.path.exists("my_chroma_db"):
    shutil.rmtree("my_chroma_db")

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text = "".join([doc.page_content for doc in docs])
length = len(text)

if length < 2000:
    chunk_size = 300
elif length < 10000:
    chunk_size = 500
else:
    chunk_size = 800

chunk_overlap = chunk_size // 10

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_documents(docs)

db_path = "my_chroma_db"

# db_exists = os.path.exists(db_path) and len(os.listdir(db_path)) > 0

# if db_exists:
#     vector_space = Chroma(
#         persist_directory=db_path,
#         embedding_function=model
#     )
# else:

vector_space = Chroma.from_documents(
        documents=chunks,
        embedding=model,
        persist_directory=db_path,
        collection_name="pdfText"
    )

print("PDF has uploaded successfully....")
query=input("Your question:")

retriever = vector_space.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

result = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in result])

prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="""
You are an assistant that answers questions using ONLY the provided PDF text.

PDF EXTRACT:
{text}

QUESTION:
{question}

TASK:
Explain the answer in a very simple, clear, and easy-to-understand way.
Use only the information from the PDF extract.
If the answer is not found in the text, say: "The answer is not available in the PDF."
"""
)

chain = prompt | llm | parser

response = chain.invoke({"text": context, "question": query})
print("\nANSWER:\n")
print(response)




