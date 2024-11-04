import faiss
import logging
import pathlib
import pickle
import sys
import torch
import yaml

from airllm import AutoModel
from gpt4all import GPT4All
from huggingface_hub import hf_hub_download
#from llama_cpp import Llama # use the langchain wrapper first and maybe this later for more control
from pathlib import Path
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import Ollama, LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout)
logger = logging.getLogger('logger')


class DocumentLoaderManager:
    """Handles loading and splitting of documents."""

    def __init__(self, data_directory, chunk_size=500, chunk_overlap=50):
        self.data_directory = Path(data_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        """Load txt, PDF, and Word documents from a directory using LangChain document loaders.
        OneDriveLoader: https://python.langchain.com/docs/integrations/document_loaders/microsoft_onedrive/
        """
        documents = []
        for file_path in self.data_directory.iterdir():  # Iterate over files in directory
            logger.debug('next file ' + str(file_path.absolute()))
            if file_path.suffix == '.txt':
                loader = TextLoader(file_path, autodetect_encoding=True)
            elif file_path.suffix == '.pdf':
                loader = UnstructuredPDFLoader(file_path)
            elif file_path.suffix == '.docx':
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                continue

            docs = loader.load()  # Load the documents
            documents.extend(docs)  # Append the loaded document(s) to the list

        logger.info('number of documents loaded: ' + str(len(documents)))

        return documents

    def split_documents(self, documents):
        """Split documents into smaller chunks using LangChain's text splitter."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_docs = text_splitter.split_documents(documents)
        return split_docs


class EmbeddingManager:
    """Handles the generation of embeddings for documents."""

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents):
        """Generate embeddings for a list of documents."""
        texts = [doc.page_content for doc in documents]  # Extract the raw text from Document objects
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings


class VectorStore:
    """Manages the FAISS index for storing and retrieving embeddings."""

    def __init__(self, path=None):
        self.path = path
        if path is None:
            raise FileNotFoundError('No path specified for vector store.')
        self.index = None
        self.documents = []

    def store_embeddings(self, embeddings, documents):
        """Store embeddings and documents in a FAISS index."""
        dimension = embeddings.shape[1]  # Embedding size
        self.index = faiss.IndexFlatL2(dimension)  # FAISS index for L2 similarity
        self.index.add(embeddings)  # Add embeddings to index
        self.documents = documents

        # Save index and corresponding documents for later retrieval
        with open(self.path, 'wb') as f:
            pickle.dump((self.index, self.documents), f)

    def load_faiss_index(self):
        """Load the FAISS index and documents from disk."""
        with open(self.path, 'rb') as f:
            self.index, self.documents = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        """Search the FAISS index for the top_k most similar documents to the query."""
        if len(self.documents) > top_k:
            top_k = len(self.documents)
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.documents[i], distances[0][i]) for i in indices[0]]


class RAGSystem:
    """Implements the Retrieval Augmented Generation (RAG) workflow."""

    def __init__(self,
                 model_name='distilgpt2',
                 model_source='huggingface_hub',
                 model_file=None,
                 model_prompt=None,
                 repo_id=None,
                 token=None,
                 vector_store_path=None):
        self.vector_store = VectorStore(vector_store_path)
        self.embedding_manager = EmbeddingManager()
        self.document_loader = None
        self.model_source = model_source
        self.model = None
        self.model_prompt = model_prompt
        
        # Detect if GPU is available and select the device
        self.device = ('cuda' if torch.cuda.is_available()
                              else 'mps' if torch.backends.mps.is_available()
                              else 'cpu')
        logger.info(f"Using device: {self.device}")

        if self.model_source == 'airllm':
            self.model = AutoModel.from_pretrained(model_name)
        elif self.model_source == 'gpt4all':
            self.model = GPT4All(model_file)
        elif self.model_source == 'ollama':
            # Initialize Ollama model (local LLM)
            self.model = Ollama(model=model_name)
        elif self.model_source == 'huggingface_hub':
            # Initialize tokenizer and model on the correct device
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, token=token)
            model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(self.device)
            self.model = pipeline('text-generation', model=model, tokenizer=self.tokenizer, device=0 if self.device == 'cuda' else -1)
        elif self.model_source == 'llama_cpp':
            #self.model = Llama.from_pretrained(repo_id=model_name, filename=model_file, token=token)
            if not pathlib.Path(model_file).exists():
                model_file = hf_hub_download(
                    repo_id=model_name,
                    filename=model_file,
                    resume_download=True
                    )
            # Callbacks support token-wise streaming
            self.model = LlamaCpp(
                model_path=model_file,
                temperature=0,
                max_tokens=2000,
                top_p=1,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=True
                )

    def initialize_document_loader(self, data_directory):
        """Initialize the document loader manager."""
        self.document_loader = DocumentLoaderManager(data_directory)

    def load_and_index_documents(self):
        """Load documents, split them, generate embeddings, and store them in FAISS."""
        raw_docs = self.document_loader.load_documents()
        docs = self.document_loader.split_documents(raw_docs)

        embeddings = self.embedding_manager.generate_embeddings(docs)
        self.vector_store.store_embeddings(embeddings, docs)

    def load_existing_index(self):
        """Load existing FAISS index from disk."""
        self.vector_store.load_faiss_index()

    def retrieve_documents(self, query, top_k=3):
        """Retrieve relevant documents based on a query."""
        query_embedding = self.embedding_manager.embedding_model.encode([query])
        return self.vector_store.search(query_embedding, top_k)

    def generate_response(self, prompt):
        """Generate a response based on the prompt using a local language model."""
        if self.model_source == 'ollama':
            # Use Ollama LLM to generate the response
            response = self.model.invoke(prompt)
            return response
        elif self.model_source in['airllm', 'gpt4all', 'huggingface_hub']:
            # Use model loaded from HuggingFace model repository to generate the response
            #max_length=150, use max_new_tokens to avoid issues with length being insufficient
            response = self.model(prompt, max_new_tokens=1000, num_return_sequences=1)
            return response[0]['generated_text']
        elif self.model_source == 'llama_cpp':
            response = self.model.invoke(prompt)
            return response

    def rag(self, query, top_k=3):
        """Perform the full Retrieval-Augmented Generation (RAG) workflow."""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        # Prepare the prompt for the language model using the retrieved documents
        context = '\n'.join([doc.page_content for doc, _ in retrieved_docs])

        if self.model_source == 'llama_cpp':
            context = context[0:512]

        system_message = f"You are a friendly AI assistant. Please answer the user's question given the following context only:\n{context}. Refer only to information that is contained within that context and nothing else in order to answer the question. Keep your answer as concise as possible while including only the relevant information. Be careful not to repeat information you have already provided in your answer. Say things only once and do not provide a summary of what you have already stated."
        prompt = self.model_prompt.format(variable=system_message)

        # Generate a response using the local language model
        response = self.generate_response(prompt)

        #if self.model_source == 'llama_cpp':
        #    return response['choices'][0]['text']

        return response


class Config:
    def __init__(self, path):
        self.path = path
        self.params = yaml.safe_load(open(self.path, 'r'))


if __name__ == "__main__":

    #model_name = 'microsoft/Phi-3.5-mini-instruct'
    #model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    #model_name = 'nemotron-mini'
    #model_name = 'nvidia/Mistral-NeMo-Minitron-8B-Base'

    #model_name = 'microsoft/phi-2'
    #model_file = None
    #token = None

    #model_name = 'bartowski/gemma-2b-aps-it-GGUF'
    #model_file = 'gemma-2b-aps-it-Q6_K_L.gguf'

    #model_name = 'microsoft/Phi-3.5-mini-instruct'

    #model_name = 'Meta-Llama-3-8B.Q8_0.gguf'
    #model_name = 'bartowski/Llama3.1-8B-Cobalt-GGUF'
    #model_file = 'Llama3.1-8B-Cobalt-Q6_K_L.gguf'
    #model_source = 'llama_cpp'
    #token = None

    #model_name = 'TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF'
    #model_file = 'tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf'

    #data_path = '../data/'
    #docs_path = data_path + '/docs/'
    #vector_store_path = data_path + '/vector_store/faiss_index.pickle'

    config = Config('Platypus2-70B-instruct.yaml')
    model_source = config.params['model']['source']
    model_name = config.params['model']['name']
    model_file = config.params['model']['file']
    model_prompt = config.params['model']['prompt']
    data_path = config.params['data']['base_path']
    docs_path = config.params['data']['docs_path']
    vector_store_type = config.params['vector_store']['type']
    vector_store_path = config.params['vector_store']['base_path']
    token = config.params['token']

    # Initialize the RAG System
    rag_system = RAGSystem(model_name=model_name,
                           model_source=model_source,
                           model_file=model_file,
                           model_prompt=model_prompt,
                           token=token,
                           vector_store_path=vector_store_path + '/' + vector_store_type + '_index.pickle')

    # Initialize the document loader with the data directory
    rag_system.initialize_document_loader(data_directory=docs_path)

    # Load, split, generate embeddings, and index the documents
    rag_system.load_and_index_documents()

    # Handle a sample query
    query = 'When does Kāinga Ora collect personal information about its customers?'
    query = 'What kinds of personal information does Kāinga Ora collect from its customers?'
    response = rag_system.rag(query)
    
    print('Response:', response)
