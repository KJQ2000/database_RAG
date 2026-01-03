import os
import json
import yaml
import shutil
import re
import pandas as pd
from typing import List, Dict, Optional
import hashlib
import json

from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from src.shared.logger import Logger
from src.shared.constant import *


class BuildVectorDB:
    """
    BuildVectorDB is responsible for:
    1. Loading knowledge base files
    2. Normalizing them into text + metadata
    3. Creating LangChain Documents
    4. Building, saving, and loading FAISS vector databases
    5. Performing similarity-based retrieval

    This class is designed for production RAG systems and supports
    incremental evolution of knowledge bases.
    """

    def __init__(self, persist_path: str, embedding_model: str, hash_store_filename: str, intent:str):
        """
        Initializes the vector database builder.

        Args:
            persist_path (str): Directory where FAISS index will be stored.
            embedding_model (str): OpenAI embedding model name.

        Raises:
            EnvironmentError: If OPENAI_API_KEY is missing.
        """
        self.logger = Logger().get_logger()
        self.logger.info("Initializing BuildVectorDB...")

        self.intent = intent
        self.index_name = 'db_rag_index' if self.intent == 'database_query' else 'store_info_index'
        self.persist_path = persist_path
        self.hash_store_filename = hash_store_filename
        self.hash_store_path = os.path.join(self.persist_path,self.hash_store_filename)
        self.embedding_model_name = embedding_model

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY not found.")
            raise EnvironmentError("OPENAI_API_KEY is required but not set.")

        try:
            self.embedding_model = OpenAIEmbeddings(
                model=self.embedding_model_name,
                openai_api_key=self.api_key,
                request_timeout=30,
                max_retries=3
            )
            self.logger.info(
                f"Embedding model initialized: {self.embedding_model_name}"
            )
        except Exception as e:
            self.logger.error("Failed to initialize embedding model.", exc_info=True)
            raise RuntimeError("Embedding model initialization failed.") from e

    # ------------------------------------------------------------------
    # Knowledge Base Loaders
    # ------------------------------------------------------------------

    def load_knowledge_base(self, file_path: str) -> Dict[str, List]:
        """
        Load a TXT knowledge base and normalize it into chunks for vector storage.

        This loader supports:
        - Database schema documentation
        - Policies / articles / FAQs
        - Mixed content TXT files

        The function automatically detects content type and applies
        appropriate parsing logic.

        Args:
            file_path (str): Path to the knowledge base TXT file.

        Returns:
            Dict[str, List]:
            {
                "texts": List[str],
                "metadatas": List[dict]
            }

        Raises:
            FileNotFoundError: If file does not exist.
            RuntimeError: If parsing fails.
        """

        self.logger.info(f"Loading knowledge base from: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Knowledge base file not found: {file_path}")
            raise FileNotFoundError(file_path)

        if not file_path.lower().endswith(".txt"):
            raise ValueError("Only TXT knowledge base files are supported.")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            if not raw_text.strip():
                self.logger.warning("Knowledge base file is empty.")
                return {"texts": [], "metadatas": []}

            # Decide parsing strategy
            if self.intent == 'database_query':
                self.logger.info("Detected schema-style knowledge base.")
                return self._load_txt_schema(file_path, raw_text)
            else:
                self.logger.info("Detected article-style knowledge base.")
                return self._load_txt_articles(file_path, raw_text)

        except Exception as e:
            self.logger.error("Failed to load knowledge base.", exc_info=True)
            raise RuntimeError("Knowledge base loading failed.") from e
    
    def _load_txt_schema(self, file_path: str, raw_text: str) -> Dict[str, List]:
        """
        Parse schema-style TXT files.

        Each table definition becomes one vector chunk.
        """

        self.logger.debug("Parsing schema-style TXT knowledge base.")

        blocks = re.split(r"\n={3,}\n", raw_text)

        texts: List[str] = []
        metadatas: List[dict] = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            table_match = re.search(r"Table:\s*(.+)", block, re.IGNORECASE)
            if not table_match:
                self.logger.debug("Skipping block without table definition.")
                continue

            table_name = (
                table_match.group(1)
                .strip()
                .lower()
                .replace(" ", "_")
            )

            texts.append(block)
            metadatas.append({
                "type": "schema",
                "entity": "table",
                "table": table_name,
                "source": "database",
                "file": os.path.basename(file_path)
            })

        self.logger.info(f"Extracted {len(texts)} schema chunks.")
        return {"texts": texts, "metadatas": metadatas}

    def _load_txt_articles(self, file_path: str, raw_text: str) -> Dict[str, List]:
        """
        Parse article-style TXT files.

        Supports:
        - Headings
        - Long paragraphs
        - Policy documents
        """

        self.logger.debug("Parsing article-style TXT knowledge base.")

        # Split by headings or blank-line sections
        sections = re.split(r"\n#{1,3}\s+|\n\n+", raw_text)

        texts: List[str] = []
        metadatas: List[dict] = []

        for idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # Chunk long articles safely
            chunks = self._chunk_text(section, max_length=800)

            for chunk_id, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "type": "article",
                    "section_id": idx,
                    "chunk_id": chunk_id,
                    "source": "knowledge_base",
                    "file": os.path.basename(file_path)
                })

        self.logger.info(f"Extracted {len(texts)} article chunks.")
        return {"texts": texts, "metadatas": metadatas}
    
    def _chunk_text(self, text: str, max_length: int = 800) -> List[str]:
        """
        Chunk text into smaller pieces without breaking sentences.

        Args:
            text (str): Input text.
            max_length (int): Max characters per chunk.

        Returns:
            List[str]: List of text chunks.
        """

        if len(text) <= max_length:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) <= max_length:
                current += " " + sentence
            else:
                chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # ------------------------------------------------------------------
    # Document + Vector DB
    # ------------------------------------------------------------------

    def build_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        Converts raw texts and metadata into LangChain Documents.
        """
        if metadatas and len(texts) != len(metadatas):
            raise ValueError("texts and metadatas length mismatch")

        documents = []
        for i, text in enumerate(texts):
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata=metadatas[i] if metadatas else {}
                )
            )

        self.logger.info(f"Built {len(documents)} documents.")
        return documents

    def build_vector_database(self, documents: List[Document], rebuild: bool = False) -> FAISS:
        """
        Builds and persists a FAISS vector database.

        Args:
            documents (List[Document]): Documents to embed.
            rebuild (bool): Whether to delete existing DB.

        Returns:
            FAISS
        """
        try:
            if rebuild and os.path.exists(self.persist_path):
                self.logger.warning("Rebuilding vector database.")
                shutil.rmtree(self.persist_path)
                

            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model
            )

            os.makedirs(self.persist_path, exist_ok=True)
            vectorstore.save_local(self.persist_path,index_name=self.index_name)

            self.logger.info("Vector database built and saved successfully.")
            return vectorstore

        except Exception as e:
            self.logger.error("Vector database build failed.", exc_info=True)
            raise RuntimeError("Vector DB build failed.") from e

    def load_vector_database(self) -> FAISS:
        """
        Loads a persisted FAISS vector database.
        """
        if not os.path.exists(self.persist_path):
            self.logger.error("Vector database not found.")
            raise FileNotFoundError(self.persist_path)

        return FAISS.load_local(
            self.persist_path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
            index_name=self.index_name
        )

    def retrieve(self, query: str, k: int = 4,with_score: bool = False):
        """
        Retrieves relevant documents for a query.

        Args:
            query (str): User query.
            k (int): Top-K results.
            with_score (bool): Include similarity score.

        Returns:
            List[Document] or List[(Document, score)]
        """
        try:
            vectorstore = self.load_vector_database()
            if with_score:
                return vectorstore.similarity_search_with_score(query, k=k)
            return vectorstore.similarity_search(query, k=k)

        except Exception as e:
            self.logger.error("Retrieval failed.", exc_info=True)
            return []
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Computes SHA256 hash of a file.

        Args:
            file_path (str): Path to file.

        Returns:
            str: SHA256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _load_hash_store(self) -> dict:
        """
        Loads stored file hashes.

        Returns:
            dict: {file_path: hash}
        """
        if not os.path.exists(self.hash_store_path):
            return {}

        with open(self.hash_store_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _save_hash_store(self, store: dict):
        """
        Saves file hashes to disk.
        """
        os.makedirs(os.path.dirname(self.hash_store_path), exist_ok=True)

        with open(self.hash_store_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        

    def build_from_knowledge_base(self, knowledge_file: str, rebuild: bool = False):
        """
        Builds or updates the vector database from a knowledge base file.

        This method performs:
            1. Knowledge base loading
            2. Normalization & chunking
            3. Document creation
            4. Vector DB build / rebuild

        Args:
            knowledge_file (str): Path to the knowledge base file.
            rebuild (bool): If True, deletes and rebuilds the vector DB.

        Returns:
            None

        Raises:
            RuntimeError: If any step fails.
        """
        self.logger.info(
            f"Starting vector DB build from knowledge base: {knowledge_file}, rebuild={rebuild}"
        )

        try:
            # Load knowledge base
            kb = self.load_knowledge_base(knowledge_file)
            self.logger.info(
                f"Loaded knowledge base with {len(kb['texts'])} chunks"
            )

            # Build LangChain Documents
            documents = self.build_documents(
                texts=kb["texts"],
                metadatas=kb["metadatas"]
            )
            self.logger.info(
                f"Converted knowledge base into {len(documents)} documents"
            )

            # Build vector database
            self.build_vector_database(
                documents=documents,
                rebuild=rebuild
            )

            self.logger.info("Vector database build completed successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to build vector DB from knowledge base: {e}",
                exc_info=True
            )
            raise RuntimeError("Vector DB build failed") from e
        
    def build_from_knowledge_base_if_changed(self, knowledge_file: str) -> bool:
        """
        Builds or updates the vector database ONLY if the knowledge base file changed.

        Args:
            knowledge_file (str): Path to knowledge base file.

        Returns:
            bool: True if rebuild occurred, False if skipped.
        """
        self.logger.info(f"Checking if vector DB rebuild is needed for: {knowledge_file}")

        try:
            current_hash = self._compute_file_hash(knowledge_file)
            hash_store = self._load_hash_store()
            old_hash = hash_store.get(knowledge_file)

            if old_hash == current_hash:
                self.logger.info("Knowledge base unchanged. Skipping vector DB rebuild.")
                return False

            self.logger.info("Knowledge base changed. Rebuilding vector DB...")

            # Rebuild vector DB
            self.build_from_knowledge_base(knowledge_file=knowledge_file,rebuild=True)

            # Update hash store
            hash_store[knowledge_file] = current_hash
            self._save_hash_store(hash_store)

            self.logger.info("Vector DB rebuilt and hash updated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed during conditional vector DB build: {e}",exc_info=True)