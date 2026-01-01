import os
import json
import yaml
import shutil
import re
import pandas as pd
from typing import List, Dict, Optional

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

    def __init__(
        self,
        persist_path: str,
        embedding_model: str
    ):
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

        self.persist_path = persist_path
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
        Loads and normalizes a knowledge base file into texts and metadata.

        Supported formats:
            - TXT (schema-style)
            - CSV
            - Excel
            - JSON
            - YAML

        Args:
            file_path (str): Path to knowledge base file.

        Returns:
            dict:
                {
                    "texts": List[str],
                    "metadatas": List[dict]
                }
        """
        self.logger.info(f"Loading knowledge base: {file_path}")

        if not os.path.exists(file_path):
            self.logger.error(f"Knowledge base file not found: {file_path}")
            raise FileNotFoundError(file_path)

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".txt":
                return self._load_txt_schema(file_path)
            elif ext == ".csv":
                return self._load_csv(file_path)
            elif ext in [".xls", ".xlsx"]:
                return self._load_excel(file_path)
            elif ext == ".json":
                return self._load_json(file_path)
            elif ext in [".yml", ".yaml"]:
                return self._load_yaml(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            self.logger.error("Failed to load knowledge base.", exc_info=True)
            raise RuntimeError("Knowledge base loading failed.") from e

    def _load_txt_schema(self, file_path: str) -> Dict[str, List]:
        """
        Parses schema-style TXT files (table + column descriptions).
        """
        self.logger.debug("Parsing TXT schema file.")

        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        blocks = re.split(r"=+\n", raw)

        texts, metadatas = [], []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            table_match = re.search(r"Table:\s*(.+)", block, re.IGNORECASE)
            if not table_match:
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
                "table": table_name,
                "source": "database",
                "file": os.path.basename(file_path)
            })

        self.logger.info(f"Extracted {len(texts)} schema chunks.")
        return {"texts": texts, "metadatas": metadatas}

    def _load_csv(self, file_path: str) -> Dict[str, List]:
        df = pd.read_csv(file_path)

        texts, metadatas = [], []
        for idx, row in df.iterrows():
            texts.append(row.to_string())
            metadatas.append({
                "type": "table_row",
                "row": idx,
                "file": os.path.basename(file_path)
            })

        return {"texts": texts, "metadatas": metadatas}

    def _load_excel(self, file_path: str) -> Dict[str, List]:
        df = pd.read_excel(file_path)

        texts, metadatas = [], []
        for idx, row in df.iterrows():
            texts.append(row.to_string())
            metadatas.append({
                "type": "table_row",
                "row": idx,
                "file": os.path.basename(file_path)
            })

        return {"texts": texts, "metadatas": metadatas}

    def _load_json(self, file_path: str) -> Dict[str, List]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "texts": [json.dumps(data, ensure_ascii=False, indent=2)],
            "metadatas": [{
                "type": "json",
                "file": os.path.basename(file_path)
            }]
        }

    def _load_yaml(self, file_path: str) -> Dict[str, List]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return {
            "texts": [yaml.dump(data, allow_unicode=True)],
            "metadatas": [{
                "type": "yaml",
                "file": os.path.basename(file_path)
            }]
        }

    # ------------------------------------------------------------------
    # Document + Vector DB
    # ------------------------------------------------------------------

    def build_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> List[Document]:
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

    def build_vector_database(
        self,
        documents: List[Document],
        rebuild: bool = False
    ) -> FAISS:
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
            vectorstore.save_local(self.persist_path)

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
            allow_dangerous_deserialization=True
        )

    def retrieve(
        self,
        query: str,
        k: int = 4,
        with_score: bool = False
    ):
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
        

    def build_from_knowledge_base(
        self,
        knowledge_file: str,
        rebuild: bool = False
    ):
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
            # 1️⃣ Load knowledge base
            kb = self.load_knowledge_base(knowledge_file)
            self.logger.info(
                f"Loaded knowledge base with {len(kb['texts'])} chunks"
            )

            # 2️⃣ Build LangChain Documents
            documents = self.build_documents(
                texts=kb["texts"],
                metadatas=kb["metadatas"]
            )
            self.logger.info(
                f"Converted knowledge base into {len(documents)} documents"
            )

            # 3️⃣ Build vector database
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