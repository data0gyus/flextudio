import os
from pathlib import Path
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_rag_system = None

class RAGSystem:
    def __init__(self, doc_dir: str = "rag_documents"):
        self.doc_dir = Path(doc_dir)
        
        print("Gemini Embeddings API 초기화...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # 청크 크기 줄이기 (메모리 절약)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        self.vectorstore = None
    
    def load_documents(self) -> List[Dict[str, str]]:
        """TXT 파일 로드"""
        documents = []
        
        if not self.doc_dir.exists():
            print(f"문서 디렉토리 없음: {self.doc_dir}")
            return documents
        
        txt_files = list(self.doc_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"TXT 파일 없음: {self.doc_dir}")
            return documents
        
        for txt_path in txt_files:
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                    if text.strip():
                        documents.append({
                            "content": text,
                            "source": txt_path.name
                        })
                        print(f"로드: {txt_path.name}")
            
            except Exception as e:
                print(f"TXT 로드 실패 ({txt_path.name}): {e}")
        
        return documents
    
    def build_vectorstore(self, force_recreate: bool = False):
        """벡터 DB 생성"""
        
        documents = self.load_documents()
        
        if not documents:
            print("로드할 문서 없음")
            return
        
        # 텍스트 분할
        all_splits = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc["content"])
            for split in splits:
                all_splits.append({
                    "content": split,
                    "source": doc["source"]
                })
        
        print(f"총 {len(all_splits)}개 청크 생성")
        
        # 청크 수 제한 (메모리 절약)
        max_chunks = 500
        if len(all_splits) > max_chunks:
            print(f"⚠️ 청크 수 제한: {len(all_splits)} → {max_chunks}")
            all_splits = all_splits[:max_chunks]
        
        # 벡터 DB 생성
        texts = [s["content"] for s in all_splits]
        metadatas = [{"source": s["source"]} for s in all_splits]
        
        print("벡터 DB 생성 중...")
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print(f"벡터 DB 생성 완료 ({len(texts)}개 청크)")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """검색"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in results
            ]
        except Exception as e:
            print(f"검색 오류: {e}")
            return []


def initialize_rag_system(force_recreate: bool = False):
    """RAG 초기화"""
    global _rag_system
    
    try:
        print("RAG 시스템 초기화 (TXT 전용)")
        _rag_system = RAGSystem()
        _rag_system.build_vectorstore(force_recreate=force_recreate)
        return _rag_system
    except Exception as e:
        print(f"RAG 초기화 실패: {e}")
        print("RAG 없이 계속 진행")
        return None

def get_rag_system():
    return _rag_system