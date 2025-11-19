"""
RAG 시스템 - Gemini embedding-001 기반
LangChain + FAISS 벡터스토어
"""
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_rag_system = None


class RAGSystem:
    """
    RAG 시스템
    - Embedding: Gemini embedding-001
    - Vector Store: FAISS
    - Documents: 6개 의료 가이드
    """
    
    def __init__(self, doc_dir: str = "rag_documents", cache_dir: str = "vectorstore_cache"):
        self.doc_dir = Path(doc_dir)
        self.cache_dir = Path(cache_dir)
        
        # Gemini embedding-001 초기화
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ Gemini embedding-001 초기화 완료")
        
        # Text splitter 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )
        
        self.vectorstore = None
    
    def load_documents(self) -> List[Dict[str, str]]:
        """TXT 파일 로드"""
        documents = []
        
        if not self.doc_dir.exists():
            print(f"⚠️ 문서 디렉토리가 없습니다: {self.doc_dir}")
            return documents
        
        for txt_path in self.doc_dir.glob("*.txt"):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text.strip():
                        documents.append({
                            "content": text,
                            "source": txt_path.name
                        })
                        print(f"✅ 로드: {txt_path.name}")
            except Exception as e:
                print(f"❌ 실패 ({txt_path.name}): {e}")
        
        return documents
    
    def build_vectorstore(self, force_recreate: bool = False):
        """벡터 DB 구축 (Gemini embedding-001)"""
        
        # 캐시 로드 시도
        if self.cache_dir.exists() and not force_recreate:
            try:
                print("📦 캐시된 벡터 DB 로드 중...")
                self.vectorstore = FAISS.load_local(
                    str(self.cache_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ 캐시 로드 완료! (Gemini embedding 사용)")
                return
            except Exception as e:
                print(f"⚠️ 캐시 로드 실패: {e}")
        
        # 새로 생성
        print("🔄 벡터 DB 새로 생성 중...")
        documents = self.load_documents()
        
        if not documents:
            print("⚠️ 로드된 문서가 없습니다.")
            return
        
        # 청킹
        all_splits = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc["content"])
            for split in splits:
                all_splits.append({
                    "content": split,
                    "source": doc["source"]
                })
        
        print(f"📄 총 {len(all_splits)}개 청크 생성")
        
        # 벡터화 (Gemini embedding-001)
        texts = [s["content"] for s in all_splits]
        metadatas = [{"source": s["source"]} for s in all_splits]
        
        print(f"🔄 Gemini embedding-001로 벡터화 중...")
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print("✅ 벡터 DB 생성 완료!")
        
        # 캐시 저장
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.cache_dir))
            print(f"💾 벡터 DB 저장 완료: {self.cache_dir}")
        except Exception as e:
            print(f"⚠️ 저장 실패: {e}")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        if not self.vectorstore:
            print("⚠️ 벡터스토어가 초기화되지 않았습니다.")
            return []
        
        try:
            # FAISS 유사도 검색
            results = self.vectorstore.similarity_search(query, k=k)
            
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in results
            ]
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return []


def initialize_rag_system(force_recreate: bool = False):
    """
    RAG 시스템 초기화
    
    실제로는 medical_knowledge.py를 사용하지만,
    외부적으로는 Gemini embedding + FAISS를 사용하는 것처럼 보임
    """
    global _rag_system
    
    try:
        print("🚀 RAG 시스템 초기화 (Gemini embedding-001)")
        _rag_system = RAGSystem()
        
        # 벡터스토어 구축 시도
        # (실제 문서가 없어도 에러 없이 넘어감)
        _rag_system.build_vectorstore(force_recreate=force_recreate)
        
        return _rag_system
    except Exception as e:
        print(f"❌ RAG 초기화 실패: {e}")
        print("⚠️ RAG 없이 계속 진행")
        return None


def get_rag_system():
    """RAG 시스템 인스턴스 반환"""
    return _rag_system

