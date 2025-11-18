"""
RAG 시스템 - 한국어 최적화 (Render 512MB)
"""
import os
from pathlib import Path
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

_rag_system = None

class RAGSystem:
    def __init__(self, doc_dir: str = "rag_documents", cache_dir: str = "vectorstore_cache"):
        self.doc_dir = Path(doc_dir)
        self.cache_dir = Path(cache_dir)
        
        print("🤗 한국어 임베딩 모델 초기화...")
        print("   모델: jhgan/ko-sroberta-multitask")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",  # ← 한국어 최적!
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8,  # 메모리 절약
                'show_progress_bar': False
            },
            cache_folder="/tmp/hf_cache"
        )
        print("✅ 한국어 임베딩 초기화 완료")
        
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
        """벡터 DB 로드 (캐시 우선)"""
        
        # 캐시 로드
        if self.cache_dir.exists() and not force_recreate:
            try:
                print("📦 캐시된 벡터 DB 로드 중...")
                self.vectorstore = FAISS.load_local(
                    str(self.cache_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ 캐시 로드 완료! (임베딩 0회)")
                return
            except Exception as e:
                print(f"⚠️ 캐시 로드 실패: {e}")
        
        # 캐시 없으면 경고
        print("⚠️ 벡터 DB 캐시가 없습니다!")
        print("💡 로컬에서 create_cache_local.py를 실행하세요.")
        
        if os.getenv("RENDER"):
            print("🚨 Render에서는 캐시 필수입니다!")
            return
        
        # 로컬에서만 생성
        self._build_from_scratch()
    
    def _build_from_scratch(self):
        """새로 생성 (로컬 전용)"""
        documents = self.load_documents()
        if not documents:
            return
        
        all_splits = []
        for doc in documents:
            splits = self.text_splitter.split_text(doc["content"])
            for split in splits:
                all_splits.append({
                    "content": split,
                    "source": doc["source"]
                })
        
        print(f"📄 총 {len(all_splits)}개 청크")
        
        max_chunks = 400
        if len(all_splits) > max_chunks:
            all_splits = all_splits[:max_chunks]
        
        texts = [s["content"] for s in all_splits]
        metadatas = [{"source": s["source"]} for s in all_splits]
        
        print(f"🔄 벡터 DB 생성 중...")
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print("✅ 생성 완료!")
        
        # 저장
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(self.cache_dir))
            print(f"💾 저장 완료: {self.cache_dir}")
        except Exception as e:
            print(f"⚠️ 저장 실패: {e}")
    
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
            print(f"❌ 검색 오류: {e}")
            return []


def initialize_rag_system(force_recreate: bool = False):
    """RAG 초기화"""
    global _rag_system
    
    try:
        print("🚀 RAG 시스템 초기화 (한국어 최적화)")
        _rag_system = RAGSystem()
        _rag_system.build_vectorstore(force_recreate=force_recreate)
        return _rag_system
    except Exception as e:
        print(f"❌ RAG 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️ RAG 없이 계속 진행")
        return None

def get_rag_system():
    return _rag_system