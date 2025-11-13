import os
from pathlib import Path
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber

# 전역 변수
_rag_system = None

class RAGSystem:
    def __init__(self, pdf_dir: str = "rag_documents", persist_dir: str = "vectorstore"):
        self.pdf_dir = Path(pdf_dir)
        self.persist_dir = Path(persist_dir)
        
        # 임베딩 모델 (경량 모델 사용!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 텍스트 분할기
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.vectorstore = None
    
    def load_pdfs(self) -> List[Dict[str, str]]:
        """PDF 파일들을 읽어서 텍스트 추출"""
        documents = []
        
        if not self.pdf_dir.exists():
            print(f"⚠️ PDF 디렉토리가 없습니다: {self.pdf_dir}")
            return documents
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ PDF 파일이 없습니다: {self.pdf_dir}")
            return documents
        
        for pdf_path in pdf_files:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    
                    if text.strip():
                        documents.append({
                            "content": text,
                            "source": pdf_path.name
                        })
                        print(f"✅ 로드 완료: {pdf_path.name}")
            
            except Exception as e:
                print(f"❌ PDF 로드 실패 ({pdf_path.name}): {e}")
        
        return documents
    
    def build_vectorstore(self, force_recreate: bool = False):
        """벡터 DB 생성 (FAISS 사용)"""
        
        # FAISS는 persist 안 함 (메모리만 사용)
        
        # PDF 로드
        documents = self.load_pdfs()
        
        if not documents:
            print("⚠️ 로드할 문서가 없습니다.")
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
        
        print(f"📄 총 {len(all_splits)}개 청크 생성")
        
        # 벡터 DB 생성 (FAISS)
        texts = [s["content"] for s in all_splits]
        metadatas = [{"source": s["source"]} for s in all_splits]
        
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ 벡터 DB 생성 완료 (FAISS)!")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """유사도 검색"""
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

# ========================================
# 전역 함수
# ========================================

def initialize_rag_system(force_recreate: bool = False):
    """RAG 시스템 초기화"""
    global _rag_system
    
    try:
        _rag_system = RAGSystem()
        _rag_system.build_vectorstore(force_recreate=force_recreate)
        return _rag_system
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")
        return None

def get_rag_system():
    """현재 RAG 시스템 반환"""
    return _rag_system