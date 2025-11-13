"""
응급처치 문서 RAG 시스템
- rag_documents 폴더의 PDF 파일들을 벡터화
- 없으면 그냥 넘어감 (일반 챗봇으로 동작)
"""

import os
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional

# SQLite 버전 문제 해결
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 경로 설정
BASE_DIR = Path(__file__).parent.parent  # 프로젝트 루트
RAG_DOCS_DIR = BASE_DIR / "rag_documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# 임베딩 모델 캐싱
_cached_embeddings = None

def get_embeddings():
    """임베딩 모델 반환 (캐싱)"""
    global _cached_embeddings
    
    if _cached_embeddings is None:
        print("📦 임베딩 모델 로딩 중...")
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sroberta-nli',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 임베딩 모델 로드 완료")
    
    return _cached_embeddings


class RAGSystem:
    """응급처치 문서 RAG 시스템"""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
    
    def extract_text_from_pdf(self, file_path: Path) -> List[Document]:
        """PDF에서 텍스트 추출"""
        documents = []
        filename = file_path.name
        
        print(f"   📄 {filename} 처리 중...")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text and text.strip():
                        documents.append(Document(
                            page_content=text.strip(),
                            metadata={
                                "source": f"{filename} - 페이지 {page_num}",
                                "file": filename,
                                "page": page_num
                            }
                        ))
            
            print(f"      ✅ {len(documents)}개 페이지 추출")
            return documents
        
        except Exception as e:
            print(f"      ❌ 실패: {e}")
            return []
    
    def load_documents(self) -> List[Document]:
        """rag_documents 폴더의 모든 PDF 로드"""
        
        # 폴더가 없으면 생성
        RAG_DOCS_DIR.mkdir(exist_ok=True)
        
        pdf_files = list(RAG_DOCS_DIR.glob("*.pdf"))
        
        if not pdf_files:
            print(f"📂 {RAG_DOCS_DIR}에 PDF 파일이 없습니다")
            return []
        
        print(f"📚 {len(pdf_files)}개 PDF 파일 발견")
        
        all_documents = []
        for pdf_file in pdf_files:
            docs = self.extract_text_from_pdf(pdf_file)
            all_documents.extend(docs)
        
        print(f"✅ 총 {len(all_documents)}개 문서 로드 완료")
        return all_documents
    
    def create_vectorstore(self, documents: List[Document]):
        """벡터스토어 생성"""
        
        if not documents:
            print("⚠️ 문서가 없어 벡터스토어를 생성하지 않습니다")
            return
        
        print("🔨 벡터스토어 생성 중...")
        
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"   - {len(split_docs)}개 청크로 분할")
        
        # 벡터스토어 생성
        VECTORSTORE_DIR.mkdir(exist_ok=True)
        
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=str(VECTORSTORE_DIR),
            collection_name="emergency_docs"
        )
        
        print("✅ 벡터스토어 생성 완료")
    
    def load_vectorstore(self) -> bool:
        """기존 벡터스토어 로드"""
        
        if not VECTORSTORE_DIR.exists():
            return False
        
        try:
            print("🔍 기존 벡터스토어 로드 중...")
            
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(VECTORSTORE_DIR),
                collection_name="emergency_docs"
            )
            
            try:
                count = self.vectorstore._collection.count()
                print(f"✅ 벡터스토어 로드 완료 ({count}개 문서)")
            except:
                print("✅ 벡터스토어 로드 완료")
            
            return True
        
        except Exception as e:
            print(f"⚠️ 벡터스토어 로드 실패: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """문서 검색"""
        
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "file": doc.metadata.get("file", ""),
                    "page": doc.metadata.get("page", "")
                }
                for doc in results
            ]
        
        except Exception as e:
            print(f"검색 실패: {e}")
            return []


# 전역 인스턴스
_rag_instance = None

def initialize_rag_system(force_recreate: bool = False) -> Optional[RAGSystem]:
    """RAG 시스템 초기화"""
    global _rag_instance
    
    print("\n" + "="*60)
    print("🚀 RAG 시스템 초기화")
    print("="*60)
    
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    
    # 기존 벡터스토어 로드 시도
    if not force_recreate and _rag_instance.load_vectorstore():
        print("="*60)
        return _rag_instance
    
    # 새로 생성
    print("📚 새로운 벡터스토어 생성")
    documents = _rag_instance.load_documents()
    
    if documents:
        _rag_instance.create_vectorstore(documents)
    else:
        print("⚠️ RAG 문서가 없습니다. 일반 챗봇 모드로 동작합니다.")
        print(f"💡 문서 추가: {RAG_DOCS_DIR} 폴더에 PDF 파일을 넣으세요")
    
    print("="*60 + "\n")
    return _rag_instance


def get_rag_system() -> Optional[RAGSystem]:
    """RAG 시스템 반환"""
    global _rag_instance
    
    if _rag_instance is None:
        initialize_rag_system()
    
    return _rag_instance


def search_documents(query: str, k: int = 3) -> List[Dict]:
    """문서 검색 (간편 함수)"""
    rag = get_rag_system()
    
    if rag:
        return rag.search(query, k)
    
    return []


if __name__ == "__main__":
    # 테스트
    print("=== RAG 시스템 테스트 ===\n")
    
    rag = initialize_rag_system(force_recreate=False)
    
    if rag and rag.vectorstore:
        # 검색 테스트
        results = rag.search("고열 응급처치", k=2)
        
        if results:
            print("\n검색 결과:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result['source']}]")
                print(f"   {result['content'][:100]}...")
        else:
            print("\n검색 결과가 없습니다")
    else:
        print("\nRAG 문서가 없습니다")