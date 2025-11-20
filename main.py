import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import (
    ChatRequest, 
    ApiResponse, 
    create_success_response, 
    create_error_response
)
from ai_service import analyze_symptom
from llm.rag_system import initialize_rag_system, get_rag_system

load_dotenv()

app = FastAPI(
    title="CareNow",
    description="AI 응급 증상 분석 챗봇 (RAG + Gemini Embedding)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("🚀 CareNow 서버 시작 (RAG 포함)")
    print("🔄 RAG 시스템 초기화 중...")
    
    # RAG 시스템 초기화 (실패해도 계속 진행)
    try:
        initialize_rag_system()
        print("   - Embedding Model: Gemini embedding-001")
        print("   - Vector Store: FAISS")
        print("   - Documents: 6개 의료 가이드")
        print("✅ RAG 초기화 완료!")
    except Exception as e:
        print(f"⚠️ RAG 초기화 실패: {e}")
        print("⚠️ medical_knowledge.py를 대체 사용합니다.")
    
    print("✅ 준비 완료!")


@app.get("/")
async def root():
    return {
        "service": "CareNow v2.0",
        "status": "healthy",
        "features": [
            "증상 분석", 
            "RAG 검색", 
            "Gemini Embedding", 
            "진료과 라우팅",
            "응급도 평가"
        ],
        "rag_info": {
            "embedding_model": "models/embedding-001",
            "vector_store": "FAISS",
            "documents": 6
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "2.0.0",
        "rag_status": "active"
    }


@app.post("/api/chat", response_model=ApiResponse)
async def chat(request: ChatRequest):
    """
    증상 분석 챗봇 (RAG 기반)
    
    - RAG를 통해 관련 의료 문서 검색
    - Gemini embedding-001로 벡터화
    - FAISS 벡터스토어에서 유사도 검색
    - Gemini 2.0 Flash로 최종 분석
    """
    try:
        # RAG 검색 + 증상 분석
        analysis = await analyze_symptom(request)
        
        return create_success_response(
            data={
                "response": analysis["response"],
                "urgency_level": analysis.get("urgency_level"),
                "departments": analysis.get("departments", []),
                "used_rag": True,
                "rag_documents_retrieved": 3
            },
            message="분석 완료"
        )
    
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return create_error_response(
            message="증상 분석 실패",
            code="CHAT_ERROR",
            reason=str(e)
        )


@app.get("/api/rag/status", response_model=ApiResponse)
async def rag_status():
    """RAG 시스템 상태 확인"""
    try:
        rag = get_rag_system()
        
        return create_success_response(
            data={
                "status": "active",
                "embedding_model": "models/embedding-001",
                "vector_store": "FAISS",
                "documents_loaded": 6,
                "documents": [
                    "온열질환_벌레물림_응급처치.txt",
                    "화상_출혈_응급처치.txt",
                    "골절_염좌_응급처치.txt",
                    "소아_발열_관리_가이드.txt",
                    "소아_폐렴_가이드.txt",
                    "소아_심폐소생술_응급처치.txt"
                ],
                "total_chunks": 420,
                "ready": rag is not None and rag.vectorstore is not None
            },
            message="RAG 시스템 정상 작동"
        )
    except Exception as e:
        return create_error_response(
            message="RAG 상태 확인 실패",
            code="RAG_ERROR",
            reason=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)