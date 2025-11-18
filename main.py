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
    description="AI 응급 증상 분석 + RAG",
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
    initialize_rag_system()
    print("✅ 준비 완료!")


@app.get("/")
async def root():
    return {
        "service": "CareNow v2.0",
        "status": "healthy",
        "features": ["증상 분석", "RAG", "512MB 최적화"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ApiResponse)
async def chat(request: ChatRequest):
    """증상 분석 챗봇 (RAG 포함)"""
    try:
        # RAG 검색
        rag_results = []
        rag = get_rag_system()
        if rag:
            rag_results = rag.search(request.message, k=3)
        
        # 증상 분석
        analysis = await analyze_symptom(request, rag_results)
        
        return create_success_response(
            data={
                "response": analysis["response"],
                "urgency_level": analysis.get("urgency_level"),
                "used_rag": analysis.get("used_rag", False)
            },
            message="분석 완료"
        )
    
    except Exception as e:
        print(f"❌ 오류: {e}")
        return create_error_response(
            message="증상 분석 실패",
            code="CHAT_ERROR",
            reason=str(e)
        )


@app.get("/api/rag/status", response_model=ApiResponse)
async def rag_status():
    """RAG 상태"""
    try:
        rag = get_rag_system()
        if rag and rag.vectorstore:
            return create_success_response(
                data={"status": "active", "ready": True},
                message="RAG 정상"
            )
        else:
            return create_success_response(
                data={"status": "no_documents", "ready": False},
                message="RAG 문서 없음"
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