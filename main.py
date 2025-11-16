import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import (ChatRequest, ApiResponse, create_success_response, create_error_response)
from ai_service import classify_symptom, generate_response

from llm.rag_system import initialize_rag_system, get_rag_system

# 환경변수 로드
load_dotenv()

# FastAPI 앱
app = FastAPI(
    title="CareNow 챗봇 API",
    description="돌봄 공백 SOS 챗봇 (Gemini 2.0 Flash)",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 RAG 초기화
@app.on_event("startup")
async def startup_event():
    print("🚀 서버 시작 중...")
    initialize_rag_system()
    print("✅ 서버 준비 완료!")


# 헬스체크
@app.get("/")
async def root():
    return {
        "service": "CareNow 챗봇 API",
        "status": "healthy",
        "version": "1.0.0",
        "llm": "Gemini 2.0 Flash"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 챗봇 엔드포인트
@app.post("/api/chat", response_model=ApiResponse)
async def chat(request: ChatRequest):
    """
    챗봇 대화 엔드포인트
    1. 의도 분류 (Gemini)
    2. RAG 검색
    3. 시설 API 호출 (필요시)
    4. 응답 생성 (Gemini)
    """
    try:
        # 1. 의도 분류
        intent = await classify_symptom(request)
        
        # 2. RAG 검색
        rag_results = []
        rag = get_rag_system()
        if rag:
            rag_results = rag.search(request.message, k=3)
        
        # 3. 시설 API 호출 (필요시)
        facilities = None
        # if intent.get("needs_facility") and request.user_location:
        #     facilities = await get_facilities(
        #         facility_type=intent.get("facility_type"),
        #         lat=request.user_location.latitude,
        #         lng=request.user_location.longitude
        #     )
        
        # 4. 응답 생성
        response = await generate_response(
            request=request,
            intent=intent,
            rag_results=rag_results,
            facilities=facilities
        )
        
        return create_success_response(
            data=[response],
            message="응답 생성 성공"
        )
    
    except Exception as e:
        print(f"챗봇 오류: {e}")
        return create_error_response(
            message="AI 응답 생성 실패",
            code="CHAT_ERROR",
            reason=str(e)
        )

# RAG 관리 엔드포인트
@app.get("/api/rag/status", response_model=ApiResponse)
async def rag_status():
    """RAG 시스템 상태 확인"""
    try:
        rag = get_rag_system()
        if rag and rag.vectorstore:
            try:
                count = rag.vectorstore._collection.count()
                return create_success_response(
                    data={
                        "status": "active",
                        "document_count": count,
                        "ready": True
                    },
                    message="RAG 시스템이 정상 작동 중입니다"
                )
            except:
                return create_success_response(
                    data={"status": "active", "ready": True},
                    message="RAG 시스템이 정상 작동 중입니다"
                )
        else:
            return create_success_response(
                data={"status": "no_documents", "ready": False},
                message="RAG 문서가 없습니다"
            )
    except Exception as e:
        return create_error_response(
            message="RAG 시스템 상태 확인 실패",
            code="RAG_STATUS_ERROR",
            reason=str(e)
        )

@app.post("/api/rag/reload", response_model=ApiResponse)
async def rag_reload():
    """RAG 시스템 재로드"""
    try:
        rag = initialize_rag_system(force_recreate=False)
        if rag:
            return create_success_response(message="RAG 시스템 재로드 완료")
        else:
            return create_success_response(
                data={"status": "no_documents"},
                message="로드할 문서가 없습니다"
            )
    except Exception as e:
        return create_error_response(
            message="RAG 시스템 재로드 실패",
            code="RAG_RELOAD_ERROR",
            reason=str(e)
        )

# 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
