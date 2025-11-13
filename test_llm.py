# test_llm.py도 업데이트
import asyncio
from dotenv import load_dotenv
from ai_service import classify_symptom, generate_response
from models import ChatRequest

load_dotenv()

async def test_rices():
    print("=" * 50)
    print("🧪 RICES 기반 AI 서비스 테스트")
    print("=" * 50)
    
    # 테스트 1: 의도 분류
    request = ChatRequest(
        message="아이가 열이 38도 있고 기침을 해요",
        user_age=5
    )
    
    intent = await classify_symptom(request)
    print(f"\n✅ 의도 분류:\n{intent}")
    
    # 테스트 2: 응답 생성
    response = await generate_response(
        request=request,
        intent=intent,
        rag_results=[],
        facilities=None
    )
    
    print(f"\n✅ 응답:\n{response['response']}")
    print(f"\n✅ 핵심 조치:\n{response['key_points']}")

if __name__ == "__main__":
    asyncio.run(test_rices())