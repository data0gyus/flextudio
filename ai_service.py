"""
CareNow AI Service
RAG(Retrieval-Augmented Generation) 기반 증상 분석
LangChain + Gemini 2.0 Flash
"""

import os 
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from models import ChatRequest
from medical_knowledge import get_relevant_knowledge
from symptom_routing import route_patient

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# 응답 스키마 정의 
class ResponseSchema(BaseModel):
    """증상 분석 응답 스키마"""
    urgency_level: str = Field(description="응급도: 응급실/외래진료/자가관찰")
    urgency_reason: str = Field(description="판단 근거 1-2문장")
    departments: List[str] = Field(description="추천 진료과 리스트")
    immediate_actions: List[str] = Field(description="즉시 취해야 할 조치 (3개 이상)")
    precautions: List[str] = Field(description="주의사항 (2개 이상)")
    friendly_message: str = Field(description="공감적 메시지 2-3문장")


async def analyze_symptom(request: ChatRequest) -> Dict[str, any]:
    """
    증상 분석 (RAG 기반)
    
    Pipeline:
    1. 증상 라우팅으로 진료과/응급도 자동 평가
    2. RAG 검색으로 관련 의료 지식 추출
    3. LangChain + Gemini 2.0 Flash로 최종 종합 분석
    """
    
    symptom_text = request.message
    
    # 1. 증상 라우팅 (진료과 + 응급도 자동 평가)
    print(f"🎯 증상 라우팅 중...")
    routing = route_patient(symptom_text)
    print(f"✅ 라우팅: {routing['primary_department']} / {routing['urgency']['level']}")
    
    # 2. RAG: 관련 의료 지식 검색
    print(f"🔍 RAG 검색 중...")
    medical_context = get_relevant_knowledge(symptom_text)
    has_medical_context = len(medical_context) > 100
    print(f"✅ RAG: {len(medical_context)} chars (매칭: {has_medical_context})")
    
    # 3. 사용자 정보
    user_info = ""
    if request.user_age:
        user_info = f"\n환자 나이: {request.user_age}세"
    
    # 4. 프롬프트 구성
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    format_instructions = parser.get_format_instructions()
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
    # RAG 컨텍스트
    rag_section = ""
    if has_medical_context:
        rag_section = f"""
<의료_지식_검색_결과>
{medical_context[:1500]}
</의료_지식_검색_결과>

위 의료 지식을 반드시 참고하여 구체적인 조치사항을 제공하세요.
"""
    
    system_prompt = f"""당신은 소아 응급 의료 상담 전문가 'CareNow'입니다.

# 자동 분석 결과 (반드시 참고!)
- 추천 진료과: {routing['primary_department']}
- 응급도 평가: {routing['urgency']['label']}
- 평가 근거: {routing['urgency']['reason']}

{rag_section}

# 응급도 분류 기준

🔴 응급실 (즉시 방문)
- 호흡곤란, 의식저하, 경련, 심한 출혈
- 40도 이상 고열 + 의식 변화
- 심한 알레르기 반응 (아나필락시스)

🟡 외래진료 (24시간 내 방문)
- 지속적인 고열 (38.5도+, 48시간+)
- 지속적인 구토/설사, 심한 통증

🟢 자가관찰 (집에서 경과 관찰)
- 경미한 발열 (38도 이하)
- 가벼운 감기 증상

# 중요 규칙
1. 자동 분석 결과를 최대한 반영하되, 더 위험하다고 판단되면 응급도를 높이세요
2. 의료 지식이 검색되었다면 구체적인 응급처치 방법을 포함하세요
3. immediate_actions는 반드시 3개 이상, 구체적으로 작성하세요
4. precautions는 반드시 2개 이상 작성하세요
5. friendly_message는 따뜻하고 공감적으로 작성하세요

{format_instructions}
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "증상: {{message}}{user_info}\n\n위 증상을 분석하여 JSON으로 응답하세요.")
    ])
    
    # 5. LangChain 실행
    chain = chat_prompt | llm | parser
    
    try:
        print(f"🤖 Gemini 분석 중...")
        result = await chain.ainvoke({
            "message": symptom_text
        })
        
        print(f"✅ 분석 완료: {result['urgency_level']}")
        
        # 포맷팅 (마크다운 제거)
        formatted = format_response(result)
        
        return {
            "response": formatted,
            "urgency_level": result.get("urgency_level", "외래진료"),
            "departments": result.get("departments", []),
        }
    
    except Exception as e:
        print(f"❌ LLM 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 폴백: 라우팅 정보로 응답
        print(f"⚠️ 폴백 모드")
        return {
            "response": format_fallback_response(routing, medical_context),
            "urgency_level": map_urgency_level(routing['urgency']['level']),
            "departments": [routing['primary_department']],
        }


def map_urgency_level(level: str) -> str:
    """라우팅 urgency level을 한글로 매핑"""
    mapping = {
        "emergency": "응급실",
        "urgent": "외래진료",
        "observation": "자가관찰"
    }
    return mapping.get(level, "외래진료")


def format_response(analysis: Dict) -> str:
    """응답 포맷팅 (마크다운 제거)"""
    
    urgency_emoji = {
        "응급실": "🔴",
        "외래진료": "🟡",
        "자가관찰": "🟢",
    }
    
    emoji = urgency_emoji.get(analysis.get("urgency_level", "외래진료"), "💡")
    
    parts = []
    
    # 공감 메시지
    parts.append(analysis.get("friendly_message", "증상 분석을 완료했습니다."))
    parts.append("")
    
    # 응급도
    parts.append(f"{emoji} 응급도: {analysis.get('urgency_level', '외래진료')}")
    parts.append(f"└─ {analysis.get('urgency_reason', '')}")
    parts.append("")
    
    # 진료과
    depts = analysis.get('departments', [])
    if depts:
        parts.append("📋 추천 진료과")
        parts.append(f"└─ {', '.join(depts)}")
        parts.append("")
    
    # 즉시 조치
    actions = analysis.get("immediate_actions", [])
    if actions:
        parts.append("✅ 즉시 취해야 할 조치")
        for action in actions:
            parts.append(f"  • {action}")
        parts.append("")
    
    # 주의사항
    precautions = analysis.get("precautions", [])
    if precautions:
        parts.append("⚠️ 주의사항")
        for prec in precautions:
            parts.append(f"  • {prec}")
        parts.append("")
    
    # 면책
    parts.append("💡 이 정보는 응급 가이드이며, 의학적 진단을 대체하지 않습니다.")
    
    return "\n".join(parts)


def format_fallback_response(routing: Dict, medical_context: str) -> str:
    """폴백 응답 (LLM 실패 시 - 향상된 버전)"""
    
    urgency = routing['urgency']
    dept = routing['primary_department']
    
    # 의료 지식 활용
    has_context = len(medical_context) > 100
    
    urgency_emoji = {
        "emergency": "🔴",
        "urgent": "🟡",
        "observation": "🟢"
    }
    
    emoji = urgency_emoji.get(urgency['level'], "💡")
    
    parts = []
    parts.append("증상 분석을 완료했습니다.")
    parts.append("")
    parts.append(f"{emoji} 응급도: {urgency['label']}")
    parts.append(f"└─ {urgency['reason']}")
    parts.append("")
    parts.append(f"📋 추천 진료과")
    parts.append(f"└─ {dept}")
    parts.append("")
    parts.append("✅ 즉시 취해야 할 조치")
    parts.append(f"  • {urgency['action']}")
    
    if has_context:
        parts.append("  • 의료 지식베이스를 참고하여 적절한 응급처치를 하세요")
        parts.append("  • 증상이 악화되면 즉시 병원 방문")
    else:
        parts.append("  • 증상을 관찰하고 악화되면 병원 방문")
        parts.append("  • 불안하면 전문의 상담 권장")
    
    parts.append("")
    parts.append("⚠️ 주의사항")
    parts.append("  • 자가 판단만으로 치료하지 마세요")
    parts.append("  • 증상 변화를 주의 깊게 관찰하세요")
    parts.append("")
    parts.append("💡 증상이 지속되거나 악화되면 반드시 병원에 방문하세요.")
    
    return "\n".join(parts)