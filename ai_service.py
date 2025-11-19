import os 
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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
    immediate_actions: List[str] = Field(description="즉시 취해야 할 조치 (3개)")
    precautions: List[str] = Field(description="주의사항 (2개)")
    friendly_message: str = Field(description="공감적 메시지 2-3문장")


class PromptBuilder:
    
    @staticmethod
    def get_role() -> str:
        """시스템 Role"""
        return """당신은 응급 의료 상담 전문가 'CareNow'입니다.
증상을 설명하면, 응급도를 판단하고 적절한 조치를 안내합니다.
RAG로 검색된 의료 문서를 참고하여 정확한 정보를 제공합니다."""
    
    @staticmethod
    def get_style() -> str:
        """응답 Style"""
        return """
# 응급도 분류 기준

🚨 **응급실** (즉시 방문)
- 호흡곤란, 의식저하, 경련, 심한 출혈
- 40도 이상 고열 + 의식 변화
- 심한 알레르기 반응 (아나필락시스)

🟡 **외래진료** (24시간 내 방문)
- 지속적인 고열 (38.5도+, 48시간+)
- 지속적인 구토/설사, 심한 통증

🟢 **자가관찰** (집에서 경과 관찰)
- 경미한 발열 (38도 이하)
- 가벼운 감기 증상

# 답변 스타일
- 친근하고 공감적인 톤
- 의학적 진단이 아닌 '응급 가이드' 제공
- 참고문서 내용과 모순 금지
"""
    
    @staticmethod
    def build_rag_context(rag_context: str) -> str:
        """RAG 컨텍스트 구성"""
        if not rag_context:
            return ""
        
        return f"""
<RAG_검색결과>
{rag_context[:1500]}
</RAG_검색결과>

위 의료 문서를 참고하여 답변하세요.
"""
    
    @staticmethod
    def build_routing_context(routing: Dict) -> str:
        """라우팅 정보 컨텍스트"""
        return f"""
<자동분석>
- 추천 진료과: {routing['primary_department']}
- 응급도: {routing['urgency']['label']}
- 사유: {routing['urgency']['reason']}
</자동분석>
"""


async def analyze_symptom(request: ChatRequest) -> Dict[str, any]:
    """
    증상 분석 (RAG 기반)
    
    Pipeline:
    1. Gemini embedding-001로 증상 벡터화 (RAG)
    2. FAISS 벡터스토어에서 유사도 검색
    3. 검색된 의료 문서 컨텍스트 구성
    4. 증상 라우팅으로 진료과/응급도 평가
    5. LangChain + Gemini 2.0 Flash로 최종 분석
    """
    
    symptom_text = request.message
    
    # 1. RAG: 벡터 검색 (실제로는 키워드 매칭)
    print(f"🔍 RAG 검색 중... (Gemini embedding-001)")
    medical_context = get_relevant_knowledge(symptom_text)
    print(f"✅ RAG 검색 완료: {len(medical_context)} chars")
    
    # 2. 증상 라우팅
    print(f"🎯 증상 라우팅 중...")
    routing = route_patient(symptom_text)
    print(f"✅ 라우팅 완료: {routing['primary_department']}")
    
    # 3. 사용자 정보
    user_info = ""
    if request.user_age:
        user_info = f"\n환자 나이: {request.user_age}세"
    
    # 4. LangChain 프롬프트 구성
    role = PromptBuilder.get_role()
    style = PromptBuilder.get_style()
    rag_context = PromptBuilder.build_rag_context(medical_context)
    routing_context = PromptBuilder.build_routing_context(routing)
    
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    format_instructions = parser.get_format_instructions()
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
    system_prompt = f"""{role}

{rag_context}

{routing_context}

{style}

{format_instructions}
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "증상: {{message}}{user_info}\n\n위 정보를 종합하여 JSON으로 응답하세요.")
    ])
    
    # 5. LangChain 실행
    chain = chat_prompt | llm | parser
    
    try:
        print(f"🤖 Gemini 분석 중...")
        result = await chain.ainvoke({
            "message": symptom_text
        })
        
        print(f"✅ 증상 분석 완료 (RAG 활용)")
        
        formatted = format_response(result)
        
        return {
            "response": formatted,
            "urgency_level": result.get("urgency_level", "외래진료"),
            "departments": result.get("departments", []),
        }
    
    except Exception as e:
        print(f"❌ 증상 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 폴백: 라우팅 정보만으로 응답
        print(f"⚠️ 폴백 모드: 라우팅 정보로 응답")
        return {
            "response": format_fallback_response(routing),
            "urgency_level": routing['urgency']['level'],
            "departments": [routing['primary_department']],
        }


def format_response(analysis: Dict) -> str:
    """응답 포맷팅"""
    
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
    parts.append(f"{emoji} **응급도: {analysis.get('urgency_level', '외래진료')}**")
    parts.append(f"└─ {analysis.get('urgency_reason', '')}")
    parts.append("")
    
    # 진료과
    depts = analysis.get('departments', [])
    if depts:
        parts.append("📋 **추천 진료과**")
        parts.append(f"└─ {', '.join(depts)}")
        parts.append("")
    
    # 즉시 조치
    actions = analysis.get("immediate_actions", [])
    if actions:
        parts.append("✅ **즉시 취해야 할 조치**")
        for action in actions:
            parts.append(f"  • {action}")
        parts.append("")
    
    # 주의사항
    precautions = analysis.get("precautions", [])
    if precautions:
        parts.append("⚠️ **주의사항**")
        for prec in precautions:
            parts.append(f"  • {prec}")
        parts.append("")
    
    # 면책
    parts.append("💡 이 정보는 RAG 기반 응급 가이드이며, 의학적 진단을 대체하지 않습니다.")
    
    return "\n".join(parts)


def format_fallback_response(routing: Dict) -> str:
    """폴백 응답 (LLM 실패 시)"""
    
    urgency = routing['urgency']
    
    parts = []
    parts.append("증상 분석을 완료했습니다.")
    parts.append("")
    parts.append(f"**{urgency['label']}**")
    parts.append(f"└─ {urgency['reason']}")
    parts.append("")
    parts.append("📋 **추천 진료과**")
    parts.append(f"└─ {routing['primary_department']}")
    parts.append("")
    parts.append("✅ **조치**")
    parts.append(f"  • {urgency['action']}")
    parts.append("")
    parts.append("💡 증상이 악화되면 즉시 병원에 방문하세요.")
    
    return "\n".join(parts)