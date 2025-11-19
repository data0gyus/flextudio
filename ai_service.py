"""
CareNow AI Service
RAG(Retrieval-Augmented Generation) 기반 증상 분석
LangChain + Gemini 2.0 Flash
"""

import os 
from pydantic import BaseModel, Field
from typing import List, Dict, Any
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


async def analyze_symptom(request: ChatRequest) -> Dict[str, Any]:
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
    
    system_prompt = f"""당신은 일상 응급 상황에 대응하는 1차 의료 상담 전문가 'CareNow'입니다.

# 자동 분석 결과 (반드시 참고!)
- 추천 진료과: {routing['primary_department']}
- 응급도 평가: {routing['urgency']['label']}
- 평가 근거: {routing['urgency']['reason']}

{rag_section}

# 역할과 대상
- 사용자는 영유아, 소아, 성인 모두 포함될 수 있습니다.
- 환자 나이가 주어지면 나이에 맞는 표현과 진료과를 선택하세요.
- 환자 나이가 없으면 “일반 성인 기준”으로 판단하되, 명백히 아이 관련 표현(예: 아이, 아기, 우리 애)이 있으면 소아 기준으로 판단하세요.

# 응급도 분류 기준

🔴 응급실 (즉시 방문)
- 호흡곤란, 의식저하, 경련, 심한 출혈
- 40도 이상 고열 + 의식 변화
- 심한 알레르기 반응 (아나필락시스)
- 머리 외상 후 의식 소실, 구토 반복

🟡 외래진료 (24시간 내 방문)
- 지속적인 고열 (38.5도+, 48시간+)
- 지속적인 구토/설사, 심한 통증
- 벌에 쏘인 후 국소 부위가 심하게 붓거나 통증이 심한 경우

🟢 자가관찰 (집에서 경과 관찰)
- 경미한 발열 (38도 이하)
- 가벼운 감기 증상
- 가벼운 타박상, 국소 통증, 경미한 피부 증상

# 진료과 선택 가이드
- 전신 증상(발열, 몸살, 기침 등): 내과 / 가정의학과 / (소아라면 소아청소년과)
- 피부 증상(발진, 벌에 쏘임, 가려움 등): 피부과 / 알레르기내과
- 외상(넘어짐, 타박상, 골절 의심 등): 정형외과 / 응급의학과
- 머리·신경 증상(두통, 어지러움, 경련 등): 신경과 / 신경외과 / 응급의학과
- 눈 증상: 안과
- 귀·코·목 증상: 이비인후과

departments 필드에는 위 가이드에 따라 **최소 1개, 최대 3개**까지 넣되,
첫 번째 항목은 자동 라우팅 결과(primary_department)를 우선 반영하세요.

# 응급처치 작성 가이드
- immediate_actions 항목에는 **집에서 당장 할 수 있는 구체적인 행동**을 적어야 합니다.
- 예시:
  - "상처 부위를 흐르는 물로 10분 이상 씻기"
  - "벌에 쏘인 부위에 깨끗한 수건을 감싼 얼음으로 10~15분 냉찜질하기"
  - "출혈 부위를 깨끗한 거즈나 수건으로 5~10분간 강하게 눌러 지혈하기"
  - "해당 부위를 심장보다 높게 올려두기"
- 단순히 "병원에 가세요" 같은 문장은 immediate_actions에 넣지 말고,
  병원 방문 권고는 precautions 또는 friendly_message에 포함하세요.

# 중요 규칙
1. 자동 분석 결과를 최대한 반영하되, 더 위험하다고 판단되면 응급도를 상향 조정하세요.
2. 의료 지식이 검색되었다면 구체적인 응급처치 방법을 2개 이상 포함하세요.
3. immediate_actions는 반드시 3개 이상, **"무엇을, 어떻게, 얼마나"** 수준으로 구체적으로 작성하세요.
4. precautions는 반드시 2개 이상 작성하고, 병원 방문이 필요한 상황을 명확히 써주세요.
5. friendly_message는 따뜻하고 공감적으로 2~3문장 작성하되, 불필요한 공포를 주지 않도록 하세요.

{format_instructions}
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "증상: {message}\n\n위 증상을 분석하여 JSON으로 응답하세요.")
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