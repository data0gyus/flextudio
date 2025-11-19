"""
CareNow AI Service
RAG(Retrieval-Augmented Generation) 기반 증상 분석
LangChain + Gemini 2.0 Flash

균형잡힌 버전: 자가관찰 45% / 외래진료 45% / 응급실 10%
"""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from models import ChatRequest
from medical_knowledge import get_relevant_knowledge
from symptom_routing import route_patient

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4,
)


# 응답 스키마 정의
class ResponseSchema(BaseModel):
    """증상 분석 응답 스키마"""
    urgency_level: str = Field(description="응급도: 응급실/외래진료/자가관찰")
    urgency_reason: str = Field(description="판단 근거 1-2문장")
    departments: List[str] = Field(description="추천 진료과 리스트")
    immediate_actions: List[str] = Field(description="즉시 취해야 할 조치 (3개 이상, 문장 단위)")
    precautions: List[str] = Field(description="주의사항 (2개 이상, 문장 단위)")
    friendly_message: str = Field(description="공감적 메시지 2-3문장")


async def analyze_symptom(request: ChatRequest) -> Dict[str, Any]:
    """
    증상 분석 (RAG 기반)

    Pipeline:
    1. 증상 라우팅으로 진료과/응급도 자동 평가 (참고용)
    2. RAG 검색으로 관련 의료 지식 추출
    3. LangChain + Gemini 2.0 Flash로 최종 종합 분석
    """

    symptom_text = request.message

    # 1. 증상 라우팅
    print("🎯 증상 라우팅 중...")
    routing = route_patient(symptom_text)
    print(f"✅ 라우팅: {routing['primary_department']} / {routing['urgency']['level']}")

    # 2. RAG: 관련 의료 지식 검색
    print("🔍 RAG 검색 중...")
    medical_context = get_relevant_knowledge(symptom_text)
    has_medical_context = len(medical_context) > 100
    print(f"✅ RAG: {len(medical_context)} chars (매칭: {has_medical_context})")

    # 3. 프롬프트 구성
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    format_instructions = parser.get_format_instructions()
    # 중괄호 이스케이프
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")

    # RAG 컨텍스트
    rag_section = ""
    if has_medical_context:
        rag_section = f"""
<의료_지식_검색_결과>
{medical_context[:1500]}
</의료_지식_검색_결과>

위 의료 지식의 내용을 적극적으로 활용하여,
응급처치 방법과 주의사항을 단계별로 상세히 설명하세요.
"""

    system_prompt = f"""당신은 일상 응급 상황에 대응하는 1차 의료 상담 전문가 'CareNow'입니다.

# 자동 분석 참고 정보
- AI 추천 진료과: {routing['primary_department']}
- AI 응급도 평가: {routing['urgency']['label']}
- AI 평가 근거: {routing['urgency']['reason']}

{rag_section}

# 역할과 대상
- 사용자는 영유아, 소아, 성인 모두 포함될 수 있습니다.
- 환자 나이가 주어지면 나이에 맞는 표현과 진료과를 선택하세요.
- 환자 나이가 없으면 "일반 성인 기준"으로 판단하되,
  명백히 아이 관련 표현(예: 아이, 아기, 우리 애)이 있으면 소아 기준으로 판단하세요.

# 응급도 분류 기준 (균형잡힌 접근)

🔴 응급실 (즉시 방문) - 약 10%
- 호흡곤란, 의식저하, 경련, 심한 출혈
- 40도 이상 고열 + 의식 변화
- 심한 가슴통증(쥐어짜는 느낌), 심한 알레르기 반응
- 머리 외상 후 의식 소실, 반복적인 구토

🟡 외래진료 (24~48시간 내 방문) - 약 45%
- 고열 지속 (39도 이상, 2~3일 이상)
- 참기 힘든 통증 (너무 아프다, 못 참겠다 등)
- 하루 종일 계속되는 구토/설사로 탈수가 걱정되는 경우
- 골절이 의심되거나, 심한 타박상·관절 부종 등
- 증상이 점점 심해지거나, 보호자가 보기에도 걱정되는 경우

🟢 자가관찰 (집에서 경과 관찰) - 약 45%
- 37~38도 정도의 가벼운 발열
- 참을 만한 정도의 두통·복통
- 가벼운 콧물·기침·코막힘
- 일시적인 소화 불량, 피로감, 몸살 기운
- "머리가 아파요", "배가 살짝 아파요", "미열이 있어요"와 같은 일상적인 증상

# 진료과 선택 가이드
- 전신 증상(발열, 몸살, 기침 등): 내과 / 가정의학과 / (소아라면 소아청소년과)
- 피부 증상: 피부과 / 알레르기내과
- 외상: 정형외과 / 응급의학과
- 머리·신경 증상: 신경과 / 신경외과
- 눈: 안과
- 귀·코·목: 이비인후과

departments 필드에는 위 가이드에 따라 최소 1개, 최대 3개까지 넣되,
첫 번째 항목은 자동 라우팅 결과(primary_department)를 우선 반영하세요.

# 응급처치(immediate_actions) 작성 가이드 - 매우 중요!
- **집에서 바로 할 수 있는 구체적인 행동**을 단계처럼 설명하세요.
- 각 항목은 **완결된 문장**으로 쓰고, **최소 20자 이상**이 되도록 자세히 적으세요.
- 가능하면 "무엇을, 어떻게, 얼마나, 왜"를 포함하세요.
  좋은 예시:
  - "조용하고 어두운 곳에서 30분 이상 휴식을 취하면서, 스마트폰 사용을 잠시 중단하도록 안내합니다."
  - "물을 한 번에 많이 마시기보다는, 10~15분 간격으로 한 컵씩 천천히 마시게 하여 탈수를 예방합니다."
  - "열이 38도 이상이면서 힘들어한다면, 체중에 맞는 해열제를 복용하고 30분~1시간 뒤 다시 체온을 확인합니다."
  - "벌에 쏘인 부위를 깨끗한 물로 씻은 후, 깨끗한 수건으로 감싼 얼음주머니를 10~15분간 대주면 붓기와 통증이 완화됩니다."
- 나쁜 예시:
  - "병원에 가세요" ← 이건 precautions에
  - "관찰하세요" ← 너무 추상적
  - "휴식" ← 구체적이지 않음
- 단순히 "병원에 가세요" 같은 문장은 immediate_actions에 넣지 말고,
  병원 방문 권고는 precautions 또는 friendly_message에 포함하세요.
- 응급도가 '자가관찰'이어도, 집에서 할 수 있는 구체적인 조치를 **최소 3개 이상** 작성해야 합니다.

# 주의사항(precautions) 작성 가이드
- 각 항목은 **한 문장 이상, 20자 이상**의 문장 형태여야 합니다.
- "이런 경우에는 바로 병원에 가야 한다"는 기준을 분명하게 써 주세요.
  좋은 예시:
  - "통증이 점점 심해지거나, 2~3일 이상 좋아지지 않으면 가까운 병·의원 진료를 꼭 권장합니다."
  - "열이 39도 이상으로 다시 오르거나, 아이가 축 늘어지고 잘 반응하지 않으면 응급실 방문을 고려해야 합니다."
  - "쏘인 곳이 크게 부어오르면서 호흡곤란, 심한 두드러기, 어지러움 등의 알레르기 반응이 나타나면 즉시 119에 신고하거나 가장 가까운 응급실로 가세요."

# 공감 메시지(friendly_message)
- 2~3문장으로, 사용자가 불안하지 않도록 따뜻하고 친절하게 작성하세요.
- 현재 증상이 얼마나 흔한지, 어떤 점을 중심으로 지켜보면 좋은지 간단히 설명해 주세요.

# 전체 출력 형식
- 반드시 JSON 형식으로만 출력해야 합니다.
- 아래 형식 지침을 엄격히 따르세요.

{format_instructions}
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "증상: {message}\n\n위 증상을 분석하여 JSON으로 응답하세요."),
    ])

    # 5. LangChain 실행
    chain = chat_prompt | llm | parser

    try:
        print("🤖 Gemini 분석 중...")
        result = await chain.ainvoke({"message": symptom_text})
        print(f"✅ 분석 완료: {result['urgency_level']}")

        formatted = format_response(result)

        return {
            "response": formatted,
            "urgency_level": result.get("urgency_level", "자가관찰"),
            "departments": result.get("departments", []),
        }

    except Exception as e:
        print(f"❌ LLM 오류: {e}")
        import traceback
        traceback.print_exc()

        # 폴백: 라우팅 정보로 응답
        print("⚠️ 폴백 모드")
        return {
            "response": format_fallback_response(routing, medical_context),
            "urgency_level": map_urgency_level(routing["urgency"]["level"]),
            "departments": [routing["primary_department"]],
        }


def map_urgency_level(level: str) -> str:
    """라우팅 urgency level을 한글로 매핑"""
    mapping = {
        "emergency": "응급실",
        "urgent": "외래진료",
        "observation": "자가관찰",
    }
    return mapping.get(level, "자가관찰")


def format_response(analysis: Dict) -> str:
    """응답 포맷팅"""

    urgency_emoji = {
        "응급실": "🔴",
        "외래진료": "🟡",
        "자가관찰": "🟢",
    }

    emoji = urgency_emoji.get(analysis.get("urgency_level", "자가관찰"), "💡")

    parts: List[str] = []

    # 공감 메시지
    parts.append(analysis.get("friendly_message", "증상 분석을 완료했습니다."))
    parts.append("")

    # 응급도
    parts.append(f"{emoji} 응급도: {analysis.get('urgency_level', '자가관찰')}")
    parts.append(f"└─ {analysis.get('urgency_reason', '')}")
    parts.append("")

    # 진료과
    depts = analysis.get("departments", [])
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
    """폴백 응답"""

    urgency = routing["urgency"]
    dept = routing["primary_department"]

    has_context = len(medical_context) > 100

    urgency_emoji = {
        "emergency": "🔴",
        "urgent": "🟡",
        "observation": "🟢",
    }

    emoji = urgency_emoji.get(urgency["level"], "💡")

    parts: List[str] = []
    parts.append("증상 분석을 완료했습니다.")
    parts.append("")
    parts.append(f"{emoji} 응급도: {urgency['label']}")
    parts.append(f"└─ {urgency['reason']}")
    parts.append("")
    parts.append("📋 추천 진료과")
    parts.append(f"└─ {dept}")
    parts.append("")
    parts.append("✅ 즉시 취해야 할 조치")
    parts.append(f"  • {urgency['action']}")

    if has_context:
        parts.append("  • 의료 지식베이스를 참고하여 적절한 응급처치를 하세요")

    parts.append("  • 증상을 관찰하고 악화되면 병원 방문")
    parts.append("")
    parts.append("⚠️ 주의사항")
    parts.append("  • 증상 변화를 주의 깊게 관찰하세요")
    parts.append("  • 악화되거나 48시간 이상 지속되면 병원 방문")
    parts.append("")
    parts.append("💡 이 정보는 응급 가이드이며, 의학적 진단을 대체하지 않습니다.")

    return "\n".join(parts)