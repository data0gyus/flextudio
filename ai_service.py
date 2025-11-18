import os
import json
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv

from models import ChatRequest

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-2.0-flash')


def build_rag_context(rag_results: List[Dict]) -> str:
    """RAG 컨텍스트"""
    if not rag_results:
        return ""
    
    rag_texts = []
    for r in rag_results:
        source = r.get('source', 'Unknown')
        content = r.get('content', '')[:200]  # 200자 제한
        rag_texts.append(f"[{source}] {content}")
    
    return f"""
<참고문서>
{chr(10).join(rag_texts)}
</참고문서>

위 참고문서 내용을 기반으로 답변하되, 모순되는 정보는 제공하지 마세요.
"""


SYSTEM_PROMPT = """당신은 응급 의료 상담 전문가 'CareNow'입니다.

# 응급도 분류 기준

🚨 **응급실** (즉시 방문)
- 호흡곤란, 의식저하, 경련, 심한 출혈
- 심한 알레르기 반응, 고열(40도+) + 의식 변화

🏥 **외래진료** (당일~익일 방문)
- 지속적인 고열(38.5도+, 48시간+)
- 지속적인 구토/설사, 심한 귀/복통

🏠 **자가관찰** (집에서 경과 관찰)
- 경미한 발열(38도 이하), 가벼운 감기

# 진료과 가이드
- 호흡기 → 소아청소년과, 이비인후과
- 피부 → 피부과, 소아청소년과
- 소화기 → 소아청소년과
- 외상 → 외과, 정형외과

# 응답 형식 (JSON만)
{
  "urgency_level": "자가관찰/외래진료/응급실",
  "urgency_reason": "판단 근거 1-2문장",
  "departments": ["진료과1", "진료과2"],
  "immediate_actions": ["조치1", "조치2", "조치3"],
  "precautions": ["주의1", "주의2"],
  "friendly_message": "공감적 메시지 3-4문장"
}

**반드시 위 JSON 형식만 응답하세요.**
"""


async def analyze_symptom(request: ChatRequest, rag_results: List[Dict] = None) -> Dict[str, str]:
    """증상 분석"""
    
    # RAG 컨텍스트
    rag_context = build_rag_context(rag_results or [])
    
    # 사용자 정보
    user_info = ""
    if request.user_age:
        user_info = f"\n환자 나이: {request.user_age}세"
    
    # 프롬프트
    user_prompt = f"""{rag_context}

증상: {request.message}{user_info}

위 증상을 분석하여 JSON으로만 응답하세요."""
    
    try:
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
        )
        
        response_text = response.text.strip()
        
        # JSON 추출
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        print(f"✅ 증상 분석 완료 (RAG: {len(rag_results or [])}개)")
        
        formatted = format_response(result)
        
        return {
            "response": formatted,
            "urgency_level": result.get("urgency_level", "외래진료"),
            "used_rag": len(rag_results or []) > 0
        }
    
    except Exception as e:
        print(f"❌ 증상 분석 오류: {e}")
        
        return {
            "response": """증상 분석 중 오류가 발생했습니다.

🏥 안전을 위해 가까운 병원 방문을 권장합니다.

✅ 즉시 조치
  • 가까운 병원 방문
  • 증상 변화 관찰
  • 필요시 119 연락

💡 이 정보는 응급 가이드이며, 의학적 진단을 대체하지 않습니다.""",
            "urgency_level": "외래진료",
            "used_rag": False
        }


def format_response(analysis: Dict) -> str:
    """예쁜 텍스트 포맷팅"""
    
    urgency_emoji = {
        "자가관찰": "🏠",
        "외래진료": "🏥",
        "응급실": "🚨"
    }
    
    emoji = urgency_emoji.get(analysis.get("urgency_level", "외래진료"), "💡")
    
    parts = []
    parts.append(analysis.get("friendly_message", ""))
    parts.append("")
    parts.append(f"{emoji} 응급도: {analysis.get('urgency_level', '외래진료')}")
    parts.append(f"└─ {analysis.get('urgency_reason', '')}")
    parts.append("")
    parts.append("📋 추천 진료과")
    parts.append(f"└─ {', '.join(analysis.get('departments', []))}")
    parts.append("")
    parts.append("✅ 즉시 취해야 할 조치")
    for action in analysis.get("immediate_actions", []):
        parts.append(f"  • {action}")
    parts.append("")
    parts.append("⚠️ 주의사항")
    for precaution in analysis.get("precautions", []):
        parts.append(f"  • {precaution}")
    parts.append("")
    parts.append("💡 이 정보는 응급 가이드이며, 의학적 진단을 대체하지 않습니다.")
    
    return "\n".join(parts)