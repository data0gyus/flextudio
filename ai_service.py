import os 
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# 의도 분류 스키마 정의 
class IntentSchema(BaseModel):
    needs_facility: bool = Field(description="시설 검색?")
    facility_type : str = Field(description="시설 타입: emergency, pharmacy, care, none")
    urgency: str = Field(description="긴급도: low, med, high")
    keywords: list[str] = Field(description="핵심 키워드 리스트")

# 응답 스키마 정의 
class ResponseSchema(BaseModel):
    """응답 생성 결과 스키마"""
    response: str = Field(description="사용자에게 전달할 답변 내용")
    key_points: List[str] = Field(description="핵심 조치사항 (3개 이내)")


class PromptBuilder:
    
    @staticmethod
    def get_intent_role() -> str:
        """의도 Role"""
        return """당신은 응급 상황 분류 전문가입니다.
사용자의 증상 메시지를 분석하여 긴급도와 필요한 시설을 정확히 판단하는 것이 당신의 역할입니다."""
    
    @staticmethod
    def get_intent_style() -> str:
        """의도 Style"""
        return """
분류 원칙:
1. 생명에 위협이 되는 증상은 반드시 "high" 긴급도로 분류
2. 애매한 경우 보수적으로 판단 (안전 우선)
3. 키워드는 의학적으로 중요한 단어만 추출
4. 시설 타입은 반드시 하나만 선택
"""
    
    @staticmethod
    def get_response_role() -> str:
        """응답 Role"""
        return """당신은 돌봄 공백 가정을 위한 응급 도우미 'CareNow'입니다.
홀로 아이를 돌보는 보호자에게 신속하고 정확한 응급 조언을 제공하는 것이 당신의 사명입니다."""
    
    @staticmethod
    def get_response_style() -> str:
        """응답 Style"""
        return """
답변 스타일:
1. 친근하고 공감적인 톤으로 안심시키기
2. 5문장 이내로 간결하고 명확하게
3. 즉시 실행 가능한 조치사항 우선 언급
4. 의학적 진단이 아닌 응급처치 안내임을 명시

제약사항:
- 긴급도가 "high"이면 반드시 119를 최우선 안내
- 참고문서 내용과 모순되는 답변 절대 금지
- 확실하지 않은 정보는 전문가 상담 권유
- 불필요한 의학 용어 사용 자제
"""
    
#     @staticmethod
#     def get_response_example() -> str:
#         """응답 Example"""
#         return """
# 예시 1 - 경미한 증상:
# 질문: "아이가 열이 37.5도 있어요"
# 답변: "아이가 미열이 있으시군요. 우선 시원한 환경에서 휴식을 취하게 해주시고, 
# 물이나 이온음료를 자주 마시게 해주세요. 열이 38도 이상 오르면 해열제를 
# 고려해보시고, 39도 이상이면 병원 방문이 필요합니다. 
# 이는 의학적 진단이 아닌 응급처치 안내입니다."
# 핵심 조치: ["휴식", "수분 섭취", "체온 관찰"]

# 예시 2 - 응급 상황:
# 질문: "아이가 호흡이 곤란해요!"
# 답변: "호흡곤란은 매우 위험한 증상입니다. 즉시 119에 신고하세요! 
# 구급대가 오는 동안 아이를 편안한 자세로 앉히고, 옷을 느슨하게 해주세요. 
# 절대 억지로 눕히지 마시고, 계속 아이 상태를 관찰해주세요."
# 핵심 조치: ["즉시 119 신고", "편안한 자세", "옷 느슨하게"]
# """
    
    @staticmethod
    def build_rag_context(rag_results: List[Dict]) -> str:
        """RAG 결과를 Context로 변환"""
        if not rag_results:
            return ""
        
        rag_texts = []
        for r in rag_results:
            source = r.get('source', 'Unknown')
            content = r.get('content', '')[:300]  # 300자 제한
            rag_texts.append(f"[{source}]\n{content}...")
        
        rag_context = "\n\n".join(rag_texts)
        
        return f"""
<참고문서>
{rag_context}
</참고문서>

위 참고문서의 내용을 기반으로 답변하세요.
문서에 명시된 내용과 모순되는 답변은 하지 마세요.
"""
    
    @staticmethod
    def build_facility_context(facilities: List[Dict]) -> str:
        """시설 정보를 Context로 변환"""
        if not facilities:
            return ""
        
        facility_texts = []
        for i, f in enumerate(facilities[:3], 1):
            name = f.get('name', '시설명 없음')
            distance = f.get('distance', 0)
            phone = f.get('phone', '')
            
            facility_info = f"{i}. {name} (거리: {distance:.1f}km)"
            if phone:
                facility_info += f" - 전화: {phone}"
            
            facility_texts.append(facility_info)
        
        facility_list = "\n".join(facility_texts)
        
        return f"""
<주변 이용 가능 시설>
{facility_list}
</주변 이용 가능 시설>

위 시설 정보를 자연스럽게 답변에 포함하세요.
반드시 "전화로 먼저 확인 후 방문"하라고 안내하세요.
"""
    
    @staticmethod
    def build_intent_context(intent: Dict) -> str:
        """의도 분류 결과를 Context로 변환"""
        urgency = intent.get('urgency', 'low')
        facility_type = intent.get('facility_type', 'none')
        keywords = intent.get('keywords', [])
        
        urgency_map = {
            'high': '매우 긴급 (생명 위협 가능)',
            'medium': '긴급 (빠른 대응 필요)',
            'low': '경미함 (관찰 필요)'
        }
        
        facility_map = {
            'emergency': '응급실',
            'pharmacy': '약국',
            'care': '돌봄센터',
            'none': '시설 불필요'
        }
        
        return f"""
<사용자 상황 분석>
- 긴급도: {urgency_map.get(urgency, urgency)}
- 필요 시설: {facility_map.get(facility_type, facility_type)}
- 핵심 증상: {', '.join(keywords) if keywords else '없음'}
</사용자 상황 분석>

위 분석 결과를 고려하여 적절한 수준의 조언을 제공하세요.
"""



# 의도 분류
async def classify_symptom(request):
    
    role = PromptBuilder.get_intent_role()    
    style = PromptBuilder.get_intent_style()

    parser = JsonOutputParser(pydantic_object=IntentSchema)
    format_instructions = parser.get_format_instructions()
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
    prompt = PromptTemplate(
        template=f"""{role}

{{message}}

{style}

# 판단 기준:

**긴급도 분류:**
- high: 호흡곤란, 의식저하, 심한 출혈, 경련, 심한 가슴통증
- medium: 고열(39도 이상), 지속적 구토/설사, 심한 복통, 심한 두통
- low: 경미한 발열(38도 이하), 감기 증상, 가벼운 상처

**시설 타입 분류:**
- emergency: 즉시 응급실 방문 필요 (긴급도 high)
- pharmacy: 약국에서 해결 가능 (경미한 증상, 약 구매)
- care: 돌봄센터 문의 필요 (장기 케어, 상담)
- none: 가정에서 관찰 가능 (매우 경미)

{format_instructions}
""",
        input_variables=["message"],
    )
    
    # Chain 구성
    chain = prompt | llm | parser
    
    try:
        result = await chain.ainvoke({
            "message": request.message
        })
        
        print(f"의도 분류 성공: {result}")
        return result
    
    except Exception as e:
        print(f"의도 분류 오류: {e}")
        # 기본값 반환
        return {
            "needs_facility": False,
            "facility_type": "none",
            "urgency": "low",
            "keywords": []
        }



# 응답 생성
async def generate_response(request, intent: Dict, rag_results: List[Dict], facilities: Optional[List[Dict]]):

    role = PromptBuilder.get_response_role()
    context_parts = []

    rag_context = PromptBuilder.build_rag_context(rag_results)
    if rag_context:
        context_parts.append(rag_context)

    facility_context = PromptBuilder.build_facility_context(facilities or [])
    if facility_context:
        context_parts.append(facility_context)
    
    intent_context = PromptBuilder.build_intent_context(intent)
    context_parts.append(intent_context)
    
    full_context = "\n".join(context_parts)
    
    style = PromptBuilder.get_response_style()
    # example = PromptBuilder.get_response_example()
    
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    format_instructions = parser.get_format_instructions()
    format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
    system_prompt = f"""{role}

{full_context}

{style}


{format_instructions}
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{{message}}")
    ])
    
    chain = chat_prompt | llm | parser
    
    try:
        llm_result = await chain.ainvoke({
            "message": request.message
        })
        
        answer = llm_result.get("response", "")
        key_points = llm_result.get("key_points", [])
        
        return {
            "response": answer,
            "key_points": key_points,
            "used_rag": len(rag_results) > 0,
            "used_facility_api": facilities is not None and len(facilities) > 0,
            "rag_sources": [r['source'] for r in rag_results] if rag_results else None,
            "facilities": facilities,
            "intent": intent
        }
    
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        raise