from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any
from datetime import datetime, timezone, timedelta

# 요청 
class Location(BaseModel):
    latitude: float
    longitude: float

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_location: Optional[Location] = None
    conversation_history: Optional[List[ChatMessage]] = []
    user_age: Optional[int] = None



# 응답 
class ErrorDetails(BaseModel):
    code: Optional[str] = None
    field: Optional[str] = None
    rejected_value: Optional[Any] = Field(None, alias="rejectedValue")
    reason: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True)

class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[ErrorDetails] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone(timedelta(hours=9)))
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            datetime: lambda v: v.strftime('%Y-%m-%dT%H:%M:%S')
        }
    )

# 헬퍼
def create_success_response(
    data: Optional[Any] = None,
    message: Optional[str] = None
) -> ApiResponse:
    """성공 응답 생성"""
    return ApiResponse(success=True, data=data, message=message)

def create_error_response(
    message: str,
    code: Optional[str] = None,
    field: Optional[str] = None,
    rejected_value: Optional[Any] = None,
    reason: Optional[str] = None
) -> ApiResponse:
    """에러 응답 생성"""
    error_details = None
    if any([code, field, rejected_value, reason]):
        error_details = ErrorDetails(
            code=code,
            field=field,
            rejected_value=rejected_value,
            reason=reason or message
        )
    return ApiResponse(success=False, message=message, error=error_details)