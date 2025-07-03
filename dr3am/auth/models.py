"""Authentication data models"""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, Dict, Any
from datetime import datetime


class UserBase(BaseModel):
    """Base user model"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    is_active: bool = True
    preferences: Dict[str, Any] = Field(default_factory=dict)


class UserCreate(UserBase):
    """User creation model"""
    user_id: str = Field(..., min_length=1, max_length=255)
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError("User ID cannot be empty")
        # Basic validation - alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("User ID can only contain letters, numbers, underscores, and hyphens")
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if v is None:
            return v
        
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Check for at least one number and one letter
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one number")
        
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        
        return v


class UserUpdate(UserBase):
    """User update model"""
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    
    @validator('password')
    def validate_password(cls, v):
        if v is None:
            return v
        
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        return v


class UserResponse(UserBase):
    """User response model (without sensitive data)"""
    user_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserInDB(UserResponse):
    """User model with hashed password (for internal use)"""
    hashed_password: Optional[str] = None


class Token(BaseModel):
    """JWT token model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v is not None and not v.strip():
            raise ValueError("User ID cannot be empty")
        return v


class LoginRequest(BaseModel):
    """Login request model"""
    user_id: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    
    @validator('user_id', 'password')
    def validate_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str = Field(..., min_length=1)
    
    @validator('refresh_token')
    def validate_refresh_token(cls, v):
        if not v.strip():
            raise ValueError("Refresh token cannot be empty")
        return v.strip()


class PasswordChangeRequest(BaseModel):
    """Password change request model"""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long")
        
        # Check for at least one number and one letter
        if not any(c.isdigit() for c in v):
            raise ValueError("New password must contain at least one number")
        
        if not any(c.isalpha() for c in v):
            raise ValueError("New password must contain at least one letter")
        
        return v
    
    @validator('new_password')
    def validate_passwords_different(cls, v, values):
        if 'current_password' in values and v == values['current_password']:
            raise ValueError("New password must be different from current password")
        return v


class APIKeyCreate(BaseModel):
    """API key creation model"""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("API key name cannot be empty")
        return v.strip()


class APIKeyResponse(BaseModel):
    """API key response model"""
    id: int
    name: str
    key_prefix: str  # First 8 characters of the key
    scopes: list[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True