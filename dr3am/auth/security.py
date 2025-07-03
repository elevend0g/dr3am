"""Security utilities for authentication and authorization"""

from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any, List
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
from sqlalchemy.orm import Session

from dr3am.utils.config import get_settings
from dr3am.models.database import User as DBUser, get_database_manager
from .models import TokenData, UserResponse


class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security_scheme = HTTPBearer(auto_error=False)
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.settings.security.access_token_expire_minutes
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.settings.security.secret_key, 
            algorithm=self.settings.security.algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.settings.security.refresh_token_expire_days
            )
        
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.settings.security.secret_key, 
            algorithm=self.settings.security.algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.settings.security.secret_key, 
                algorithms=[self.settings.security.algorithm]
            )
            
            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_user_from_token(self, db: Session, token: str) -> Optional[UserResponse]:
        """Get user from JWT token"""
        try:
            payload = self.verify_token(token)
            user_id: str = payload.get("sub")
            
            if user_id is None:
                return None
            
            # Get user from database
            db_user = db.query(DBUser).filter(DBUser.user_id == user_id).first()
            if db_user is None:
                return None
            
            return UserResponse.from_orm(db_user)
            
        except HTTPException:
            return None
    
    def create_api_key(self, user_id: str, name: str, scopes: List[str] = None) -> str:
        """Create an API key for a user"""
        scopes = scopes or []
        
        # Generate a secure random key
        key = secrets.token_urlsafe(32)
        
        # Create key hash for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # In a real implementation, you'd store this in the database
        # For now, we'll return the key directly
        return key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return user info"""
        # In a real implementation, you'd:
        # 1. Hash the provided key
        # 2. Look it up in the database
        # 3. Check if it's active and not expired
        # 4. Return user info and scopes
        
        # For now, return None (not implemented)
        return None


# Global auth manager instance
auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Get the authentication manager"""
    return auth_manager


def get_db_session():
    """Get database session (dependency)"""
    settings = get_settings()
    db_manager = get_database_manager(settings.get_database_url())
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
) -> UserResponse:
    """Get the current authenticated user"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = auth_mgr.get_user_from_token(db, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_scopes(required_scopes: List[str]):
    """Dependency factory for requiring specific scopes"""
    async def scopes_dependency(
        current_user: UserResponse = Depends(get_current_active_user),
        # In a real implementation, you'd extract scopes from the token
    ) -> UserResponse:
        # For now, just return the user
        # In a real implementation, you'd check if the user has the required scopes
        return current_user
    
    return scopes_dependency


# Convenience functions
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token"""
    return auth_manager.create_access_token(data, expires_delta)


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a refresh token"""
    return auth_manager.create_refresh_token(data, expires_delta)


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify a token"""
    return auth_manager.verify_token(token, token_type)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password"""
    return auth_manager.verify_password(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return auth_manager.get_password_hash(password)


class OptionalAuth:
    """Optional authentication dependency"""
    
    def __init__(self):
        self.security_scheme = HTTPBearer(auto_error=False)
    
    async def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        db: Session = Depends(get_db_session),
        auth_mgr: AuthManager = Depends(get_auth_manager)
    ) -> Optional[UserResponse]:
        """Get the current user if authenticated, None otherwise"""
        if credentials is None:
            return None
        
        try:
            user = auth_mgr.get_user_from_token(db, credentials.credentials)
            return user
        except HTTPException:
            return None


# Instance for optional authentication
optional_auth = OptionalAuth()