"""Authentication routes"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from dr3am.models.database import User as DBUser
from dr3am.utils.config import get_settings
from .models import (
    UserCreate, UserResponse, UserUpdate, Token, LoginRequest, 
    RefreshTokenRequest, PasswordChangeRequest
)
from .security import (
    get_auth_manager, get_db_session, get_current_active_user,
    AuthManager, create_access_token, create_refresh_token,
    verify_password, get_password_hash
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.query(DBUser).filter(DBUser.user_id == user_data.user_id).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this ID already exists"
        )
    
    # Check email if provided
    if user_data.email:
        existing_email = db.query(DBUser).filter(DBUser.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
    
    # Hash password if provided
    hashed_password = None
    if user_data.password:
        hashed_password = auth_mgr.get_password_hash(user_data.password)
    
    # Create user
    db_user = DBUser(
        user_id=user_data.user_id,
        username=user_data.username,
        email=user_data.email,
        is_active=user_data.is_active,
        preferences=user_data.preferences,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Login with username/user_id and password"""
    # Get user from database
    user = db.query(DBUser).filter(DBUser.user_id == form_data.username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not user.hashed_password or not auth_mgr.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create tokens
    settings = get_settings()
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
    refresh_token_expires = timedelta(days=settings.security.refresh_token_expire_days)
    
    access_token = create_access_token(
        data={"sub": user.user_id}, 
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user.user_id}, 
        expires_delta=refresh_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        refresh_token=refresh_token
    )


@router.post("/login-json", response_model=Token)
async def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Login with JSON payload"""
    # Get user from database
    user = db.query(DBUser).filter(DBUser.user_id == login_data.user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password"
        )
    
    # Verify password
    if not user.hashed_password or not auth_mgr.verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password"
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create tokens
    settings = get_settings()
    access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
    refresh_token_expires = timedelta(days=settings.security.refresh_token_expire_days)
    
    access_token = create_access_token(
        data={"sub": user.user_id}, 
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user.user_id}, 
        expires_delta=refresh_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        refresh_token=refresh_token
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = auth_mgr.verify_token(refresh_data.refresh_token, token_type="refresh")
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user from database
        user = db.query(DBUser).filter(DBUser.user_id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        settings = get_settings()
        access_token_expires = timedelta(minutes=settings.security.access_token_expire_minutes)
        
        access_token = create_access_token(
            data={"sub": user.user_id}, 
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds())
        )
        
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get current user information"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: UserResponse = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Update current user information"""
    # Get user from database
    db_user = db.query(DBUser).filter(DBUser.user_id == current_user.user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields
    if user_update.username is not None:
        db_user.username = user_update.username
    
    if user_update.email is not None:
        # Check if email is already taken by another user
        existing_user = db.query(DBUser).filter(
            DBUser.email == user_update.email,
            DBUser.user_id != current_user.user_id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already taken"
            )
        db_user.email = user_update.email
    
    if user_update.preferences is not None:
        db_user.preferences = user_update.preferences
    
    if user_update.password is not None:
        db_user.hashed_password = auth_mgr.get_password_hash(user_update.password)
    
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: UserResponse = Depends(get_current_active_user),
    db: Session = Depends(get_db_session),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Change user password"""
    # Get user from database
    db_user = db.query(DBUser).filter(DBUser.user_id == current_user.user_id).first()
    if not db_user or not db_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password change not available for this user"
        )
    
    # Verify current password
    if not auth_mgr.verify_password(password_data.current_password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Hash new password
    db_user.hashed_password = auth_mgr.get_password_hash(password_data.new_password)
    
    db.commit()
    
    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Logout user (client should discard tokens)"""
    # In a real implementation, you might:
    # - Add the token to a blacklist
    # - Store revoked tokens in Redis
    # - Use token versioning
    
    return {"message": "Logged out successfully"}


@router.delete("/me")
async def delete_current_user(
    current_user: UserResponse = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Delete current user account"""
    # Get user from database
    db_user = db.query(DBUser).filter(DBUser.user_id == current_user.user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Soft delete - just deactivate
    db_user.is_active = False
    db.commit()
    
    return {"message": "User account deactivated successfully"}