from django.urls import path
# Importamos las NUEVAS vistas que crearemos en el siguiente paso
from .views.views import MultiStageLoginView, UserDataView 

# Importamos la vista de logout que nos da SimpleJWT
from rest_framework_simplejwt.views import TokenBlacklistView

urlpatterns = [
    
    # Endpoint: POST /api/v1/auth/login 
    path('auth/login', MultiStageLoginView.as_view(), name='api_login'),
    
    # Endpoint: GET /api/v1/auth/me 
    path('auth/me', UserDataView.as_view(), name='api_me'),
    
    # Endpoint: POST /api/v1/auth/logout 
    path('auth/logout', TokenBlacklistView.as_view(), name='api_logout'),
]