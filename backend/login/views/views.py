# Nuevos imports de DRF y JWT
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate

# Importaciones no borradas
import logging
from ..models.models import Usuario
from django.db import connection
import base64
import json
import io
from typing import Optional

try:
    import numpy as np
    import cv2
except Exception:  # pragma: no cover
    np = None
    cv2 = None

try:
    import face_recognition
except Exception:
    face_recognition = None

def _compute_embedding_from_b64(b64_str) -> Optional['np.ndarray']:
    log = logging.getLogger('facial')
    if not b64_str:
        log.debug('compute_embedding: b64_str vacío')
        return None
    if np is None:
        log.debug('compute_embedding: numpy no disponible')
        return None
    try:
        header, encoded = b64_str.split(',') if ',' in b64_str else ('', b64_str)
        img_bytes = base64.b64decode(encoded)
        image = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            log.debug('compute_embedding: cv2.imdecode devolvió None')
            return None
        if face_recognition is not None:
            rgb = frame[:, :, ::-1]
            boxes = face_recognition.face_locations(rgb, model='hog')
            log.debug(f'compute_embedding: boxes={len(boxes)}')
            if not boxes:
                return None
            encs = face_recognition.face_encodings(rgb, boxes)
            log.debug(f'compute_embedding: encs={len(encs)}')
            if not encs:
                return None
            return np.array(encs[0], dtype=np.float32)
        else:
            # Fallback: usar promedio de píxeles de la región central como "huella" rudimentaria
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            crop = frame[max(cy-100,0):cy+100, max(cx-100,0):cx+100]
            if crop.size == 0:
                log.debug('compute_embedding: crop vacío en fallback')
                return None
            emb = cv2.resize(crop, (16, 16)).astype('float32').reshape(-1)
            emb = emb / (np.linalg.norm(emb) + 1e-6)
            return emb
    except Exception as e:
        logging.getLogger('facial').exception(f'compute_embedding: excepción {e}')
        return None


def _compare_embeddings(stored_bytes: bytes, live_emb) -> bool:
    if stored_bytes is None or live_emb is None or np is None:
        return False
    try:
        stored = np.frombuffer(stored_bytes, dtype=np.float32)
        if face_recognition is not None and stored.shape[0] in (128, 129):
            # distancia euclidiana típica < 0.6
            dist = np.linalg.norm(stored[:128] - live_emb[:128])
            return dist < 0.6
        else:
            # coseno para el fallback
            num = float(np.dot(stored, live_emb))
            den = (np.linalg.norm(stored) * np.linalg.norm(live_emb) + 1e-6)
            sim = num / den
            return sim > 0.9
    except Exception:
        return False


def _compare_to_collection(user: Usuario, live_emb) -> bool:
    """Compara el embedding vivo contra la colección de embeddings del usuario.
    Mantiene la lógica: si no hay colección, usa el método de compatibilidad _compare_embeddings.
    Usa umbral estricto base 0.45 con leve adaptación hasta 0.55 por intentos fallidos.
    """
    try:
        if live_emb is None:
            return False
        if np is None:
            # Sin numpy no podemos comparar colecciones; usar compatibilidad
            return _compare_embeddings(user.facial_data, live_emb)
        # Si no hay colección, caer al camino de compatibilidad
        if not user.facial_embeddings:
            return _compare_embeddings(user.facial_data, live_emb)

        base_thr = 0.45
        thr = min(base_thr + (user.failed_attempts or 0) * 0.03, 0.55)

        live = np.array(live_emb, dtype=np.float32)
        for emb_list in user.facial_embeddings:
            stored = np.array(emb_list, dtype=np.float32)
            # Euclidiana en primeras 128 dims (face_recognition)
            dist = float(np.linalg.norm(stored[:128] - live[:128]))
            if dist < thr:
                return True
        return False
    except Exception:
        return False


def _validate_position_collection(user: Usuario, live_pos) -> bool:
    """Valida la posición contra alguna de las posiciones registradas en el usuario.
    Si no hay colección, usa la posición de compatibilidad existente.
    Tolerancias estrictas y ligera adaptación por intentos fallidos.
    """
    try:
        if not live_pos:
            return False
        positions = user.positions or ([] if user.position_data is None else [user.position_data])
        if not positions:
            return False

        # Tolerancias base
        attempts = user.failed_attempts or 0
        tol_xy = max(0.05, 0.10 - attempts * 0.01)      # 0.10 -> 0.05
        tol_scale = max(0.08, 0.15 - attempts * 0.01)   # 0.15 -> 0.08

        for p in positions:
            # Formato {x,y,scale}
            if all(k in p for k in ('x', 'y', 'scale')) and all(k in live_pos for k in ('x', 'y', 'scale')):
                if (
                    abs(p['x'] - live_pos['x']) <= tol_xy and
                    abs(p['y'] - live_pos['y']) <= tol_xy and
                    abs(p['scale'] - live_pos['scale']) <= tol_scale
                ):
                    return True
            # Formato angular {roll,pitch,yaw,dist}
            if all(k in p for k in ('roll', 'pitch', 'yaw', 'dist')) and all(k in live_pos for k in ('roll', 'pitch', 'yaw', 'dist')):
                tol_ang = max(8, 15 - attempts * 1)
                tol_dist = max(0.12, 0.22 - attempts * 0.02)
                if (
                    abs(p['roll'] - live_pos['roll']) <= tol_ang and
                    abs(p['pitch'] - live_pos['pitch']) <= tol_ang and
                    abs(p['yaw'] - live_pos['yaw']) <= tol_ang and
                    abs(p['dist'] - live_pos['dist']) <= tol_dist
                ):
                    return True
        return False
    except Exception:
        return False

def _validate_position(stored_pos, live_pos) -> bool:
    try:
        # posición esperada: dict con {x,y,scale} o {roll,pitch,yaw,dist}
        keys = ('x', 'y', 'scale')
        if all(k in stored_pos for k in keys) and all(k in live_pos for k in keys):
            tol_xy = 0.12
            tol_scale = 0.20
            ok = (
                abs(stored_pos['x'] - live_pos['x']) <= tol_xy and
                abs(stored_pos['y'] - live_pos['y']) <= tol_xy and
                abs(stored_pos['scale'] - live_pos['scale']) <= tol_scale
            )
            return ok
        angles = ('roll', 'pitch', 'yaw', 'dist')
        if all(k in stored_pos for k in angles) and all(k in live_pos for k in angles):
            tol_ang = 15  # grados
            tol_dist = 0.25
            return (
                abs(stored_pos['roll'] - live_pos['roll']) <= tol_ang and
                abs(stored_pos['pitch'] - live_pos['pitch']) <= tol_ang and
                abs(stored_pos['yaw'] - live_pos['yaw']) <= tol_ang and
                abs(stored_pos['dist'] - live_pos['dist']) <= tol_dist
            )
        return False
    except Exception:
        return False

# --- 3. Vistas de API (El código NUEVO que cumple la documentación) ---

class MultiStageLoginView(APIView):
    """
    Maneja el flujo de login multi-etapas como pide la documentación.
    POST /api/v1/auth/login
    """
    permission_classes = [AllowAny] # Vista pública

    def post(self, request, *args, **kwargs):
        data = request.data
        
        # --- Etapa 1: Usuario y Contraseña ---
        if 'username' in data and 'password' in data:
            # Usamos el email como username, como define tu modelo
            user = authenticate(username=data['username'], password=data['password']) 
            
            if user and user.is_active:
                # Guardamos el ID en la sesión para el siguiente paso
                request.session['login_user_id'] = user.id
                return Response({"status": "pending_facial_recognition"})
            
            return Response({"error": "Credenciales inválidas"}, status=400)

        # --- Etapa 2: Reconocimiento Facial (Tu lógica) ---
        elif 'facialToken' in data:
            user_id = request.session.get('login_user_id')
            if not user_id:
                return Response({"error": "Flujo de login inválido, inicie de nuevo"}, status=400)
            
            try:
                user = Usuario.objects.get(id=user_id)
                position = data.get('position_data', {}) # El front debe enviar 'position_data'
                
                # ¡AQUÍ REUTILIZAMOS TU LÓGICA EXISTENTE!
                live_emb = _compute_embedding_from_b64(data['facialToken'])
                if live_emb is None:
                    return Response({"error": "Rostro no detectado"}, status=401)
                
                match = _compare_to_collection(user, live_emb)
                position_ok = _validate_position_collection(user, position)

                if match and position_ok:
                    user.failed_attempts = 0
                    user.save(update_fields=['failed_attempts'])
                    # Éxito, pedimos el siguiente paso
                    return Response({"status": "pending_dni_code"})
                
                user.failed_attempts = min(user.failed_attempts + 1, 5)
                user.save(update_fields=['failed_attempts'])
                return Response({"error": "Reconocimiento facial fallido"}, status=401)

            except Usuario.DoesNotExist:
                return Response({"error": "Usuario no encontrado"}, status=404)
        
        # --- Etapa 3 y 4: DNI y Código Manual ---
        elif 'dni' in data and 'code' in data:
            user_id = request.session.get('login_user_id')
            if not user_id:
                return Response({"error": "Flujo de login inválido, inicie de nuevo"}, status=400)
            
            user = Usuario.objects.get(id=user_id)
            
            # (Aquí va la lógica de validación de DNI y Código Manual)
            dni_valido = (user.dni == data['dni']) 
            codigo_valido = (data['code'] == "CODIGO_SECRETO") # <-- REEMPLAZAR CON LÓGICA REAL

            if dni_valido and codigo_valido:
                # ¡ÉXITO FINAL! Generamos el token JWT
                refresh = RefreshToken.for_user(user)
                request.session.flush() # Limpiamos la sesión
                
                # Respuesta final según la documentación
                return Response({
                    'token': str(refresh.access_token),
                    'user': {
                        'id': user.id,
                        'nombre': f"{user.nombres} {user.apellidos}",
                        'rol': "Admin" # O user.rol si lo tienes
                    }
                })
            
            return Response({"error": "DNI o código manual inválido"}, status=401)

        return Response({"error": "Payload de login incorrecto o fuera de etapa"}, status=400)


class UserDataView(APIView):
    """
    Devuelve los datos del usuario autenticado.
    GET /api/v1/auth/me
    """
    permission_classes = [IsAuthenticated] # Requiere un Token JWT válido

    def get(self, request, *args, **kwargs):
        user = request.user
        return Response({
            'id': user.id,
            'nombre': f"{user.nombres} {user.apellidos}",
            'rol': "Admin" # O user.rol
        })