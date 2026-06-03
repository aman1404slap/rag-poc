import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / '.env')

SECRET_KEY = os.environ.get('SECRET_KEY', 'poc-dev-only-insecure-key-change-for-production')

DEBUG = True

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_results',
    'search',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'videorag.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.csrf',
            ],
        },
    },
]

WSGI_APPLICATION = 'videorag.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'data' / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'data'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / 'data'
VIDEOS_DIR = DATA_DIR / 'videos'
KEYFRAMES_DIR = DATA_DIR / 'keyframes'
INDEX_DIR = DATA_DIR / 'index'

# Create dirs if they don't exist
for _d in [VIDEOS_DIR, KEYFRAMES_DIR, INDEX_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Celery ───────────────────────────────────────────────────────────────────
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = 'django-db'
CELERY_CACHE_BACKEND = 'django-cache'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
# solo pool: tasks run in the worker's main process — no fork, no CUDA re-init error
CELERY_WORKER_POOL = 'solo'

# ── ML model config (swap here to upgrade) ───────────────────────────────────
ML = {
    'CLIP_MODEL': 'ViT-B-32',
    'CLIP_PRETRAINED': 'openai',
    'TEXT_EMB_MODEL': 'all-MiniLM-L6-v2',
    # Caption / VLM (primary)
    'VLM_MODEL': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'VLM_LOAD_4BIT': True,
    'VLM_MAX_NEW_TOKENS': 256,
    'VLM_NUM_FRAMES': 1,          # 1 frame avoids multi-image vision token CUDA issues on 12GB
    'VLM_MAX_SIDE': 768,          # resize long edge before VLM
    'VLM_TAG_CONF': 0.6,
    # Fallback caption
    'BLIP_MODEL': 'Salesforce/blip-image-captioning-base',
    # Action recognition (X-CLIP open-vocab primary; VideoMAE fallback)
    'XCLIP_MODEL': 'microsoft/xclip-base-patch32',
    'XCLIP_FRAMES': 8,
    'ACTION_ENGINE': 'xclip',  # 'xclip' | 'videomae'
    'VIDEOMAE_MODEL': 'MCG-NJU/videomae-base-finetuned-kinetics',
    'VIDEOMAE_FRAMES': 16,
    'ACTION_VOCAB': [
        'walking', 'running', 'jumping', 'climbing', 'swimming', 'dancing',
        'talking', 'presenting', 'typing', 'writing', 'reading', 'cooking',
        'eating', 'drinking', 'driving', 'riding', 'playing', 'working',
        'assembling', 'building', 'drilling', 'cutting', 'painting',
        'drawing', 'filming', 'photographing', 'exercising', 'stretching',
        'sitting', 'standing', 'lying', 'falling', 'throwing', 'catching',
        'kicking', 'hitting', 'shooting', 'archery', 'fencing', 'boxing',
        'wrestling', 'skateboarding', 'snowboarding', 'surfing', 'skiing',
        'gardening', 'cleaning', 'washing', 'shopping', 'carrying',
        'pushing', 'pulling', 'lifting', 'opening', 'closing',
        'rotoscoping', 'compositing', 'tracking', 'keying', 'editing',
        'rendering', 'animating', 'modeling', 'texturing',
    ],
    # Object detection
    'YOLO_MODEL': 'yolo11l.pt',
    'OBJECT_FRAMES': 3,
    # OCR
    'OCR_ENGINE': 'paddle',  # 'paddle' | 'easy'
    'OCR_FRAMES': 3,
    'SCENE_THRESHOLD': 30.0,
    'MIN_CLIP_SEC': 1.0,
    'FALLBACK_WINDOW_SEC': 10,
    # detection thresholds
    'ACTION_TOP_K': 5,
    'ACTION_MIN_CONF': 0.05,
    'OCR_MIN_CONF': 0.30,
    'OBJECT_MIN_CONF': 0.25,
    # absolute confidence scoring
    'TEXT_SIM_CAL': (0.15, 0.65),
    'VISUAL_SIM_CAL': (0.18, 0.35),
    'SCORE_WEIGHTS': {'text': 0.55, 'visual': 0.45},
    'TAG_BOOST_WEIGHT': 0.5,
}
