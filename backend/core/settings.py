from pathlib import Path
import os
from dotenv import load_dotenv


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: list[str] | None = None) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]

BASE_DIR = Path(__file__).resolve().parent.parent


load_dotenv(os.path.join(BASE_DIR, '.env'))

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-p4ileu%j8#-#+n&s-x+d=r_yubn=+8%x1v89+c2s(jyshn$66n",
)

DEBUG = env_bool("DJANGO_DEBUG", default=True)

ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    ["localhost", "127.0.0.1"] if DEBUG else [],
)

if not DEBUG:
    if SECRET_KEY.startswith("django-insecure-"):
        raise ValueError("DJANGO_SECRET_KEY must be set to a secure value when DJANGO_DEBUG=False.")
    if not ALLOWED_HOSTS:
        raise ValueError("DJANGO_ALLOWED_HOSTS must be set when DJANGO_DEBUG=False.")


INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',
    'corsheaders',

    'api.apps.ApiConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


STATIC_URL = 'static/'

STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

JWT_ACCESS_TOKEN_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES", "60"))
JWT_REFRESH_TOKEN_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_DAYS", "14"))

LOCAL_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS", LOCAL_CORS_ORIGINS if DEBUG else [])

if DEBUG:
    CORS_ALLOWED_ORIGINS = sorted(set(CORS_ALLOWED_ORIGINS + LOCAL_CORS_ORIGINS))

CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")

if not DEBUG and not CORS_ALLOWED_ORIGINS:
    raise ValueError("CORS_ALLOWED_ORIGINS must include your Vercel frontend URL when DJANGO_DEBUG=False.")

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", default=not DEBUG)
SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "0" if DEBUG else "3600"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", default=not DEBUG)
SECURE_HSTS_PRELOAD = env_bool("DJANGO_SECURE_HSTS_PRELOAD", default=not DEBUG)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("No GOOGLE_API_KEY found in environment variables.")


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("No TAVILY_API_KEY found in environment variables.")

PROVIDER_TIMEOUT_SECONDS = int(os.getenv("PROVIDER_TIMEOUT_SECONDS", "45"))
TAVILY_TIMEOUT_SECONDS = int(os.getenv("TAVILY_TIMEOUT_SECONDS", "20"))
THESYS_TIMEOUT_SECONDS = int(os.getenv("THESYS_TIMEOUT_SECONDS", "35"))
GEMINI_STREAM_TIMEOUT_SECONDS = int(os.getenv("GEMINI_STREAM_TIMEOUT_SECONDS", "60"))
METADATA_TIMEOUT_SECONDS = int(os.getenv("METADATA_TIMEOUT_SECONDS", "20"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_CLASSIFIER_MODEL = os.getenv("GEMINI_CLASSIFIER_MODEL", GEMINI_MODEL)
GEMINI_METADATA_MODEL = os.getenv("GEMINI_METADATA_MODEL", GEMINI_MODEL)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "app": {
            "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "app",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": os.getenv("LOG_LEVEL", "INFO"),
    },
}

