from pathlib import Path


VERSION_ID = "3451"
TRANSLATION_CODE = "VAHNT"
BASE_URL = "https://www.bible.com/audio-bible"
MIN_AUDIO_BYTES = 10_000
DEFAULT_DELAY_SECONDS = 2.0
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5.0
PAGE_TIMEOUT_MS = 60_000
PLAYBACK_WAIT_SECONDS = 8

ROOT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = ROOT_DIR / "audio"
LOG_DIR = ROOT_DIR / "logs"
PROGRESS_PATH = ROOT_DIR / "progress.json"
FAILED_PATH = ROOT_DIR / "failed.json"
METADATA_PATH = ROOT_DIR / "chapter_metadata.json"
LOG_PATH = LOG_DIR / "downloader.log"
