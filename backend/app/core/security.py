import re
from pathlib import Path
from app.core.exceptions import AudioProcessingError

SAFE_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

def sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filenames to prevent directory traversal or shell injection.
    Replaces non-alphanumeric characters with underscores.
    """
    path_obj = Path(filename)
    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", path_obj.stem)
    safe_ext = path_obj.suffix.lower()
    
    if not safe_ext or len(safe_ext) > 6:
        raise AudioProcessingError("Invalid audio file extension.")
        
    return f"{safe_stem}{safe_ext}"
