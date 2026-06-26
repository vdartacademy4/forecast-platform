import os
import uuid
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def generate_unique_filename(filename):
    ext = get_file_extension(filename)
    return f"{uuid.uuid4().hex}.{ext}"


def save_uploaded_file(file, upload_folder):
    original_name = secure_filename(file.filename)
    unique_name = generate_unique_filename(original_name)
    file_path = os.path.join(upload_folder, unique_name)
    file.save(file_path)
    file_size = os.path.getsize(file_path)
    return {
        'original_name': original_name,
        'stored_name': unique_name,
        'file_path': file_path,
        'file_size': file_size,
        'extension': get_file_extension(original_name)
    }
