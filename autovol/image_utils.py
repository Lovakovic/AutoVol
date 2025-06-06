import base64
import os
import mimetypes
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image


class ImageUtils:
    """Utility class for handling image files in forensic analysis."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    
    # VertexAI constraints
    MAX_REQUEST_SIZE_MB = 20
    MAX_REQUEST_SIZE_BYTES = MAX_REQUEST_SIZE_MB * 1024 * 1024
    
    @staticmethod
    def validate_image_format(file_path: str) -> bool:
        """
        Validate if the file is a supported image format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if format is supported, False otherwise
        """
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return False
                
            # Check extension
            extension = path_obj.suffix.lower()
            if extension not in ImageUtils.SUPPORTED_FORMATS:
                return False
                
            # Try to open with PIL to validate it's actually an image
            with Image.open(file_path) as img:
                img.verify()
                
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def get_image_mime_type(file_path: str) -> Optional[str]:
        """
        Get the MIME type for an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            MIME type string or None if cannot be determined
        """
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('image/'):
                return mime_type
            
            # Fallback: determine from PIL format
            with Image.open(file_path) as img:
                format_lower = img.format.lower() if img.format else None
                if format_lower == 'jpeg':
                    return 'image/jpeg'
                elif format_lower == 'png':
                    return 'image/png'
                elif format_lower == 'bmp':
                    return 'image/bmp'
                elif format_lower == 'gif':
                    return 'image/gif'
                elif format_lower in ['tiff', 'tif']:
                    return 'image/tiff'
                elif format_lower == 'webp':
                    return 'image/webp'
                    
        except Exception:
            pass
            
        return None
    
    @staticmethod
    def check_image_size_constraints(file_path: str) -> Tuple[bool, str]:
        """
        Check if image meets VertexAI size constraints.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        try:
            file_size = os.path.getsize(file_path)
            
            # Check raw file size (rough estimate, actual base64 will be larger)
            base64_size_estimate = (file_size * 4) // 3  # Base64 is ~33% larger
            
            if base64_size_estimate > ImageUtils.MAX_REQUEST_SIZE_BYTES:
                size_mb = file_size / (1024 * 1024)
                return False, f"Image too large: {size_mb:.1f}MB (estimated base64 > {ImageUtils.MAX_REQUEST_SIZE_MB}MB limit)"
            
            # Check image dimensions (for reference)
            with Image.open(file_path) as img:
                width, height = img.size
                if width * height > 10000000:  # 10M pixels as a reasonable upper bound
                    return False, f"Image dimensions too large: {width}x{height} pixels"
                    
            return True, ""
            
        except Exception as e:
            return False, f"Error checking image constraints: {str(e)}"
    
    @staticmethod
    def encode_image_to_base64(image_path: str) -> Tuple[str, str]:
        """
        Encode an image file to base64 for VertexAI consumption.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (base64_string, mime_type)
            
        Raises:
            ValueError: If image is invalid or too large
            FileNotFoundError: If image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not ImageUtils.validate_image_format(image_path):
            raise ValueError(f"Unsupported or invalid image format: {image_path}")
        
        is_valid_size, size_error = ImageUtils.check_image_size_constraints(image_path)
        if not is_valid_size:
            raise ValueError(f"Image size constraint violation: {size_error}")
        
        mime_type = ImageUtils.get_image_mime_type(image_path)
        if not mime_type:
            raise ValueError(f"Could not determine MIME type for: {image_path}")
        
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                
            return base64_string, mime_type
            
        except Exception as e:
            raise ValueError(f"Error encoding image to base64: {str(e)}")
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """
        Get comprehensive information about an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image metadata
        """
        info = {
            'path': image_path,
            'exists': False,
            'is_valid': False,
            'mime_type': None,
            'file_size_bytes': 0,
            'file_size_mb': 0.0,
            'dimensions': None,
            'format': None,
            'mode': None,
            'size_constraint_ok': False,
            'size_constraint_reason': None
        }
        
        try:
            if not os.path.exists(image_path):
                return info
            
            info['exists'] = True
            info['file_size_bytes'] = os.path.getsize(image_path)
            info['file_size_mb'] = info['file_size_bytes'] / (1024 * 1024)
            
            info['is_valid'] = ImageUtils.validate_image_format(image_path)
            if not info['is_valid']:
                return info
            
            info['mime_type'] = ImageUtils.get_image_mime_type(image_path)
            
            is_size_ok, size_reason = ImageUtils.check_image_size_constraints(image_path)
            info['size_constraint_ok'] = is_size_ok
            info['size_constraint_reason'] = size_reason if not is_size_ok else None
            
            # Get PIL image info
            with Image.open(image_path) as img:
                info['dimensions'] = img.size
                info['format'] = img.format
                info['mode'] = img.mode
                
        except Exception as e:
            info['error'] = str(e)
            
        return info
    
    @staticmethod
    def find_images_in_directory(directory_path: str, recursive: bool = True) -> list:
        """
        Find all image files in a directory.
        
        Args:
            directory_path: Path to search
            recursive: Whether to search subdirectories
            
        Returns:
            List of image file paths
        """
        image_files = []
        
        try:
            path_obj = Path(directory_path)
            if not path_obj.exists() or not path_obj.is_dir():
                return image_files
            
            pattern = "**/*" if recursive else "*"
            
            for file_path in path_obj.glob(pattern):
                if file_path.is_file():
                    extension = file_path.suffix.lower()
                    if extension in ImageUtils.SUPPORTED_FORMATS:
                        if ImageUtils.validate_image_format(str(file_path)):
                            image_files.append(str(file_path))
                            
        except Exception:
            pass
            
        return sorted(image_files)