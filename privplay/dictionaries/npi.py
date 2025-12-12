"""NPI database lookup utilities."""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_npi_db_path() -> Path:
    """Get path to NPI database."""
    from ..config import get_config
    config = get_config()
    return config.data_dir / "npi" / "npi.db"


def is_npi_available() -> bool:
    """Check if NPI database is downloaded."""
    return get_npi_db_path().exists()


def validate_npi_checksum(npi: str) -> bool:
    """
    Validate NPI using Luhn algorithm.
    
    NPI must be 10 digits, and the check digit (last digit) must satisfy
    the Luhn algorithm when prefixed with "80840" (the health industry prefix).
    """
    if not npi or len(npi) != 10 or not npi.isdigit():
        return False
    
    # Prefix with 80840 for Luhn calculation
    prefixed = "80840" + npi
    
    # Luhn algorithm
    total = 0
    for i, digit in enumerate(reversed(prefixed)):
        d = int(digit)
        if i % 2 == 1:  # Double every second digit from right
            d *= 2
            if d > 9:
                d -= 9
        total += d
    
    return total % 10 == 0


def lookup_npi(npi: str) -> Optional[Dict[str, Any]]:
    """
    Look up an NPI in the database.
    
    Args:
        npi: 10-digit NPI number
        
    Returns:
        Dict with provider info, or None if not found/DB unavailable
    """
    db_path = get_npi_db_path()
    
    if not db_path.exists():
        logger.warning("NPI database not available. Run: phi-train download npi")
        return None
    
    # Validate format
    if not npi or len(npi) != 10 or not npi.isdigit():
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM npi WHERE npi = ?", (npi,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
        
    except Exception as e:
        logger.error(f"NPI lookup failed: {e}")
        return None


def search_npi_by_name(name: str, limit: int = 10) -> list:
    """
    Search NPI database by provider name.
    
    Args:
        name: Provider name (partial match)
        limit: Maximum results to return
        
    Returns:
        List of matching provider records
    """
    db_path = get_npi_db_path()
    
    if not db_path.exists():
        logger.warning("NPI database not available")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search by provider name or last name
        cursor.execute("""
            SELECT * FROM npi 
            WHERE provider_name LIKE ? OR last_name LIKE ?
            LIMIT ?
        """, (f"%{name}%", f"%{name}%", limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"NPI search failed: {e}")
        return []


def is_valid_npi(npi: str) -> bool:
    """
    Check if a string is a valid NPI.
    
    Validates:
    1. Format (10 digits starting with 1 or 2)
    2. Luhn checksum
    3. Optionally, existence in database (if available)
    """
    if not npi or len(npi) != 10 or not npi.isdigit():
        return False
    
    if npi[0] not in ("1", "2"):
        return False
    
    if not validate_npi_checksum(npi):
        return False
    
    return True


def enrich_with_npi(npi: str) -> Optional[Dict[str, Any]]:
    """
    Enrich an NPI detection with provider details.
    
    Returns enriched data if NPI is valid and found in database.
    """
    if not is_valid_npi(npi):
        return None
    
    return lookup_npi(npi)
