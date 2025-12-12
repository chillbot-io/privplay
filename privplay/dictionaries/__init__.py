"""Dictionary-based detection for PHI/PII entities."""

from .loader import (
    load_payers,
    load_lab_tests,
    load_drugs,
    load_hospitals,
    load_all_dictionaries,
    register_with_presidio,
    DictionaryDetector,
    get_dictionary_status,
)

from .downloader import (
    download_fda_drugs,
    download_cms_hospitals,
    download_npi_database,
    download_all,
    get_download_status,
)

from .npi import (
    lookup_npi,
    search_npi_by_name,
    is_valid_npi,
    validate_npi_checksum,
    is_npi_available,
    enrich_with_npi,
)

__all__ = [
    # Loader
    "load_payers",
    "load_lab_tests", 
    "load_drugs",
    "load_hospitals",
    "load_all_dictionaries",
    "register_with_presidio",
    "DictionaryDetector",
    "get_dictionary_status",
    # Downloader
    "download_fda_drugs",
    "download_cms_hospitals",
    "download_npi_database",
    "download_all",
    "get_download_status",
    # NPI
    "lookup_npi",
    "search_npi_by_name",
    "is_valid_npi",
    "validate_npi_checksum",
    "is_npi_available",
    "enrich_with_npi",
]
