"""Dictionary data downloader for FDA drugs, CMS hospitals, and NPI database."""

import csv
import io
import json
import logging
import sqlite3
import zipfile
from pathlib import Path
from typing import Optional, Callable
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Data source URLs
FDA_NDC_URL = "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip"
CMS_HOSPITALS_URL = "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0?limit=10000&offset=0&count=true&results=true&schema=true&format=json"
NPI_DOWNLOAD_URL = "https://download.cms.gov/nppes/NPPES_Data_Dissemination_{month}_{year}.zip"


def get_data_dir() -> Path:
    """Get the data directory for dictionaries."""
    from ..config import get_config
    config = get_config()
    data_dir = config.data_dir / "dictionaries"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_npi_dir() -> Path:
    """Get the NPI database directory."""
    from ..config import get_config
    config = get_config()
    npi_dir = config.data_dir / "npi"
    npi_dir.mkdir(parents=True, exist_ok=True)
    return npi_dir


def download_file(url: str, dest: Path, progress_callback: Optional[Callable] = None) -> bool:
    """Download a file with optional progress callback."""
    try:
        logger.info(f"Downloading {url}")
        
        req = urllib.request.Request(url, headers={"User-Agent": "privplay/0.1"})
        
        with urllib.request.urlopen(req, timeout=300) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)
        
        logger.info(f"Downloaded to {dest}")
        return True
        
    except urllib.error.URLError as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def download_fda_drugs(progress_callback: Optional[Callable] = None) -> bool:
    """
    Download FDA NDC drug database and extract drug names.
    
    Creates: ~/.privplay/dictionaries/drugs.txt
    """
    data_dir = get_data_dir()
    drugs_file = data_dir / "drugs.txt"
    temp_zip = data_dir / "ndc_temp.zip"
    
    try:
        # Download the zip
        if not download_file(FDA_NDC_URL, temp_zip, progress_callback):
            return False
        
        # Extract and parse
        drug_names = set()
        
        with zipfile.ZipFile(temp_zip, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    with zf.open(name) as f:
                        data = json.load(f)
                        
                        for result in data.get("results", []):
                            # Brand name
                            brand = result.get("brand_name", "")
                            if brand:
                                drug_names.add(brand.strip())
                            
                            # Generic name
                            generic = result.get("generic_name", "")
                            if generic:
                                # Generic names can be comma-separated
                                for g in generic.split(","):
                                    g = g.strip()
                                    if g:
                                        drug_names.add(g)
                            
                            # Active ingredients
                            for ing in result.get("active_ingredients", []):
                                ing_name = ing.get("name", "")
                                if ing_name:
                                    drug_names.add(ing_name.strip())
        
        # Write to file
        with open(drugs_file, "w") as f:
            f.write("# FDA NDC Drug Names\n")
            f.write(f"# Count: {len(drug_names)}\n")
            f.write("# Source: https://open.fda.gov/apis/drug/ndc/\n\n")
            for name in sorted(drug_names):
                # Skip very short names (likely abbreviations that cause FPs)
                if len(name) >= 3:
                    f.write(f"{name}\n")
        
        # Cleanup
        temp_zip.unlink()
        
        logger.info(f"Extracted {len(drug_names)} drug names to {drugs_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process FDA drug data: {e}")
        if temp_zip.exists():
            temp_zip.unlink()
        return False


def download_cms_hospitals(progress_callback: Optional[Callable] = None) -> bool:
    """
    Download CMS hospital list.
    
    Creates: ~/.privplay/dictionaries/hospitals.txt
    """
    data_dir = get_data_dir()
    hospitals_file = data_dir / "hospitals.txt"
    
    try:
        hospital_names = set()
        offset = 0
        limit = 10000
        
        while True:
            url = f"https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0?limit={limit}&offset={offset}&results=true&format=json"
            
            req = urllib.request.Request(url, headers={"User-Agent": "privplay/0.1"})
            
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.load(response)
                results = data.get("results", [])
                
                if not results:
                    break
                
                for hospital in results:
                    name = hospital.get("facility_name", "")
                    if name:
                        hospital_names.add(name.strip())
                
                logger.info(f"Fetched {len(results)} hospitals (offset {offset})")
                
                if len(results) < limit:
                    break
                    
                offset += limit
        
        # Write to file
        with open(hospitals_file, "w") as f:
            f.write("# CMS Hospital Names\n")
            f.write(f"# Count: {len(hospital_names)}\n")
            f.write("# Source: https://data.cms.gov/provider-data/\n\n")
            for name in sorted(hospital_names):
                f.write(f"{name}\n")
        
        logger.info(f"Extracted {len(hospital_names)} hospital names to {hospitals_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download CMS hospitals: {e}")
        return False


def download_npi_database(progress_callback: Optional[Callable] = None) -> bool:
    """
    Download NPPES NPI database and create SQLite database.
    
    Creates: ~/.privplay/npi/npi.db
    
    WARNING: This is a large download (~1GB compressed, ~9GB uncompressed).
    """
    import datetime
    
    npi_dir = get_npi_dir()
    npi_db = npi_dir / "npi.db"
    
    # Find the most recent monthly file
    # NPI files are released monthly, usually mid-month
    now = datetime.datetime.now()
    
    # Try current month, then previous months
    for months_back in range(3):
        try_date = now - datetime.timedelta(days=30 * months_back)
        month = try_date.strftime("%B")
        year = try_date.strftime("%Y")
        
        url = NPI_DOWNLOAD_URL.format(month=month, year=year)
        temp_zip = npi_dir / f"npi_{year}_{month}.zip"
        
        logger.info(f"Trying NPI file for {month} {year}")
        
        if download_file(url, temp_zip, progress_callback):
            break
    else:
        logger.error("Could not find NPI download file")
        return False
    
    try:
        # Create SQLite database
        if npi_db.exists():
            npi_db.unlink()
        
        conn = sqlite3.connect(npi_db)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE npi (
                npi TEXT PRIMARY KEY,
                entity_type TEXT,
                provider_name TEXT,
                first_name TEXT,
                last_name TEXT,
                credential TEXT,
                specialty TEXT,
                address_1 TEXT,
                city TEXT,
                state TEXT,
                zip TEXT,
                phone TEXT
            )
        """)
        cursor.execute("CREATE INDEX idx_npi_name ON npi(provider_name)")
        cursor.execute("CREATE INDEX idx_npi_last ON npi(last_name)")
        
        # Extract and load CSV
        with zipfile.ZipFile(temp_zip, "r") as zf:
            for name in zf.namelist():
                if "npidata" in name.lower() and name.endswith(".csv"):
                    logger.info(f"Processing {name}")
                    
                    with zf.open(name) as f:
                        # Read as text
                        text_wrapper = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                        reader = csv.DictReader(text_wrapper)
                        
                        batch = []
                        count = 0
                        
                        for row in reader:
                            npi = row.get("NPI", "")
                            if not npi:
                                continue
                            
                            entity_type = row.get("Entity Type Code", "")
                            
                            if entity_type == "1":  # Individual
                                first = row.get("Provider First Name", "")
                                last = row.get("Provider Last Name (Legal Name)", "")
                                provider_name = f"{first} {last}".strip()
                                credential = row.get("Provider Credential Text", "")
                            else:  # Organization
                                provider_name = row.get("Provider Organization Name (Legal Business Name)", "")
                                first = ""
                                last = ""
                                credential = ""
                            
                            specialty = row.get("Healthcare Provider Taxonomy Code_1", "")
                            address = row.get("Provider First Line Business Practice Location Address", "")
                            city = row.get("Provider Business Practice Location Address City Name", "")
                            state = row.get("Provider Business Practice Location Address State Name", "")
                            zip_code = row.get("Provider Business Practice Location Address Postal Code", "")
                            phone = row.get("Provider Business Practice Location Address Telephone Number", "")
                            
                            batch.append((
                                npi, entity_type, provider_name, first, last,
                                credential, specialty, address, city, state, zip_code, phone
                            ))
                            
                            if len(batch) >= 10000:
                                cursor.executemany(
                                    "INSERT OR REPLACE INTO npi VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                    batch
                                )
                                conn.commit()
                                count += len(batch)
                                batch = []
                                
                                if progress_callback:
                                    progress_callback(count, 0)
                                
                                logger.info(f"Loaded {count:,} records")
                        
                        # Final batch
                        if batch:
                            cursor.executemany(
                                "INSERT OR REPLACE INTO npi VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                batch
                            )
                            count += len(batch)
                        
                        conn.commit()
                        logger.info(f"Total NPI records: {count:,}")
                    
                    break  # Only process the main file
        
        conn.close()
        
        # Cleanup
        temp_zip.unlink()
        
        logger.info(f"Created NPI database at {npi_db}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process NPI data: {e}")
        if temp_zip.exists():
            temp_zip.unlink()
        return False


def download_all(progress_callback: Optional[Callable] = None) -> dict:
    """
    Download all dictionary data.
    
    Returns dict with status of each download.
    """
    results = {
        "drugs": download_fda_drugs(progress_callback),
        "hospitals": download_cms_hospitals(progress_callback),
        "npi": download_npi_database(progress_callback),
    }
    return results


def get_download_status() -> dict:
    """Check which dictionaries are downloaded."""
    data_dir = get_data_dir()
    npi_dir = get_npi_dir()
    
    return {
        "drugs": (data_dir / "drugs.txt").exists(),
        "hospitals": (data_dir / "hospitals.txt").exists(),
        "npi": (npi_dir / "npi.db").exists(),
        "payers": True,  # Bundled
        "lab_tests": True,  # Bundled
    }
