#!/usr/bin/env python3
"""
Simple NCBI FTP Downloader with Concurrent Download Support
"""

import os
import gzip
import shutil
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Callable
import time
import logging
import polars as pl
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_ncbi_file(url: str, 
                    output_dir: str = ".", 
                    extract: bool = True,
                    overwrite: bool = False,
                    timeout: int = 30,
                    show_progress: bool = True) -> Optional[str]:

    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get filename from URL
    filename = url.split('/')[-1]
    if not filename:
        logger.error(f"Could not extract filename from URL: {url}")
        return None
    
    local_file = output_path / filename
    
    # Check if file already exists
    if local_file.exists() and not overwrite:
        logger.info(f"File already exists: {local_file}")
        if extract and filename.endswith('.gz'):
            extracted_file = local_file.with_suffix('')
            if extracted_file.exists():
                logger.info(f"Extracted file already exists: {extracted_file}")
                return str(extracted_file)
        return str(local_file)
    
    # Download the file
    try:
        logger.info(f"Downloading: {filename}")
        
        if show_progress:
            # Create a progress hook for urllib
            class ProgressHook:
                def __init__(self, filename, total_size=None):
                    self.filename = filename
                    self.total_size = total_size
                    self.downloaded = 0
                    self.last_update = 0
                
                def __call__(self, count, block_size, total_size):
                    self.total_size = total_size
                    self.downloaded = count * block_size
                    
                    # Update progress every 1 second or at completion
                    current_time = time.time()
                    if current_time - self.last_update > 1 or self.downloaded >= total_size:
                        percent = (self.downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"\r  Progress: {self.downloaded / (1024*1024):.1f} MB / "
                            f"{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", 
                            end='', flush=True)
                        self.last_update = current_time
            
            # Open URL and get file size
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            # Download with progress
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                progress_hook = ProgressHook(filename, total_size)
                with open(local_file, 'wb') as outfile:
                    shutil.copyfileobj(response, outfile, length=1024*1024)  # 1MB chunks
                    # Update progress at the end
                    if total_size > 0:
                        print()  # New line after progress
        else:
            # Simple download without progress
            urllib.request.urlretrieve(url, local_file)
        
        logger.info(f"Downloaded: {local_file} ({local_file.stat().st_size / (1024*1024):.1f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partial download
        if local_file.exists():
            local_file.unlink()
        return None
    
    # Extract if requested
    if extract and filename.endswith('.gz'):
        extracted_file = local_file.with_suffix('')
        
        try:
            logger.info(f"Extracting: {filename}")
            with gzip.open(local_file, 'rb') as f_in:
                with open(extracted_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Extracted to: {extracted_file}")
            
            # Optionally remove the compressed file to save space
            # local_file.unlink()
            
            return str(extracted_file)
        
        except Exception as e:
            logger.error(f"Failed to extract {local_file}: {e}")
            return None
    
    return str(local_file)


def download_multiple_files(urls: List[str],
                            output_dir: str = ".",
                            extract: bool = True,
                            max_workers: int = 4,
                            overwrite: bool = False,
                            timeout: int = 30,
                            show_progress: bool = True,
                            progress_callback: Optional[Callable] = None) -> List[str]:

    downloaded_files = []
    completed = 0
    total = len(urls)
    
    logger.info(f"Starting download of {total} files with {max_workers} concurrent workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(
                download_ncbi_file,
                url=url,
                output_dir=output_dir,
                extract=extract,
                overwrite=overwrite,
                timeout=timeout,
                show_progress=show_progress
            ): url for url in urls
        }
        
        # Process completed downloads
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed += 1
            
            try:
                result = future.result()
                if result:
                    downloaded_files.append(result)
                    logger.info(f"[{completed}/{total}] Completed: {url.split('/')[-1]}")
                else:
                    logger.error(f"[{completed}/{total}] Failed: {url.split('/')[-1]}")
            except Exception as e:
                logger.error(f"[{completed}/{total}] Error downloading {url}: {e}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(completed, total, url.split('/')[-1])
    
    logger.info(f"Download complete: {len(downloaded_files)}/{total} files downloaded successfully")
    return downloaded_files


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='download_annotations.py')
    parser.add_argument('--annotations_table', default='metadata/annotations.csv')
    parser.add_argument('--output_dir', default='data/annotations')
    args = parser.parse_args()
    
    assert os.path.exists(args.annotations_table)
    assert os.path.isdir(args.output_dir)
    
    annotations = pl.read_csv(args.annotations_table)
    urls = annotations.filter((~pl.col('gtf').is_null()) & (pl.col('liftOver_required') == False)).unique('genome')['gtf'].to_list()
    
    # Define progress callback
    def progress_callback(completed, total, filename):
        print(f"Progress: {completed}/{total} - Completed: {filename}")
    
    # Download with 3 concurrent workers
    downloaded = download_multiple_files(
        urls,
        output_dir=args.output_dir,
        extract=True,
        max_workers=5,
        show_progress=True,
        progress_callback=progress_callback
    )
    
    print(f"\nSuccessfully downloaded {len(downloaded)} files")