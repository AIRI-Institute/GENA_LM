import os
import yaml
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List
import re

def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load YAML file safely."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_header_from_content(content: str) -> str:
    """Extract header from config content by finding the separator."""
    # Look for a separator like "--------------Header end--------------" or similar
    header_end_markers = [
        "--------------Header end--------------",
        "#" * 50,
        "-" * 50,
        "=" * 50
    ]
    
    for marker in header_end_markers:
        if marker in content:
            parts = content.split(marker, 1)
            return parts[0] + marker + "\n"
    
    return ""

def has_same_header(header1: str, header2: str) -> bool:
    """Compare two headers for similarity."""
    # Normalize headers by removing whitespace and comments
    def normalize(header: str) -> str:
        # Remove comments
        lines = []
        for line in header.split('\n'):
            line = line.strip()
            # Skip empty lines and comment-only lines
            if line and not line.startswith('#'):
                lines.append(line)
        return '\n'.join(lines)
    
    return normalize(header1) == normalize(header2)

def update_header_in_config(config_path: Path, new_header: str, split_name: str, backup: bool = True) -> bool:
    """Update or add header to a config file."""
    try:
        # Read the entire file content
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Create backup if requested
        if backup:
            backup_path = config_path.with_suffix('.yaml.backup')
            shutil.copy2(config_path, backup_path)
            print(f"  Created backup: {backup_path}")
        
        # Extract current header if it exists
        current_header = extract_header_from_content(content)
        
        # Check if header exists and is different
        if current_header:
            if has_same_header(current_header, new_header):
                print(f"  Header already matches, no changes needed")
                return True
            else:
                print(f"  Replacing existing header")
                # Remove old header
                body = content[len(current_header):].lstrip('\n')
        else:
            print(f"  Adding new header")
            body = content
        
        # Update TASK_NAME in header with split_name
        updated_header = re.sub(
            r'TASK_NAME:\s*expression/model_expression',
            f'TASK_NAME: expression/{split_name}',
            new_header
        )
        
        # Write updated content
        with open(config_path, 'w') as f:
            f.write(updated_header)
            if body:
                f.write('\n' + body)
        
        return True
        
    except Exception as e:
        print(f"  Error updating {config_path}: {e}")
        return False

def process_config_files(header_file_path: Path, configs_dir_path: Path, 
                        backup: bool = True, dry_run: bool = False) -> None:
    """Process all YAML config files in the directory."""
    
    # Load header file
    print(f"Loading header from: {header_file_path}")
    try:
        with open(header_file_path, 'r') as f:
            header_content = f.read()
    except Exception as e:
        print(f"Error reading header file: {e}")
        return
    
    # Check if configs directory exists
    if not configs_dir_path.exists() or not configs_dir_path.is_dir():
        print(f"Error: Configs directory not found: {configs_dir_path}")
        return
    
    # Find all YAML files
    yaml_files = list(configs_dir_path.glob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files in {configs_dir_path}")
    
    if not yaml_files:
        print("No YAML files found to process")
        return
    
    # Process each file
    successful = 0
    failed = 0
    
    for config_file in sorted(yaml_files):
        print(f"\nProcessing: {config_file.name}")
        
        # Extract split name from filename (remove .yaml extension)
        split_name = config_file.stem
        
        if dry_run:
            print(f"  [DRY RUN] Would update header for split: {split_name}")
            successful += 1
            continue
        
        # Update the header
        if update_header_in_config(config_file, header_content, split_name, backup):
            successful += 1
            print(f"  Updated successfully")
        else:
            failed += 1
            print(f"  Failed to update")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files processed: {len(yaml_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if dry_run:
        print(f"  [DRY RUN MODE - No files were modified]")

def main():
    parser = argparse.ArgumentParser(
        description="Add or replace headers in YAML config files"
    )
    parser.add_argument(
        "header_file",
        default='./common_header.yaml',
        help="Path to the header YAML file"
    )
    parser.add_argument(
        "configs_dir",
        help="Path to directory containing YAML config files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern for config files (default: *.yaml)"
    )
    
    args = parser.parse_args()
    
    header_path = Path(args.header_file)
    configs_dir = Path(args.configs_dir)
    
    if not header_path.exists():
        print(f"Error: Header file not found: {header_path}")
        return
    
    process_config_files(
        header_path,
        configs_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()