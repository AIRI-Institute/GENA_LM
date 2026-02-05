import os
import re
import argparse
from pathlib import Path
import shutil

def generate_bash_script(template_path, yaml_dir, output_dir, config_name=None, yaml_path=None):
    """
    Generate a bash script from template.
    
    Args:
        template_path: Path to the template bash script
        yaml_dir: Directory containing YAML config files
        output_dir: Directory to write generated bash scripts
        config_name: If provided, use this specific config
        yaml_path: If provided, use this specific YAML file path
    """
    # Read the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which YAML files to process
    if yaml_path:
        # Single YAML file specified
        yaml_files = [Path(yaml_path)]
    elif config_name:
        # Specific config name
        yaml_file = Path(yaml_dir) / f"{config_name}.yaml"
        if not yaml_file.exists():
            # Try with .yml extension
            yaml_file = Path(yaml_dir) / f"{config_name}.yml"
            if not yaml_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_name} in {yaml_dir}")
        yaml_files = [yaml_file]
    else:
        # All YAML files in directory
        yaml_dir = Path(yaml_dir)
        yaml_files = list(yaml_dir.glob("*.yaml")) + list(yaml_dir.glob("*.yml"))
        yaml_files = sorted(yaml_files)
    
    generated_files = []
    
    for yaml_file in yaml_files:
        # Get config name from filename
        config_name = yaml_file.stem
        
        # Replace the experiment config path in template
        # Find the line with --experiment_config
        lines = template.split('\n')
        for i, line in enumerate(lines):
            if '--experiment_config' in line:
                # Extract the current config path from the line
                match = re.search(r'--experiment_config\s+"([^"]+)"', line)
                if match:
                    current_config_path = match.group(1)
                    # Replace only the config name part
                    new_config_path = str(yaml_file.absolute())
                    # Replace the entire path in the line
                    lines[i] = line.replace(current_config_path, new_config_path)
                break
        
        # Reconstruct the script
        script_content = '\n'.join(lines)
        
        # Write to output file
        output_file = output_dir / f"slurm_finetune_expression_{config_name}.sh"
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        output_file.chmod(0o755)
        
        generated_files.append(str(output_file))
        
        print(f"Generated: {output_file}")
        print(f"  Using config: {yaml_file}")
    
    return generated_files

def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM bash scripts for fine-tuning from YAML configs"
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the template bash script"
    )
    parser.add_argument(
        "--yaml-dir",
        required=True,
        help="Directory containing YAML config files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write generated bash scripts"
    )
    parser.add_argument(
        "--config",
        help="Specific config name (without extension) to process"
    )
    parser.add_argument(
        "--yaml-file",
        help="Specific YAML file path to use (overrides --config and --yaml-dir)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.template).exists():
        print(f"Error: Template file not found: {args.template}")
        return
    
    if args.yaml_file and not Path(args.yaml_file).exists():
        print(f"Error: YAML file not found: {args.yaml_file}")
        return
    
    if not args.yaml_file and not Path(args.yaml_dir).exists():
        print(f"Error: YAML directory not found: {args.yaml_dir}")
        return
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print(f"Template: {args.template}")
        print(f"YAML dir: {args.yaml_dir}")
        print(f"Output dir: {args.output_dir}")
        
        if args.yaml_file:
            print(f"Specific YAML file: {args.yaml_file}")
            config_name = Path(args.yaml_file).stem
            print(f"Would generate: slurm_finetune_expression_{config_name}.sh")
        elif args.config:
            print(f"Specific config: {args.config}")
            print(f"Would generate: slurm_finetune_expression_{args.config}.sh")
        else:
            yaml_dir = Path(args.yaml_dir)
            yaml_files = list(yaml_dir.glob("*.yaml")) + list(yaml_dir.glob("*.yml"))
            print(f"Found {len(yaml_files)} YAML files:")
            for yaml_file in sorted(yaml_files):
                config_name = yaml_file.stem
                print(f"  - {config_name} -> slurm_finetune_expression_{config_name}.sh")
        return
    
    try:
        generated_files = generate_bash_script(
            template_path=args.template,
            yaml_dir=args.yaml_dir,
            output_dir=args.output_dir,
            config_name=args.config,
            yaml_path=args.yaml_file
        )
        
        print(f"\nSuccessfully generated {len(generated_files)} bash scripts in {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()