"""
Python script to download TitaNet-small model from NVIDIA NGC

This script provides multiple methods to obtain the TitaNet model:
1. Direct download from NGC (requires ngc-cli)
2. Instructions for manual download
3. Check if model already exists
"""
import os
import sys
import subprocess


def check_model_exists():
    """Check if TitaNet model already exists"""
    model_path = "titanet_small.nemo"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ TitaNet model found: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    return False


def download_with_ngc():
    """Download model using NGC CLI"""
    print("Attempting to download with NGC CLI...")
    
    # Check if NGC CLI is installed
    try:
        result = subprocess.run(['ngc', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"NGC CLI version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("NGC CLI not found. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'ngc-cli'],
                         check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install NGC CLI: {e}")
            return False
    
    # Download model
    try:
        print("Downloading TitaNet-small v1.19.0...")
        print("This may take several minutes...")
        
        # Create temp directory
        temp_dir = "/tmp/titanet_download"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download
        result = subprocess.run([
            'ngc', 'registry', 'model', 'download-version',
            'nvidia/nemo/titanet_small:1.19.0',
            '--dest', temp_dir
        ], check=True, capture_output=True, text=True)
        
        # Find the .nemo file
        nemo_file = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.nemo'):
                    nemo_file = os.path.join(root, file)
                    break
            if nemo_file:
                break
        
        if nemo_file:
            # Copy to current directory
            import shutil
            shutil.copy(nemo_file, 'titanet_small.nemo')
            print(f"\n✓ Successfully downloaded TitaNet model!")
            print(f"  Saved to: {os.path.abspath('titanet_small.nemo')}")
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            return True
        else:
            print("ERROR: Could not find .nemo file after download")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def print_manual_instructions():
    """Print manual download instructions"""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\nOption 1: NGC Web Interface")
    print("-" * 40)
    print("1. Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small")
    print("2. Click on 'Download' button")
    print("3. Select version 1.19.0")
    print("4. Download the .nemo file")
    print(f"5. Save as: {os.path.abspath('titanet_small.nemo')}")
    
    print("\nOption 2: Direct Link (requires NGC account)")
    print("-" * 40)
    print("Use NGC CLI after logging in:")
    print("  ngc config set")
    print("  ngc registry model download-version nvidia/nemo/titanet_small:1.19.0")
    
    print("\nOption 3: Alternative NeMo Models")
    print("-" * 40)
    print("If titanet_small is not available, you can try:")
    print("  - titanet_large")
    print("  - ecapa_tdnn")
    print("Note: You may need to adjust embed_dim in config.py accordingly")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    print("="*80)
    print("TitaNet Model Download Helper")
    print("="*80)
    print()
    
    # Check if model already exists
    if check_model_exists():
        response = input("\nModel already exists. Re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Try automatic download
    print("\n" + "-"*80)
    print("Attempting automatic download with NGC CLI...")
    print("-"*80)
    
    success = download_with_ngc()
    
    if not success:
        print("\n" + "-"*80)
        print("Automatic download failed.")
        print("-"*80)
        print_manual_instructions()
    else:
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Test the model:")
        print("   python test_forward.py")
        print("\n2. Start training:")
        print("   python train.py")
        print()


if __name__ == "__main__":
    main()

