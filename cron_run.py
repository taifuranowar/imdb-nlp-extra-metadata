#!/usr/bin/env python3
# filepath: run_training_scripts.py
import os
import glob
import subprocess
import sys
import time
from datetime import datetime

def get_train_scripts():
    """Find all Python scripts that start with 'train_'"""
    # Get all Python files starting with 'train_'
    scripts = glob.glob('train_*.py')
    scripts.sort()  # Sort them alphabetically
    return scripts

def display_scripts(scripts):
    """Display the scripts with numbered indices"""
    print("\n=== Available Training Scripts ===")
    for i, script in enumerate(scripts, 1):
        print(f"[{i}] {script}")
    print("===============================\n")

def get_user_selection(scripts):
    """Get user input for which scripts to run"""
    max_index = len(scripts)
    while True:
        try:
            selection = input("Enter script numbers to run (comma separated, e.g., '1,3,5'): ")
            indices = [int(x.strip()) for x in selection.split(',')]
            
            # Validate indices
            invalid_indices = [idx for idx in indices if idx < 1 or idx > max_index]
            if invalid_indices:
                print(f"Invalid indices: {', '.join(map(str, invalid_indices))}. Valid range: 1-{max_index}")
                continue
                
            # Return the selected scripts
            return [scripts[idx-1] for idx in indices]
        except ValueError:
            print("Please enter valid numbers separated by commas")

def run_script(script_path):
    """Run a single script and show its output interactively"""
    print(f"\n\n{'='*80}")
    print(f"RUNNING: {script_path}")
    print(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        # Run the script with real-time output
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line-buffered
        )
        
        # Print output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
                sys.stdout.flush()
        
        # Get the return code
        return_code = process.poll()
        
        print(f"\n{'='*80}")
        print(f"FINISHED: {script_path}")
        print(f"END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"RETURN CODE: {return_code}")
        print(f"{'='*80}\n")
        
        return return_code == 0
    except Exception as e:
        print(f"\nERROR running {script_path}: {str(e)}")
        return False

def main():
    print("Training Script Runner")
    print("This tool will run selected training scripts sequentially")
    
    # Get all training scripts
    scripts = get_train_scripts()
    
    if not scripts:
        print("No training scripts found in the current directory!")
        return
    
    # Display scripts and get user selection
    display_scripts(scripts)
    selected_scripts = get_user_selection(scripts)
    
    if not selected_scripts:
        print("No scripts selected. Exiting.")
        return
    
    print(f"\nWill run {len(selected_scripts)} scripts in sequence:")
    for i, script in enumerate(selected_scripts, 1):
        print(f"  {i}. {script}")
    
    # Confirm before running
    confirm = input("\nProceed with execution? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Execution cancelled.")
        return
    
    # Run selected scripts
    start_time = time.time()
    successful = 0
    failed = 0
    
    for script in selected_scripts:
        success = run_script(script)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n=== EXECUTION SUMMARY ===")
    print(f"Total scripts: {len(selected_scripts)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=========================")

if __name__ == "__main__":
    main()