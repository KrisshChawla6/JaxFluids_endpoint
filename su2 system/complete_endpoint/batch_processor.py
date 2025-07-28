#!/usr/bin/env python3
"""
Batch Processor for Complete Wind Tunnel Endpoint
Processes multiple JSON configuration files in sequence or parallel
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from complete_wind_tunnel_endpoint import process_json_request, CompleteWindTunnelResponse

class BatchProcessor:
    """Batch processor for multiple wind tunnel configurations"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.results = []
    
    def process_single_config(self, config_file: str) -> Dict[str, Any]:
        """Process a single configuration file"""
        
        start_time = time.time()
        
        try:
            print(f"🚀 Processing: {config_file}")
            result = process_json_request(config_file)
            
            processing_time = time.time() - start_time
            
            return {
                'config_file': config_file,
                'success': result.success,
                'message': result.message,
                'simulation_directory': result.simulation_directory,
                'output_files': result.output_files,
                'processing_time': processing_time,
                'wind_tunnel_time': result.wind_tunnel_time,
                'config_generation_time': result.config_generation_time,
                'total_time': result.total_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'config_file': config_file,
                'success': False,
                'message': f"Exception during processing: {str(e)}",
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def process_sequential(self, config_files: List[str]) -> List[Dict[str, Any]]:
        """Process configuration files sequentially"""
        
        print(f"🔄 Processing {len(config_files)} configurations sequentially...")
        
        results = []
        for i, config_file in enumerate(config_files, 1):
            print(f"\n📁 [{i}/{len(config_files)}] {config_file}")
            result = self.process_single_config(config_file)
            results.append(result)
            
            if result['success']:
                print(f"   ✅ Completed in {result['processing_time']:.2f}s")
            else:
                print(f"   ❌ Failed: {result['message']}")
        
        return results
    
    def process_parallel(self, config_files: List[str]) -> List[Dict[str, Any]]:
        """Process configuration files in parallel"""
        
        print(f"⚡ Processing {len(config_files)} configurations in parallel (max {self.max_workers} workers)...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self.process_single_config, config_file): config_file 
                for config_file in config_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config_file = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        print(f"✅ {config_file}: Completed in {result['processing_time']:.2f}s")
                    else:
                        print(f"❌ {config_file}: Failed - {result['message']}")
                        
                except Exception as e:
                    print(f"❌ {config_file}: Exception - {str(e)}")
                    results.append({
                        'config_file': config_file,
                        'success': False,
                        'message': f"Future exception: {str(e)}",
                        'error': str(e)
                    })
        
        # Sort results by original order
        config_order = {config: i for i, config in enumerate(config_files)}
        results.sort(key=lambda x: config_order.get(x['config_file'], 999))
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]], output_file: str = None):
        """Generate a summary report of batch processing results"""
        
        total_configs = len(results)
        successful_configs = sum(1 for r in results if r['success'])
        failed_configs = total_configs - successful_configs
        
        total_time = sum(r.get('processing_time', 0) for r in results)
        avg_time = total_time / total_configs if total_configs > 0 else 0
        
        # Create summary
        summary = {
            'batch_processing_summary': {
                'total_configurations': total_configs,
                'successful': successful_configs,
                'failed': failed_configs,
                'success_rate': f"{(successful_configs/total_configs*100):.1f}%" if total_configs > 0 else "0%",
                'total_processing_time': f"{total_time:.2f}s",
                'average_time_per_config': f"{avg_time:.2f}s"
            },
            'detailed_results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"📄 Summary report saved to: {output_file}")
        
        # Print summary to console
        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"=" * 50)
        print(f"Total configurations: {total_configs}")
        print(f"Successful: {successful_configs}")
        print(f"Failed: {failed_configs}")
        print(f"Success rate: {(successful_configs/total_configs*100):.1f}%" if total_configs > 0 else "0%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per config: {avg_time:.2f}s")
        
        if failed_configs > 0:
            print(f"\n❌ Failed configurations:")
            for result in results:
                if not result['success']:
                    print(f"   • {result['config_file']}: {result['message']}")
        
        return summary

def find_json_files(directory: str) -> List[str]:
    """Find all JSON files in a directory"""
    
    json_files = []
    for file_path in Path(directory).glob("*.json"):
        json_files.append(str(file_path))
    
    return sorted(json_files)

def create_example_configs(output_dir: str = "example_configs"):
    """Create example configuration files for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Example configurations
    configs = [
        {
            "name": "euler_low_speed",
            "config": {
                "object_mesh_file": "../packaged_wind-tunnel_endpoint/propeller_only.vtk",
                "tunnel_type": "compact",
                "flow_type": "EULER",
                "mach_number": 0.2,
                "angle_of_attack": 0.0,
                "max_iterations": 50,
                "output_directory": "batch_output_euler_low",
                "simulation_name": "euler_low_speed"
            }
        },
        {
            "name": "euler_medium_speed",
            "config": {
                "object_mesh_file": "../packaged_wind-tunnel_endpoint/propeller_only.vtk",
                "tunnel_type": "standard",
                "flow_type": "EULER",
                "mach_number": 0.3,
                "angle_of_attack": 5.0,
                "max_iterations": 75,
                "output_directory": "batch_output_euler_medium",
                "simulation_name": "euler_medium_speed"
            }
        },
        {
            "name": "euler_high_aoa",
            "config": {
                "object_mesh_file": "../packaged_wind-tunnel_endpoint/propeller_only.vtk",
                "tunnel_type": "standard",
                "flow_type": "EULER",
                "mach_number": 0.25,
                "angle_of_attack": 10.0,
                "max_iterations": 100,
                "output_directory": "batch_output_euler_high_aoa",
                "simulation_name": "euler_high_aoa"
            }
        },
        {
            "name": "prompt_based",
            "config": {
                "object_mesh_file": "../packaged_wind-tunnel_endpoint/propeller_only.vtk",
                "tunnel_type": "compact",
                "prompt": "Create a quick Euler simulation at Mach 0.15 with 3 degrees angle of attack and 60 iterations",
                "output_directory": "batch_output_prompt",
                "simulation_name": "prompt_based"
            }
        }
    ]
    
    created_files = []
    for config_info in configs:
        config_file = output_path / f"{config_info['name']}.json"
        with open(config_file, 'w') as f:
            json.dump(config_info['config'], f, indent=2)
        created_files.append(str(config_file))
        print(f"📄 Created: {config_file}")
    
    return created_files

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description="Batch processor for Complete Wind Tunnel Endpoint")
    parser.add_argument("input", nargs='?', help="JSON config file, directory with JSON files, or 'create-examples'")
    parser.add_argument("--parallel", action="store_true", help="Process configurations in parallel")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers (default: 2)")
    parser.add_argument("--output", type=str, help="Output file for summary report")
    
    args = parser.parse_args()
    
    if not args.input:
        print("Usage: python batch_processor.py <input> [options]")
        print("\nInput options:")
        print("  config.json          - Process single JSON file")
        print("  config_directory/    - Process all JSON files in directory")
        print("  create-examples      - Create example configuration files")
        print("\nOptions:")
        print("  --parallel           - Process in parallel")
        print("  --workers N          - Number of parallel workers")
        print("  --output file.json   - Save summary report to file")
        sys.exit(1)
    
    # Handle special case: create examples
    if args.input == "create-examples":
        print("🏗️ Creating example configuration files...")
        created_files = create_example_configs()
        print(f"\n✅ Created {len(created_files)} example configuration files")
        print("Run: python batch_processor.py example_configs/ --parallel")
        sys.exit(0)
    
    # Determine input files
    if os.path.isfile(args.input):
        # Single file
        config_files = [args.input]
    elif os.path.isdir(args.input):
        # Directory
        config_files = find_json_files(args.input)
        if not config_files:
            print(f"❌ No JSON files found in directory: {args.input}")
            sys.exit(1)
    else:
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)
    
    print(f"🚀 Batch Processing Complete Wind Tunnel Configurations")
    print(f"=" * 60)
    print(f"📁 Found {len(config_files)} configuration files")
    print(f"⚡ Mode: {'Parallel' if args.parallel else 'Sequential'}")
    if args.parallel:
        print(f"👥 Workers: {args.workers}")
    
    # Process configurations
    processor = BatchProcessor(max_workers=args.workers)
    
    start_time = time.time()
    
    if args.parallel and len(config_files) > 1:
        results = processor.process_parallel(config_files)
    else:
        results = processor.process_sequential(config_files)
    
    total_batch_time = time.time() - start_time
    
    # Generate summary
    print(f"\n⏱️ Total batch processing time: {total_batch_time:.2f}s")
    
    summary_file = args.output or "batch_summary.json"
    processor.generate_summary_report(results, summary_file)
    
    # Exit with appropriate code
    successful_count = sum(1 for r in results if r['success'])
    if successful_count == len(results):
        print(f"\n🎉 All configurations processed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {len(results) - successful_count} configurations failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 