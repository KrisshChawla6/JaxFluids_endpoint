#!/usr/bin/env python3
"""
Comprehensive Debug and Test Suite for Complete Wind Tunnel Endpoint
Provides detailed diagnostics, validation, and troubleshooting capabilities
"""

import os
import sys
import json
import time
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from complete_wind_tunnel_endpoint import (
    CompleteWindTunnelEndpoint,
    CompleteWindTunnelRequest,
    CompleteWindTunnelResponse,
    process_json_request
)

class ComprehensiveDebugger:
    """Comprehensive debugging and testing system"""
    
    def __init__(self):
        self.test_results = []
        self.debug_log = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log a debug message"""
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.debug_log.append(log_entry)
        print(log_entry)
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate the environment and dependencies"""
        
        self.log("🔍 Validating Environment", "INFO")
        
        validation = {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'path_accessible': True,
            'dependencies': {},
            'file_permissions': {},
            'issues': []
        }
        
        # Check Python version
        if sys.version_info < (3, 8):
            validation['issues'].append("Python 3.8+ required")
        
        # Check dependencies
        required_modules = ['numpy', 'pathlib', 'json', 'time', 'shutil']
        for module in required_modules:
            try:
                __import__(module)
                validation['dependencies'][module] = "✅ Available"
            except ImportError:
                validation['dependencies'][module] = "❌ Missing"
                validation['issues'].append(f"Missing module: {module}")
        
        # Check file permissions
        try:
            test_file = Path("test_permissions.tmp")
            test_file.write_text("test")
            test_file.unlink()
            validation['file_permissions']['write'] = "✅ OK"
        except Exception as e:
            validation['file_permissions']['write'] = f"❌ Error: {e}"
            validation['issues'].append("No write permissions")
        
        self.log(f"Environment validation: {len(validation['issues'])} issues found")
        return validation
    
    def validate_input_file(self, file_path: str) -> Dict[str, Any]:
        """Validate input mesh file"""
        
        self.log(f"🔍 Validating Input File: {file_path}", "INFO")
        
        validation = {
            'exists': False,
            'readable': False,
            'size_mb': 0,
            'format': 'unknown',
            'content_valid': False,
            'issues': []
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            validation['issues'].append("File does not exist")
            return validation
        
        validation['exists'] = True
        
        # Check if readable
        try:
            with open(file_path, 'r') as f:
                f.read(100)  # Read first 100 chars
            validation['readable'] = True
        except Exception as e:
            validation['issues'].append(f"File not readable: {e}")
            return validation
        
        # Get file size
        try:
            size_bytes = os.path.getsize(file_path)
            validation['size_mb'] = size_bytes / (1024 * 1024)
        except Exception as e:
            validation['issues'].append(f"Cannot get file size: {e}")
        
        # Determine format
        if file_path.lower().endswith('.su2'):
            validation['format'] = 'SU2'
            validation['content_valid'] = self._validate_su2_content(file_path)
        elif file_path.lower().endswith('.vtk'):
            validation['format'] = 'VTK'
            validation['content_valid'] = self._validate_vtk_content(file_path)
        else:
            validation['issues'].append("Unknown file format")
        
        self.log(f"Input validation: {validation['format']} format, {validation['size_mb']:.1f}MB")
        return validation
    
    def _validate_su2_content(self, file_path: str) -> bool:
        """Validate SU2 file content"""
        try:
            with open(file_path, 'r') as f:
                content = f.read(1000)  # Read first 1000 chars
                
            # Check for required SU2 sections
            required_sections = ['NDIME=', 'NELEM=', 'NPOIN=']
            for section in required_sections:
                if section not in content:
                    self.log(f"Missing SU2 section: {section}", "WARNING")
                    return False
            
            return True
        except Exception as e:
            self.log(f"SU2 content validation error: {e}", "ERROR")
            return False
    
    def _validate_vtk_content(self, file_path: str) -> bool:
        """Validate VTK file content"""
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                
            # Check VTK header
            if not first_line.startswith('# vtk DataFile Version'):
                self.log("Invalid VTK header", "WARNING")
                return False
            
            return True
        except Exception as e:
            self.log(f"VTK content validation error: {e}", "ERROR")
            return False
    
    def validate_output_directory(self, output_dir: str) -> Dict[str, Any]:
        """Validate output directory"""
        
        self.log(f"🔍 Validating Output Directory: {output_dir}", "INFO")
        
        validation = {
            'exists': False,
            'writable': False,
            'space_available': True,
            'issues': []
        }
        
        # Check if directory exists or can be created
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            validation['exists'] = True
        except Exception as e:
            validation['issues'].append(f"Cannot create directory: {e}")
            return validation
        
        # Check if writable
        try:
            test_file = Path(output_dir) / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            validation['writable'] = True
        except Exception as e:
            validation['issues'].append(f"Directory not writable: {e}")
        
        # Check available space (simplified)
        try:
            import shutil
            total, used, free = shutil.disk_usage(output_dir)
            free_gb = free / (1024**3)
            if free_gb < 1.0:  # Less than 1GB
                validation['space_available'] = False
                validation['issues'].append(f"Low disk space: {free_gb:.1f}GB available")
        except Exception as e:
            self.log(f"Cannot check disk space: {e}", "WARNING")
        
        return validation
    
    def run_comprehensive_test(self, config_file: str) -> Dict[str, Any]:
        """Run comprehensive test with detailed diagnostics"""
        
        self.log("🚀 Starting Comprehensive Test", "INFO")
        
        test_result = {
            'success': False,
            'config_file': config_file,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'environment_validation': {},
            'input_validation': {},
            'output_validation': {},
            'processing_result': {},
            'output_files': [],
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Validate environment
            test_result['environment_validation'] = self.validate_environment()
            
            # Step 2: Load and validate configuration
            if not os.path.exists(config_file):
                test_result['issues'].append(f"Config file not found: {config_file}")
                return test_result
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Step 3: Validate input file
            input_file = config.get('object_mesh_file', '')
            test_result['input_validation'] = self.validate_input_file(input_file)
            
            # Step 4: Validate output directory
            output_dir = config.get('output_directory', 'default_output')
            test_result['output_validation'] = self.validate_output_directory(output_dir)
            
            # Step 5: Check for critical issues
            critical_issues = []
            if test_result['environment_validation']['issues']:
                critical_issues.extend(test_result['environment_validation']['issues'])
            if test_result['input_validation']['issues']:
                critical_issues.extend(test_result['input_validation']['issues'])
            if test_result['output_validation']['issues']:
                critical_issues.extend(test_result['output_validation']['issues'])
            
            if critical_issues:
                test_result['issues'] = critical_issues
                self.log(f"❌ Critical issues found: {len(critical_issues)}", "ERROR")
                for issue in critical_issues:
                    self.log(f"   • {issue}", "ERROR")
                return test_result
            
            # Step 6: Run the actual processing
            self.log("🏃 Running Complete Wind Tunnel Processing", "INFO")
            
            processing_start = time.time()
            result = process_json_request(config_file)
            processing_time = time.time() - processing_start
            
            test_result['processing_result'] = {
                'success': result.success,
                'message': result.message,
                'wind_tunnel_file': result.wind_tunnel_file,
                'vtk_file': result.vtk_file,
                'config_file': result.config_file,
                'simulation_directory': result.simulation_directory,
                'total_time': result.total_time,
                'processing_time': processing_time,
                'output_files': result.output_files or []
            }
            
            # Step 7: Validate outputs
            if result.success:
                test_result['output_files'] = self._validate_output_files(result)
                test_result['success'] = True
                self.log("✅ Processing completed successfully", "INFO")
            else:
                test_result['issues'].append(f"Processing failed: {result.message}")
                self.log(f"❌ Processing failed: {result.message}", "ERROR")
            
        except Exception as e:
            error_msg = f"Test failed with exception: {str(e)}"
            test_result['issues'].append(error_msg)
            self.log(error_msg, "ERROR")
            self.log(traceback.format_exc(), "DEBUG")
        
        finally:
            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']
        
        # Generate recommendations
        test_result['recommendations'] = self._generate_recommendations(test_result)
        
        return test_result
    
    def _validate_output_files(self, result: CompleteWindTunnelResponse) -> List[Dict[str, Any]]:
        """Validate generated output files"""
        
        self.log("🔍 Validating Output Files", "INFO")
        
        file_validations = []
        
        files_to_check = [
            ('wind_tunnel_mesh', result.wind_tunnel_file),
            ('vtk_visualization', result.vtk_file),
            ('config_file', result.config_file),
        ]
        
        for file_type, file_path in files_to_check:
            if not file_path:
                continue
                
            validation = {
                'type': file_type,
                'path': file_path,
                'exists': False,
                'size_mb': 0,
                'readable': False,
                'issues': []
            }
            
            if os.path.exists(file_path):
                validation['exists'] = True
                
                try:
                    size_bytes = os.path.getsize(file_path)
                    validation['size_mb'] = size_bytes / (1024 * 1024)
                except Exception as e:
                    validation['issues'].append(f"Cannot get file size: {e}")
                
                try:
                    with open(file_path, 'r') as f:
                        f.read(100)
                    validation['readable'] = True
                except Exception as e:
                    validation['issues'].append(f"File not readable: {e}")
            else:
                validation['issues'].append("File does not exist")
            
            file_validations.append(validation)
            self.log(f"   {file_type}: {'✅' if validation['exists'] else '❌'} {validation['size_mb']:.1f}MB")
        
        return file_validations
    
    def _generate_recommendations(self, test_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Environment recommendations
        env_issues = test_result['environment_validation'].get('issues', [])
        if env_issues:
            recommendations.append("Install missing dependencies with: pip install -r requirements.txt")
        
        # Input file recommendations
        input_issues = test_result['input_validation'].get('issues', [])
        if input_issues:
            recommendations.append("Verify input mesh file path and format")
            recommendations.append("Ensure input file is a valid SU2 or VTK mesh")
        
        # Performance recommendations
        if test_result.get('duration', 0) > 60:
            recommendations.append("Consider using 'coarse' mesh quality for faster processing")
            recommendations.append("Use 'compact' tunnel type for smaller domains")
        
        # Output recommendations
        if test_result['success']:
            recommendations.append("Run SU2_CFD in the simulation directory to start the CFD simulation")
            recommendations.append("Use ParaView to visualize the VTK files")
        
        return recommendations
    
    def generate_debug_report(self, test_result: Dict[str, Any], output_file: str = None):
        """Generate comprehensive debug report"""
        
        report = {
            'test_summary': {
                'success': test_result['success'],
                'duration': f"{test_result.get('duration', 0):.2f}s",
                'issues_count': len(test_result.get('issues', [])),
                'recommendations_count': len(test_result.get('recommendations', []))
            },
            'detailed_results': test_result,
            'debug_log': self.debug_log,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.log(f"📄 Debug report saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DEBUG REPORT")
        print("="*60)
        print(f"Success: {'✅' if test_result['success'] else '❌'}")
        print(f"Duration: {test_result.get('duration', 0):.2f}s")
        print(f"Issues: {len(test_result.get('issues', []))}")
        print(f"Recommendations: {len(test_result.get('recommendations', []))}")
        
        if test_result.get('issues'):
            print(f"\n❌ Issues Found:")
            for issue in test_result['issues']:
                print(f"   • {issue}")
        
        if test_result.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in test_result['recommendations']:
                print(f"   • {rec}")
        
        return report

def create_test_configurations() -> List[str]:
    """Create various test configurations for debugging"""
    
    test_configs = []
    
    # Test 1: Basic configuration
    basic_config = {
        "object_mesh_file": "C:\\Users\\kriss\\Desktop\\simulation_agent_endpoint\\projects\\propeller\\mesh\\5_bladed_Propeller_medium_tetrahedral.su2",
        "tunnel_type": "compact",
        "flow_type": "EULER",
        "mach_number": 0.2,
        "angle_of_attack": 0.0,
        "max_iterations": 50,
        "output_directory": "debug_test_basic",
        "simulation_name": "basic_test"
    }
    
    # Test 2: High-fidelity configuration
    hifi_config = {
        "object_mesh_file": "C:\\Users\\kriss\\Desktop\\simulation_agent_endpoint\\projects\\propeller\\mesh\\5_bladed_Propeller_medium_tetrahedral.su2",
        "tunnel_type": "standard",
        "flow_type": "RANS",
        "turbulence_model": "SA",
        "mach_number": 0.3,
        "reynolds_number": 5000000,
        "angle_of_attack": 5.0,
        "max_iterations": 200,
        "mesh_quality": "fine",
        "output_directory": "debug_test_hifi",
        "simulation_name": "hifi_test"
    }
    
    # Test 3: Natural language prompt
    prompt_config = {
        "object_mesh_file": "C:\\Users\\kriss\\Desktop\\simulation_agent_endpoint\\projects\\propeller\\mesh\\5_bladed_Propeller_medium_tetrahedral.su2",
        "prompt": "Create a quick Euler simulation at Mach 0.25 with 3 degrees angle of attack and 75 iterations",
        "output_directory": "debug_test_prompt",
        "simulation_name": "prompt_test"
    }
    
    configs = [
        ("debug_basic_config.json", basic_config),
        ("debug_hifi_config.json", hifi_config),
        ("debug_prompt_config.json", prompt_config)
    ]
    
    for filename, config in configs:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        test_configs.append(filename)
        print(f"📄 Created test config: {filename}")
    
    return test_configs

def main():
    """Main function for comprehensive debugging"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Debug and Test Suite")
    parser.add_argument("config", nargs='?', help="JSON configuration file to test")
    parser.add_argument("--create-tests", action="store_true", help="Create test configuration files")
    parser.add_argument("--run-all-tests", action="store_true", help="Run all test configurations")
    parser.add_argument("--report", type=str, help="Output file for debug report")
    
    args = parser.parse_args()
    
    debugger = ComprehensiveDebugger()
    
    if args.create_tests:
        print("🏗️ Creating test configuration files...")
        test_configs = create_test_configurations()
        print(f"\n✅ Created {len(test_configs)} test configurations")
        print("Run with: python comprehensive_debug_test.py --run-all-tests")
        return
    
    if args.run_all_tests:
        print("🚀 Running all test configurations...")
        test_files = ["debug_basic_config.json", "debug_hifi_config.json", "debug_prompt_config.json"]
        
        all_results = []
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n🧪 Testing: {test_file}")
                result = debugger.run_comprehensive_test(test_file)
                all_results.append(result)
            else:
                print(f"❌ Test file not found: {test_file}")
        
        # Generate combined report
        report_file = args.report or "comprehensive_debug_report.json"
        combined_report = {
            'test_results': all_results,
            'summary': {
                'total_tests': len(all_results),
                'successful_tests': sum(1 for r in all_results if r['success']),
                'failed_tests': sum(1 for r in all_results if not r['success'])
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(combined_report, f, indent=2, default=str)
        
        print(f"\n📄 Combined report saved to: {report_file}")
        return
    
    if not args.config:
        print("Usage: python comprehensive_debug_test.py <config.json>")
        print("   or: python comprehensive_debug_test.py --create-tests")
        print("   or: python comprehensive_debug_test.py --run-all-tests")
        return
    
    # Run single test
    print(f"🧪 Running comprehensive test for: {args.config}")
    result = debugger.run_comprehensive_test(args.config)
    
    # Generate report
    report_file = args.report or f"debug_report_{int(time.time())}.json"
    debugger.generate_debug_report(result, report_file)

if __name__ == "__main__":
    main() 