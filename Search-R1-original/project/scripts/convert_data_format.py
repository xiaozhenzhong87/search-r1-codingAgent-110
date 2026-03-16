#!/usr/bin/env python3
"""
Convert JSON data to Parquet format for veRL training
"""
import json
import pandas as pd
import sys
import argparse

def convert_json_to_parquet(json_file: str, output_file: str = None):
    """Convert JSON to Parquet"""
    if output_file is None:
        output_file = json_file.replace('.json', '.parquet')
    
    print(f"Converting {json_file} to parquet...")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df.to_parquet(output_file, index=False)
        
        print(f"✓ Saved to {output_file}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        return output_file
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON to Parquet')
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output parquet file (optional)')
    
    args = parser.parse_args()
    convert_json_to_parquet(args.input, args.output)
