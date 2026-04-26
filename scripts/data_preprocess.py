import sys

def convert_to_dense_csv(input_file, output_file, num_features=3072, row_limit=10000):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for i, line in enumerate(infile):
                if i >= row_limit:
                    break
                
                parts = line.split()
                if not parts:
                    continue
                
                # 1. Initialize a "dense" row with zeros
                # Using strings so we can join them easily later
                dense_row = ["0"] * num_features
                
                # 2. Fill in the non-zero values found in the LIBSVM file
                # parts[0] is the label; parts[1:] are the index:value pairs
                for item in parts[1:]:
                    if ':' in item:
                        idx_str, val_str = item.split(':')
                        idx = int(idx_str) - 1  # LIBSVM indices usually start at 1
                        
                        if 0 <= idx < num_features:
                            dense_row[idx] = val_str
                
                # 3. Write the full dense row to CSV
                outfile.write(",".join(dense_row) + "\n")
                
        print(f"Done! Saved {i} rows with {num_features} columns to {output_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except ValueError as e:
        print(f"Data Error: Ensure indices are integers. {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert.py <input> <output> [num_features] [row_limit]")
        print("Default: 3072 features, 10000 rows (optimized for CIFAR-10)")
    else:
        # Get arguments or use defaults
        inp = sys.argv[1]
        out = sys.argv[2]
        feats = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
        limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
        
        convert_to_dense_csv(inp, out, feats, limit)
