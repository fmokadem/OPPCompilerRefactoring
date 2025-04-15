import os
import gc
import numpy as np
import argparse
import re

def parse_solution_line(line):
    """Parses a single line representing a refactoring solution."""
    solution = str()
    try:
        # Handle potential missing parentheses or other format issues
        # Split on comma only if followed by a letter (potential start of new refactoring op)
        # This regex helps handle cases where locations might have commas.
        raw_refs = re.split(r',(?=[a-zA-Z])', line) 
        
        for ref in raw_refs:
            ref = ref.strip()
            if '(' not in ref: # Basic check
                 # print(f"Warning: Skipping malformed ref (no '('): {ref}")
                 continue
                 
            parts = ref.split('(', 1) # Split only on the first '('
            op = parts[0].strip()
            
            if len(parts) < 2 or not parts[1].strip():
                # print(f"Warning: Skipping malformed ref (missing location part): {ref}")
                continue

            loc_part = parts[1].strip()
            # Remove trailing ')' if it exists AT THE VERY END
            if loc_part.endswith(')'):
                 loc_part = loc_part[:-1].strip()

            loc = loc_part.split(';')[:2] # Take only first two parts

            cleaned_loc = []
            for item in loc:
                item_strip = item.strip()
                # Remove "Class " prefix carefully
                if item_strip.startswith("Class ") and len(item_strip) > 6:
                     class_name = item_strip[6:].strip()
                     if class_name:
                         cleaned_loc.append(class_name)
                elif item_strip: # Append only if not empty
                     cleaned_loc.append(item_strip)

            if op and cleaned_loc: 
                 solution += f"{op} {' '.join(cleaned_loc)} " 
                 
        return solution.strip() 
    except Exception as e:
        # print(f"Error parsing solution line: {line.strip()} - {e}")
        return None

def parse_objectives_line(line):
    """Parses a single line representing objective values."""
    try:
        match = re.search(r'\((.*?)\)', line) # Find content within parentheses
        if not match:
             # print(f"Warning: Could not find objective values in parentheses: {line.strip()}")
             return None
             
        objectives_str = match.group(1)
        # Split by comma and handle potential extra spaces
        objectives = [-1 * float(obj.strip()) for obj in objectives_str.split(',') if obj.strip()] 
        
        if len(objectives) >= 6:
            return objectives[:6]
        else:
            # print(f"Warning: Objective line has fewer than 6 values ({len(objectives)} found): {line.strip()}")
            return None
    except Exception as e:
        # print(f"Error parsing objectives line: {line.strip()} - {e}")
        return None

def process_system_data(system_name):
    """Processes execution traces for a given system."""
    input_dir = os.path.join("data", system_name)
    output_dir = "processed_data"
    output_filename = f"FinalSolutions-multilabel-{system_name}.txt"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Processing system: {system_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_path}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Look for .txt files specifically
    data_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.txt')]
    if not data_files:
        print(f"Warning: No .txt data files found in {input_dir}")
        return

    all_single_solutions = []

    for filename in data_files:
        filepath = os.path.join(input_dir, filename)
        print(f"\nProcessing file: {filepath}")
        
        solutions = []
        objectives_list = []
        line_num = 0

        try:
            # Specify encoding and handle errors
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: 
                while True:
                    line_num += 1
                    solution_line = f.readline()
                    if not solution_line: # End of file
                        break 

                    line_num += 1
                    objectives_line = f.readline()
                    if not objectives_line: # Paired line missing
                        # print(f"Warning: Missing objective line after solution line {line_num-1} in {filename}")
                        break

                    parsed_solution = parse_solution_line(solution_line)
                    parsed_objectives = parse_objectives_line(objectives_line)

                    # Ensure both parts were parsed successfully and are not empty strings
                    if parsed_solution and parsed_objectives is not None: 
                        solutions.append(parsed_solution)
                        objectives_list.append(parsed_objectives)
                    # else:
                        # print(f"Skipping lines {line_num-1}-{line_num} in {filename} due to parsing errors or empty results.")

        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue # Skip to the next file

        if not solutions or not objectives_list:
            print(f"Warning: No valid solution/objective pairs found in {filename}")
            continue

        print(f"File {filename} found {len(solutions)} solution/objective pairs.")
        
        objectives_np = np.array(objectives_list)
        threshold = np.zeros(6) # Keep threshold as all zeros
        
        single_solutions_file = []
        count_improved = 0
        for i, sol in enumerate(solutions):
             if i >= len(objectives_np): # Safety check for array bounds
                  print(f"Warning: Index out of bounds accessing objectives at index {i} for solution '{sol[:50]}...' in {filename}")
                  continue
             obj = objectives_np[i]
             # Ensure obj has 6 elements before indexing
             if len(obj) == 6:
                  label = [1 if obj[x] > threshold[x] else 0 for x in range(6)]
                  if sum(label) > 0:
                       single_solutions_file.append(sol + " " + ", ".join(map(str, label))) 
                       count_improved += 1
             else:
                  # print(f"Warning: Objective data has incorrect length ({len(obj)}) at index {i} for solution '{sol[:50]}...' in {filename}")
                  continue
                
        print(f"Total of {count_improved} solutions improved at least one objective in this file.")
        all_single_solutions.extend(single_solutions_file)
        
        # Clean up memory for the current file
        del objectives_np, solutions, objectives_list, single_solutions_file
        gc.collect()

    # --- Post-loop processing ---
    if not all_single_solutions:
        print(f"\nNo solutions improving objectives found across all files for system {system_name}.")
        return

    print(f"\nTotal solutions improving objectives across all files for {system_name}: {len(all_single_solutions)}")
    
    # Remove duplicates across all files
    unique_single_solutions = sorted(list(set(all_single_solutions)))
    print(f"Total unique single solutions for {system_name}: {len(unique_single_solutions)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    try:
        # Overwrite existing file for the system ('w' mode) and specify encoding
        with open(output_path, 'w', encoding='utf-8') as f: 
            f.write("\n".join(unique_single_solutions) + "\n")
        print(f"Successfully saved unique solutions to {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess refactoring execution traces.')
    parser.add_argument('system_name', type=str, 
                        help='Name of the system to process (e.g., jhotdraw, ant). Assumes data is in data/<system_name>/ and files are .txt.')
    args = parser.parse_args()
    
    process_system_data(args.system_name) 