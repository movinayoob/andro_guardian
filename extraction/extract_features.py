from collections import Counter
import csv
import os

def load_syscall_dict(dict_file_path):
    with open(dict_file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def extract_syscall_frequencies(trace_file_path, syscall_dict):
    freq_counter = Counter()
    with open(trace_file_path, 'r', errors='ignore') as file:
        for line in file:
            if '(' in line:
                syscall = line.split('(')[0].strip()
                if syscall in syscall_dict:
                    freq_counter[syscall] += 1
    return freq_counter

# Create feature vector
def create_feature_vector(freq_counter, syscall_dict, label="benign"):
    vector = [freq_counter.get(syscall, 0) for syscall in syscall_dict]
    vector.append(label)  # Add the label at the end
    return vector

if __name__ == "__main__":
    """ syscall_dict_path = "system_calls_feature_dict.txt"
    trace_path = "Data/Benign/systemcalls/systemcalls/0a0e1dcdc37a458f111dfaca297abace5404b09827c20f91ca88f2a0c850a64e"  # Rename your file if different

    syscall_dict = load_syscall_dict(syscall_dict_path)
    freq_counter = extract_syscall_frequencies(trace_path, syscall_dict)
    feature_vector = create_feature_vector(freq_counter, syscall_dict)

     # Print or save to CSV
    print("Feature Vector:", feature_vector)

    with open("output_features.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(syscall_dict)         # Header
        writer.writerow(feature_vector)       # Feature values """
    
    syscall_dict_path = "data/other/system_calls_feature_dict.txt"
    output_csv_path = "data/processed/output_features_all.csv"
    benign_data_folder = "data/raw/Benign/systemcalls/systemcalls"
    malware_data_folder = "data/raw/Malware/systemcalls"
    benign_label = "benign"
    malware_label = "malware"

    syscall_dict = load_syscall_dict(syscall_dict_path)
    #freq_counter = extract_syscall_frequencies(trace_path, syscall_dict)
    #feature_vector = create_feature_vector(freq_counter, syscall_dict, label="Benign")
    
    # Add "label" as the last column in the CSV header
    header = syscall_dict + ["label"]

    feature_rows = []
    """ for i, filename in enumerate(os.listdir(benign_data_folder)):
        if i >= 20000: break
        file_path = os.path.join(benign_data_folder, filename)
        if os.path.isfile(file_path):
            freq_counter = extract_syscall_frequencies(file_path, syscall_dict)
            vector = create_feature_vector(freq_counter, syscall_dict, label=benign_label)
            feature_rows.append(vector) """

    for i, filename in enumerate(os.listdir(malware_data_folder)):
        if i >= 18621: break
        file_path = os.path.join(malware_data_folder, filename)
        if os.path.isfile(file_path):
            freq_counter = extract_syscall_frequencies(file_path, syscall_dict)
            vector = create_feature_vector(freq_counter, syscall_dict, label=malware_label)
            feature_rows.append(vector)

    # Write to output CSV
    file_exists = os.path.isfile(output_csv_path)

    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)  # Only write header if file is new
        writer.writerows(feature_rows)  # Append new rows

        print(f"Extracted features for {len(feature_rows)} files → {output_csv_path}")