import numpy as np 
import csv 
import glob 
import os
import re
from collections import defaultdict
import torch
import h5py
import argparse
from collections import Counter


def extract_keys(input_files, annotation_files):
    # Define patterns for the two naming conventions
    pattern_activity = r"input_([a-z]+_\d+_\d+_\d+)"  # Matches input_ACTIVITY_XXXX_XXXX_XX
    pattern_name = r"input_\d+_([a-z]+_[a-z]+_\d+)"    # Matches input_XXXX_NAME_ACTIVITY_XX

    # Dictionary to hold associations
    file_associations = defaultdict(lambda: {"input": None, "annotation": None})

    # Process input files
    for file in input_files:
        match_activity = re.match(pattern_activity, file)
        match_name = re.match(pattern_name, file)
        if match_activity:
            key = match_activity.group(1)
            file_associations[key]["input"] = file
        elif match_name:
            key = match_name.group(1)
            file_associations[key]["input"] = file

    # Process annotation files
    for file in annotation_files:
        # Extract keys based on the annotation naming conventions
        match_activity = re.match(r"annotation_([a-z]+_\d+_\d+_\d+)", file)
        match_name = re.match(r"annotation_([a-z]+_[a-z]+_\d+)", file)
        if match_activity:
            key = match_activity.group(1)
            file_associations[key]["annotation"] = file
        elif match_name:
            key = match_name.group(1)
            file_associations[key]["annotation"] = file

    # Convert defaultdict to regular dictionary for final output
    return dict(file_associations)


def create_dataset(input_path, annotation_path):
    input_files = os.listdir(input_path)
    annotation_files = os.listdir(annotation_path) 
    file_associations = extract_keys(input_files, annotation_files) 

    label_encoder = {"NoActivity": 0, "bed": 1, "fall": 2, "pickup": 3, "run": 4, "sitdown": 5, "standup": 6, "walk": 7}
    #create np array that will store the attributes in the format [capture, rows, columns]
    #only store the rows that have an associated annotation different from 'NoActivity'
    #ignore 1st column from input file since it contains timestamps
    #create vector that will store the labels
    data = []
    labels = []

    for capture, files in file_associations.items():
        input_file = files["input"]
        annotation_file = files["annotation"]

        if not input_file or not annotation_file:
            # Skip if either input or annotation file is missing
            continue

        # Load input and annotation files
        input_data = np.loadtxt(os.path.join(input_path, input_file), delimiter=',')
        annotation_data = np.loadtxt(os.path.join(annotation_path, annotation_file), delimiter=',', dtype=str)

        # Check if input and annotation file sizes match
        if input_data.shape[0] != annotation_data.shape[0]:
            print(f"Skipping {capture}: Mismatch in row counts between input and annotation files.")
            continue

        # Ignore the first column (timestamps) in input file
        input_data = input_data[:, 1:]

        # Downsample to 100Hz using a moving average sliding window for data
        # and majority voting for annotations
        window_size = 10  # Corresponds to 1000Hz -> 100Hz
        input_data_downsampled = []
        annotation_data_downsampled = []

        for i in range(0, input_data.shape[0], window_size):
            # Downsample data by averaging
            window_data = input_data[i:i + window_size]
            if len(window_data) < window_size:
                break  # Skip incomplete windows
            input_data_downsampled.append(np.mean(window_data, axis=0))
            
            # Select the annotation by majority voting within the window
            window_labels = annotation_data[i:i + window_size]
            label_counts = Counter(window_labels)
            most_common_label, _ = label_counts.most_common(1)[0]
            annotation_data_downsampled.append(most_common_label)

        input_data_downsampled = np.array(input_data_downsampled)
        annotation_data_downsampled = np.array(annotation_data_downsampled)

        #Free unused memory
        del input_data, annotation_data

        # Group data into frames with 10 time slices
        frame_size = 10  # 10 time slices
        num_frames = input_data_downsampled.shape[0] // frame_size


        for i in range(num_frames):
            frame_start = i*frame_size
            frame_end = frame_start + frame_size

            #Extract frame data and annotations
            frame_data = input_data_downsampled[frame_start:frame_end]
            frame_labels = annotation_data_downsampled[frame_start:frame_end]

            #Determine the frame label (majority vote)
            frame_label = label_encoder[
                max(set(frame_labels), key=frame_labels.tolist().count)
            ]

            data.append(torch.tensor(frame_data, dtype=torch.float32))
            labels.append(torch.tensor(frame_label, dtype=torch.uint8))
        
        # Free memory after processing the capture
        del input_data_downsampled, annotation_data_downsampled

    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return data, labels


def pad_to_3d(data, labels):
    max_rows = max(tensor.size(0) for tensor in data)  # Determine the maximum number of rows

    for i in range(len(data)):
        tensor = data[i]
        label = labels[i]
        padding_rows = max_rows - tensor.size(0)  # Calculate the number of rows to pad

        if padding_rows > 0:
            # Repeat the last row and append it to the tensor
            last_row = tensor[-1].unsqueeze(0)  # Extract the last row and add a batch dimension
            repeated_rows = last_row.repeat(padding_rows, *[1] * (tensor.dim() - 1))  # Repeat for padding_rows times
            data[i] = torch.cat((tensor, repeated_rows), dim=0)  # Concatenate along rows

            # Append the last label value padding_rows times
            last_label = label[-1] 
            labels[i] = torch.cat((label, last_label.repeat(padding_rows)))

    return torch.stack(data), torch.stack(labels)  # Stack data into a 3D tensor and return modified labels


def cvt_to_csi_ratio(data):
    proc_data = torch.zeros_like(data)
    for idx, tensor in enumerate(data):
        num_antennas = 3
        num_subcarriers = 30
        A_n = np.roll(tensor[:,:num_antennas*num_subcarriers], shift=num_subcarriers, axis=1)
        A_n[:,:num_subcarriers] = tensor[:,:num_subcarriers]
        A_d = np.roll(tensor[:,:num_antennas*num_subcarriers], shift=-num_subcarriers, axis=1)
        A_d[:,2*num_subcarriers:3*num_subcarriers] = tensor[:,2*num_subcarriers:3*num_subcarriers]

        theta_n = np.roll(tensor[:,num_antennas*num_subcarriers:], shift=num_subcarriers, axis=1)
        theta_n[:,:num_subcarriers] = tensor[:,num_antennas*num_subcarriers:(num_antennas+1)*num_subcarriers]
        theta_d = np.roll(tensor[:,num_antennas*num_subcarriers:], shift=-num_subcarriers, axis=1)
        theta_d[:,2*num_subcarriers:3*num_subcarriers] = tensor[:,(num_antennas+2)*num_subcarriers:(num_antennas+3)*num_subcarriers]

        proc_data[idx, :, :num_antennas*num_subcarriers] = torch.tensor(np.divide(A_n, A_d, where=A_d!=0))
        proc_data[idx, :, num_antennas*num_subcarriers:] = torch.tensor(theta_n - theta_d)

    return proc_data



def main(args):
    # Use the arguments passed to configure paths
    input_path = args.input_path
    annotation_path = args.annotation_path

    data, labels = create_dataset(input_path, annotation_path)

    #padded_data, padded_labels = pad_to_3d(data, labels)

    csi_ratio_padded_data = cvt_to_csi_ratio(data)
   
    # Save the results to the specified output files
    with h5py.File(args.output_data_path, "w") as f:
        f.create_dataset("X", data=data)
    with h5py.File(args.output_labels_path, "w") as f:
        f.create_dataset("y", data=labels)

    with h5py.File(args.output_csi_ratio_data_path, "w") as f:
        f.create_dataset("X", data=csi_ratio_padded_data)
    with h5py.File(args.output_csi_ratio_labels_path, "w") as f:
        f.create_dataset("y", data=labels)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and save HAR dataset.")
    
    # Add arguments for input and output paths
    parser.add_argument("--input_path", type=str, default="../data/UT_HAR_OG/input", 
                        help="Path to the input data directory.")
    parser.add_argument("--annotation_path", type=str, default="../data/UT_HAR_OG/annotation", 
                        help="Path to the annotation data directory.")
    parser.add_argument("--output_data_path", type=str, default="../data/UT_HAR_OG/X.h5", 
                        help="Path to save the padded data (default: ../data/UT_HAR_OG/X.h5).")
    parser.add_argument("--output_labels_path", type=str, default="../data/UT_HAR_OG/y.h5", 
                        help="Path to save the padded labels (default: ../data/UT_HAR_OG/y.h5).")
    parser.add_argument("--output_csi_ratio_data_path", type=str, default="../data/UT_HAR_CSI_RATIO/X.h5", 
                        help="Path to save the CSI ratio data (default: ../data/UT_HAR_CSI_RATIO/X.h5).")
    parser.add_argument("--output_csi_ratio_labels_path", type=str, default="../data/UT_HAR_CSI_RATIO/y.h5", 
                        help="Path to save the CSI ratio labels (default: ../data/UT_HAR_CSI_RATIO/y.h5).")

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(args)
