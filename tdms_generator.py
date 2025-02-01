#!/usr/bin/env python3
"""
TDMS Test File Generator for IDE usage

This module provides a function `generate_tdms_file()` that creates a TDMS file
with the following features:

- **Input parameters:**  
  Specify the number of groups, channels per group, desired output file size (in MB),
  the entropy mode, and the chunk size (number of rows per chunk).

- **Row Count from File Size:**  
  The number of rows (i.e. data points per channel) is determined by dividing the desired
  file size (in bytes) by the total number of bytes per “row” (i.e. the sum of the size
  of one element per channel).

- **Varied Data Types:**  
  Each channel is assigned a random data type chosen from a list of common types:
  float64, float32, int64, int32, int16, and uint8.

- **Data Entropy Options:**  
  Data can be generated with:
    - "high": fully random values,
    - "low": almost constant values,
    - "mixed-low": each channel is randomly assigned high entropy with a 30% chance (70% low),
    - "mixed": a 50/50 chance per channel,
    - "mixed-high": each channel is randomly assigned high entropy with a 70% chance (30% low).

- **Auto-Generated Names:**  
  Groups are named “G1”, “G2”, etc., and channels are named with an abbreviation for
  their data type and entropy (for example, “C1_f32H” indicates channel 1 of type float32
  with high entropy).

- **Memory-Efficient Chunking:**  
  The file is written in segments (each containing a configurable number of rows)
  to ensure memory usage is acceptable even when generating files up to ~10 GB in size.

Usage Example (from an IDE):
--------------------------------------------------
from tdms_generator import generate_tdms_file

# Generate a 50 MB TDMS file with 2 groups, 4 channels per group,
# using the "mixed-high" entropy mode and writing 100,000 rows per segment.
generate_tdms_file(
    output="test_file.tdms",
    group_count=2,
    channels_per_group=4,
    size_mb=50,
    entropy_mode="mixed-high",
    chunk_size=100000
)
--------------------------------------------------
"""

import math
import numpy as np
from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject

def generate_channel_data(dtype, entropy_mode, size):
    """
    Generate a 1D numpy array of the given length and data type.
    
    - For "high" entropy, the data are fully random.
    - For "low" entropy, the data are nearly constant.
    
    Parameters:
      - dtype: a numpy data type (e.g. np.float32)
      - entropy_mode: "high" or "low"
      - size: number of elements
      
    Returns:
      A numpy array of the specified dtype and length.
    """
    if entropy_mode == "high":
        if np.issubdtype(dtype, np.floating):
            return np.random.random(size).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.random.randint(info.min, info.max, size=size, dtype=dtype)
        else:
            raise ValueError("Unsupported dtype for random generation")
    elif entropy_mode == "low":
        if np.issubdtype(dtype, np.floating):
            constant = 0.5
        elif np.issubdtype(dtype, np.integer):
            constant = 100
        else:
            raise ValueError("Unsupported dtype for low entropy generation")
        return np.full(size, constant, dtype=dtype)
    else:
        raise ValueError("Unknown entropy mode: {}".format(entropy_mode))

def generate_tdms_file(
    output: str,
    group_count: int,
    channels_per_group: int,
    size_mb: float,
    entropy_mode: str = "high",
    chunk_size: int = 100000
):
    """
    Generate a TDMS file with random test data.
    
    Parameters:
      - output: The file path for the TDMS file.
      - group_count: Number of groups in the file.
      - channels_per_group: Number of channels per group.
      - size_mb: Desired file size in megabytes.
      - entropy_mode: Controls data entropy. Options:
            "high"      - all channels are high entropy (fully random),
            "low"       - all channels are low entropy (nearly constant),
            "mixed-low" - each channel is randomly assigned high entropy with 30% probability (70% low),
            "mixed"     - 50/50 chance per channel,
            "mixed-high"- each channel is randomly assigned high entropy with 70% probability (30% low).
      - chunk_size: Number of rows (data points) to write per segment.
      
    This function computes the number of rows per channel needed to approximately meet the
    desired file size, builds the configuration for groups and channels, and writes the file in
    segments to ensure low memory usage.
    """
    # Define available data types as tuples: (abbreviation, numpy dtype)
    available_types = [
        ("f64", np.float64),
        ("f32", np.float32),
        ("i64", np.int64),
        ("i32", np.int32),
        ("i16", np.int16),
        ("u8",  np.uint8)
    ]
    
    # Set probability for high entropy in mixed modes.
    if entropy_mode == "mixed-low":
        high_prob = 0.05
    elif entropy_mode == "mixed":
        high_prob = 0.50
    elif entropy_mode == "mixed-high":
        high_prob = 0.70
    elif entropy_mode in {"high", "low"}:
        high_prob = None  # not used
    else:
        raise ValueError("Invalid entropy_mode value: {}. Must be one of "
                         "'high', 'low', 'mixed-low', 'mixed', or 'mixed-high'".format(entropy_mode))
    
    # Build configuration for groups and channels.
    # Each group is identified by its name, and each channel has a name with a data type abbreviation and entropy suffix.
    groups_config = {}
    for g in range(group_count):
        group_name = f"G{g+1}"
        groups_config[group_name] = {}
        for c in range(channels_per_group):
            abbr, dtype = available_types[np.random.randint(0, len(available_types))]
            if entropy_mode in {"mixed-low", "mixed", "mixed-high"}:
                chosen_entropy = "high" if np.random.random() < high_prob else "low"
            else:
                chosen_entropy = entropy_mode
            ch_suffix = "H" if chosen_entropy == "high" else "L"
            channel_name = f"C{c+1}_{abbr}{ch_suffix}"
            groups_config[group_name][channel_name] = {"dtype": dtype, "entropy": chosen_entropy}
    
    # Calculate the total number of bytes per row (each row consists of one element per channel).
    total_row_bytes = 0
    for channels in groups_config.values():
        for conf in channels.values():
            total_row_bytes += np.dtype(conf["dtype"]).itemsize

    desired_bytes = int(size_mb * 1024 * 1024)
    row_count = desired_bytes // total_row_bytes
    if row_count < 1:
        raise ValueError("Desired file size is too small for the given configuration.")
    
    print(f"Generating TDMS file '{output}' with:")
    print(f"  Groups: {group_count}")
    print(f"  Channels per group: {channels_per_group}")
    print(f"  Total channels: {group_count * channels_per_group}")
    print(f"  Rows per channel: {row_count} (approx. {row_count * total_row_bytes / (1024*1024):.2f} MB)")
    
    num_chunks = math.ceil(row_count / chunk_size)
    print(f"Writing data in {num_chunks} segment(s) (segment size = {chunk_size} rows per channel)...")
    
    with TdmsWriter(output) as writer:
        first_segment = True
        for chunk in range(num_chunks):
            start_row = chunk * chunk_size
            end_row = min((chunk + 1) * chunk_size, row_count)
            current_chunk_size = end_row - start_row
            
            segment_objects = []
            # For the very first segment, include the RootObject and one GroupObject per group.
            if first_segment:
                segment_objects.append(RootObject(properties={}))
            
            for group_name, channels in groups_config.items():
                if first_segment:
                    segment_objects.append(GroupObject(group_name, properties={}))
                for channel_name, conf in channels.items():
                    dtype = conf["dtype"]
                    mode = conf["entropy"]
                    data = generate_channel_data(dtype, mode, current_chunk_size)
                    # Create a new ChannelObject for this data segment.
                    segment_objects.append(ChannelObject(group_name, channel_name, data, properties={}))
            
            writer.write_segment(segment_objects)
            print(f"  Segment {chunk+1}/{num_chunks} written ({current_chunk_size} rows per channel).")
            first_segment = False
            
    print("Done. TDMS file generated.")

if __name__ == '__main__':
    generate_tdms_file(
        output="test_file.tdms",
        group_count=3,
        channels_per_group=200,
        size_mb=100,             # Desired file size in MB
        entropy_mode="mixed-low",  # Options: "high", "low", "mixed-low", "mixed", "mixed-high"
        chunk_size=100000
    )