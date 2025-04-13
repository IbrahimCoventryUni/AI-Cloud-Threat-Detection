import pandas as pd
from collections import defaultdict
import time

# Initialize dictionaries to track SYN packet counts
syn_packet_count = defaultdict(int)
last_reset_time = time.time()

# Function to calculate SYN flood features
def calculate_syn_flood_features(row):
    global syn_packet_count, last_reset_time

    # Ensure we are handling TCP packets (proto = 6)
    if row.get("proto", -1) == 6 and row.get("state", -1) == 1:  # TCP SYN packet
        key = (row.get("sport", 0), row.get("dsport", 0))  # Use source/destination ports as keys

        # Update SYN packet count
        syn_packet_count[key] += 1

        # Reset counts every second
        if time.time() - last_reset_time >= 1:
            syn_packet_count.clear()
            last_reset_time = time.time()

        # Add SYN flood-specific features
        row["syn_packet_rate"] = syn_packet_count[key]
        row["is_syn_flood"] = int(syn_packet_count[key] > 100)  # Threshold for SYN flood
    else:
        row["syn_packet_rate"] = 0
        row["is_syn_flood"] = 0

    return row


# Process the dataset in chunks
chunksize = 100000  # Adjust this based on your system's memory
output_file = "../data/balanced_unsw_updated2.csv"

# Open the output file in write mode and write the header
with open(output_file, "w", newline="") as f:
    header_written = False

    # Read the dataset in chunks
    for chunk in pd.read_csv("../data/balanced_unsw_updated.csv", chunksize=chunksize):
        # Apply the function to each row in the chunk
        chunk = chunk.apply(calculate_syn_flood_features, axis=1)

        # Write the chunk to the output file
        chunk.to_csv(f, mode="a", header=not header_written, index=False)
        header_written = True

print("Dataset processing complete. Updated dataset saved to:", output_file)