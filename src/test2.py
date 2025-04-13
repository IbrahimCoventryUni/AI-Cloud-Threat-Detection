from scapy.all import sniff, conf, get_if_list

# Force Scapy to use Npcap (if on Windows)
conf.use_npcap = True

# Print available interfaces
print("Available interfaces:", get_if_list())

# Use the correct interface for sniffing
iface = "vEthernet (WSL (Hyper-V firewall))"  # Replace with the correct interface name
print(f"Sniffing on interface: {iface}")

# Define a callback function to process captured packets
def packet_callback(packet):
    print(f"Captured packet: {packet.summary()}")

# Start packet sniffing
print("Starting packet capture...")
sniff(iface=iface, prn=packet_callback, count=10)
print("Packet capture complete.")