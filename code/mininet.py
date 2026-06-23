import socket
import time
from datetime import datetime
import sys


MININET_HOST = "143.248.49.149"
MININET_PORT = 6789


SERVER_HOST_NAME = "h_ny_0"


SENDING_HOSTS = [
    "h_lon_0",
    "h_lon_1",
    "h_lon_2",
    "h_lon_3",
    "h_lon_4",
    "h_lon_5",
    "h_lon_6",
    "h_lon_7",
    "h_ams_0",
    "h_ams_1",
    "h_ams_2",
    "h_ams_3",
    "h_ams_4",
    "h_ams_5",
    "h_ams_6",
    "h_ams_7",
    "h_sg_0",
    "h_sg_1",
    "h_sg_2",
    "h_sg_3",
    "h_sg_4",
    "h_sg_5",
    "h_sg_6",
    "h_sg_7",
    "h_tyo_0",
    "h_tyo_1",
    "h_tyo_2",
    "h_tyo_3",
    "h_tyo_4",
    "h_tyo_5",
    "h_tyo_6",
    "h_tyo_7",
    "h_syd_0",
    "h_syd_1",
    "h_syd_2",
    "h_syd_3",
    "h_syd_4",
    "h_syd_5",
    "h_syd_6",
    "h_syd_7",
    "h_sfo_0",
    "h_sfo_1",
    "h_sfo_2",
    "h_sfo_3",
    "h_sfo_4",
    "h_sfo_5",
    "h_sfo_6",
    "h_sfo_7",
    "h_ny_1",
    "h_ny_2",
]

assert len(SENDING_HOSTS) == 50, f"Expected 50 sending hosts, got {len(SENDING_HOSTS)}"


class SimpleNotebook:
    """
    Simple notebook for recording total algorithm runtimes.
    """

    def __init__(self, filename=None):
        """Initialize the notebook."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"algorithm_times_{timestamp}.txt"
        else:
            self.filename = filename

        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("Algorithm execution time log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Log start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

        print(f"Time log created: {self.filename}")

    def write_algorithm_time(self, algorithm_name, total_time_seconds):
        """Record the total runtime for one algorithm."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        total_minutes = total_time_seconds / 60

        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] {algorithm_name}: {total_time_seconds:.2f} seconds ({total_minutes:.2f} minutes)\n"
            )

        print(
            f"Recorded total time for {algorithm_name}: {total_time_seconds:.2f} seconds"
        )

    def read_all_times(self):
        """Read all recorded runtimes."""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                content = f.read()
                print("Current recorded algorithm runtimes:")
                print(content)
                return content
        except FileNotFoundError:
            print("Record file does not exist")
            return ""


def revert_ip(host_str: str) -> str:
    """Convert a host name such as 'h_lon_0' to an IP string such as '20.0.1.1'."""
    reverse_map = {
        "ny": "10",
        "lon": "20",
        "ams": "30",
        "sg": "40",
        "tyo": "50",
        "syd": "60",
        "sfo": "70",
    }
    try:
        _, loc, index_str = host_str.split("_")
        return f"{reverse_map[loc]}.0.1.{int(index_str) + 1}"
    except (ValueError, KeyError) as e:
        print(f"Could not convert host name: {host_str}. Error: {e}", file=sys.stderr)
        return None


def parse_timestamp(line: str):
    """Parse a timestamp from a response line."""
    try:

        time_str = line.split(" ")[0]
        return datetime.strptime(time_str, "%H:%M:%S.%f")
    except (ValueError, IndexError):
        return None


class MininetClient:
    """Simple client for connecting to and interacting with Mininet."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = b""

    def connect(self):
        """Connect to the Mininet server."""
        print(f"Connecting to Mininet server {self.host}:{self.port}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print("Connection successful.")

            time.sleep(0.5)
            self.sock.setblocking(False)
            try:
                while True:
                    self.sock.recv(4096)
            except BlockingIOError:
                pass
            self.sock.setblocking(True)

        except ConnectionRefusedError:
            print(
                "Connection refused. Make sure the Mininet server is running and the address is correct.",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error while connecting: {e}", file=sys.stderr)
            sys.exit(1)

    def send_command(self, command: str):
        """Send a single command to Mininet."""
        full_command = command if command.endswith("\n") else command + "\n"
        self.sock.sendall(full_command.encode("utf-8"))

    def receive_line(self, timeout=None):
        """
        Receive one line of data.
        Uses an internal buffer to handle partial lines read from the TCP stream.
        """
        self.sock.settimeout(timeout)
        try:
            while b"\n" not in self.buffer:
                data = self.sock.recv(4096)
                if not data:
                    return None
                self.buffer += data

            line_end = self.buffer.find(b"\n")
            line = self.buffer[:line_end]
            self.buffer = self.buffer[line_end + 1 :]
            return line.decode("utf-8").strip()
        except socket.timeout:
            return "TIMEOUT"
        except Exception as e:
            print(f"Error while receiving data: {e}", file=sys.stderr)
            return None

    def close(self):
        """Close the socket connection."""
        if self.sock:
            print("\nClosing connection...")
            self.sock.close()
            print("Connection closed.")


def run_simulation_single_connection(
    client, rounds: int, packet_size: int, algorithm_name: str = "", notebook=None
):
    """Run one algorithm configuration on an established connection."""

    experiment_start_time = time.time()
    total_times = []

    for i in range(rounds):
        print(f"\n{'=' * 20} Round {i + 1}/{rounds} started {'=' * 20}")

        first_response_ts = None
        last_response_ts = None
        recv_end_count_phase1 = 0
        recv_end_count_phase2 = 0

        print(
            f"\n--- Phase 1: {len(SENDING_HOSTS)} clients send packets of size {packet_size} ---"
        )
        for host_name in SENDING_HOSTS:
            command = f"{host_name} send {packet_size}"
            client.send_command(command)

        print(
            "\n--- Waiting for the server to receive all packets (50 'recv_end' messages) ---"
        )

        while recv_end_count_phase1 < len(SENDING_HOSTS):
            line = client.receive_line()
            if not line or "mininet>" in line:
                continue

            print(f"Received response: {line}")

            ts = parse_timestamp(line)
            if ts:
                if first_response_ts is None:
                    first_response_ts = ts
                last_response_ts = ts

            if "recv_end" in line:
                recv_end_count_phase1 += 1

        print(
            f"\n--- Phase 1 complete ({recv_end_count_phase1}/{len(SENDING_HOSTS)} 'recv_end' received) ---"
        )

        print(
            f"\n--- Phase 2: server ({SERVER_HOST_NAME}) replies to {len(SENDING_HOSTS)} clients ---"
        )
        for host_name in SENDING_HOSTS:
            ip_address = revert_ip(host_name)
            if ip_address:

                command = f"server send {ip_address} {packet_size}"
                client.send_command(command)

        print(
            "\n--- Phase 2 commands sent; waiting for clients to receive all packets (50 'recv_end' messages) ---"
        )

        while recv_end_count_phase2 < len(SENDING_HOSTS):
            line = client.receive_line(timeout=100000)
            if line == "TIMEOUT":
                print("--- Timeout; treating this round as finished ---")
                break
            if not line or "mininet>" in line:
                continue

            print(f"Received response: {line}")

            ts = parse_timestamp(line)
            if ts:
                last_response_ts = ts

            if "recv_end" in line:
                recv_end_count_phase2 += 1

        print(
            f"\n--- Phase 2 complete ({recv_end_count_phase2}/{len(SENDING_HOSTS)} 'accept_end' received) ---"
        )

        if first_response_ts and last_response_ts:
            duration = last_response_ts - first_response_ts
            print(f"\n----------- Round {i + 1} results -----------")
            print(
                f"  - First response time: {first_response_ts.strftime('%H:%M:%S.%f')}"
            )
            print(f"  - Last response time: {last_response_ts.strftime('%H:%M:%S.%f')}")
            print(f"  - Total time interval: {duration.total_seconds():.6f} seconds")
            print(f"------------------------------------")
            total_times.append(duration.total_seconds())
        else:
            print(f"\nRound {i + 1} did not record a valid timestamp.")

    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time

    if total_times:
        avg_time = sum(total_times) / len(total_times)
        print(f"\n{'=' * 20} Simulation summary {'=' * 20}")
        print(f"All {rounds} rounds completed.")
        print(f"Average time interval: {avg_time:.6f} seconds")
        print(f"Total process time: {total_experiment_time:.6f} seconds")
        print(f"Total process time: {total_experiment_time / 60:.2f} minutes")

        if notebook:
            notebook.write_algorithm_time(algorithm_name, total_experiment_time)

        return {
            "algorithm": algorithm_name,
            "rounds": rounds,
            "packet_size": packet_size,
            "avg_time": avg_time,
            "total_experiment_time": total_experiment_time,
            "individual_times": total_times,
        }
    else:
        return None


def run_simulation(rounds: int, packet_size: int, algorithm_name: str = ""):
    """Run the full packet send/receive simulation for backward compatibility."""
    client = MininetClient(MININET_HOST, MININET_PORT)
    client.connect()

    try:
        return run_simulation_single_connection(
            client, rounds, packet_size, algorithm_name
        )
    finally:
        client.close()


def run_batch_experiments():
    """Run batch experiments with 50 parameter combinations over one connection."""

    notebook = SimpleNotebook()

    experiment_configs = [  # UrbanSound8K, MIND, MNIST, CIFAR10, GTSRB
        ("FedAvg", 27, 5639080),
        ("FedAvg", 76, 13465280),
        ("FedAvg", 13, 798440),
        ("FedAvg", 17, 4819496),
        ("FedAvg", 61, 44763564),
        ("FedProx", 25, 5639080),
        ("FedProx", 76, 13465280),
        ("FedProx", 12, 798440),
        ("FedProx", 15, 4819496),
        ("FedProx", 45, 44763564),
        ("SCAFFOLD", 264, 5639080),
        ("SCAFFOLD", 1, 13465280),
        ("SCAFFOLD", 44, 798440),
        ("SCAFFOLD", 98, 4819496),
        ("SCAFFOLD", 533, 44763564),
        ("pFedMe", 142, 5639080),
        ("pFedMe", 1, 13465280),
        ("pFedMe", 34, 798440),
        ("pFedMe", 45, 4819496),
        ("pFedMe", 99, 44763564),
        ("Ditto", 466, 5639080),
        ("Ditto", 284, 13465280),
        ("Ditto", 22, 798440),
        ("Ditto", 37, 4819496),
        ("Ditto", 55, 44763564),
        ("FedALA", 25, 5639080),
        ("FedALA", 143, 13465280),
        ("FedALA", 11, 798440),
        ("FedALA", 21, 4819496),
        ("FedALA", 65, 44763564),
        ("PAGE", 27, 5639080),
        ("PAGE", 162, 13465280),
        ("PAGE", 10, 798440),
        ("PAGE", 10, 4819496),
        ("PAGE", 16, 44763564),
        ("FedSampling", 26, 5639080),
        ("FedSampling", 206, 13465280),
        ("FedSampling", 11, 798440),
        ("FedSampling", 52, 4819496),
        ("FedSampling", 64, 44763564),
        ("ClusteredSampling", 31, 5639080),
        ("ClusteredSampling", 169, 13465280),
        ("ClusteredSampling", 21, 798440),
        ("ClusteredSampling", 21, 4819496),
        ("ClusteredSampling", 63, 44763564),
        ("Hone", 34, 5639080),
        ("Hone", 151, 13465280),
        ("Hone", 8, 798440),
        ("Hone", 11, 4819496),
        ("Hone", 19, 44763564),
    ]

    all_results = []

    print(
        f"Starting batch experiments: {len(experiment_configs)} parameter combinations"
    )
    print("Using single-connection mode to avoid frequent reconnects")
    print("=" * 80)

    client = MininetClient(MININET_HOST, MININET_PORT)
    client.connect()

    try:
        batch_start_time = time.time()

        for i, (algorithm, rounds, packet_size) in enumerate(experiment_configs, 1):
            print(
                f"\n[{i}/{len(experiment_configs)}] Algorithm: {algorithm}, config: {rounds} rounds, {packet_size} bytes"
            )

            try:
                result = run_simulation_single_connection(
                    client, rounds, packet_size, algorithm, notebook
                )
                if result:
                    all_results.append(result)
                    print(
                        f"[OK] Completed: average time {result['avg_time']:.6f} seconds, total time {result['total_experiment_time']:.2f} seconds"
                    )
                else:
                    print("[FAILED] Experiment failed")
            except Exception as e:
                print(f"[ERROR] Experiment error: {e}")

                continue

        batch_end_time = time.time()
        total_batch_time = batch_end_time - batch_start_time

        print(f"\n{'='*80}")
        print("Batch experiments complete.")
        print(
            f"Successfully completed: {len(all_results)}/{len(experiment_configs)} experiments"
        )
        print(
            f"Total batch time: {total_batch_time:.2f} seconds ({total_batch_time / 60:.2f} minutes)"
        )
        print(f"{'='*80}")

    finally:

        client.close()

    save_results_to_file(all_results)

    print(f"\n{'='*60}")
    print("Algorithm runtime summary:")
    notebook.read_all_times()

    return all_results


def save_results_to_file(results):
    """Save experiment results to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mininet_experiment_results_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Mininet network experiment report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write("=" * 80 + "\n\n")

        algorithms = {}
        for result in results:
            algo = result["algorithm"]
            if algo not in algorithms:
                algorithms[algo] = []
            algorithms[algo].append(result)

        for algorithm, algo_results in algorithms.items():
            f.write(f"Algorithm: {algorithm}\n")
            f.write("-" * 60 + "\n")

            for i, result in enumerate(algo_results, 1):
                f.write(
                    f"  Test {i}: {result['rounds']} rounds, {result['packet_size']} bytes\n"
                )
                f.write(
                    f"    Average time interval: {result['avg_time']:.6f} seconds\n"
                )
                f.write(
                    f"    Total experiment time: {result['total_experiment_time']:.6f} seconds ({result['total_experiment_time'] / 60:.2f} minutes)\n"
                )
                f.write(
                    f"    Per-round times: {[f'{t:.6f}' for t in result['individual_times']]}\n"
                )
                f.write("\n")

            avg_times = [r["avg_time"] for r in algo_results]
            total_times = [r["total_experiment_time"] for r in algo_results]

            f.write(f"  {algorithm} algorithm statistics:\n")
            f.write(
                f"    Average time interval - min: {min(avg_times):.6f} seconds, max: {max(avg_times):.6f} seconds, mean: {sum(avg_times) / len(avg_times):.6f} seconds\n"
            )
            f.write(
                f"    Total experiment time - min: {min(total_times):.2f} seconds, max: {max(total_times):.2f} seconds, mean: {sum(total_times) / len(total_times):.2f} seconds\n"
            )
            f.write("\n" + "=" * 80 + "\n\n")

        f.write("Global statistics\n")
        f.write("-" * 60 + "\n")
        all_avg_times = [r["avg_time"] for r in results]
        all_total_times = [r["total_experiment_time"] for r in results]

        f.write(
            f"All experiment average time intervals: min {min(all_avg_times):.6f} seconds, max {max(all_avg_times):.6f} seconds, mean {sum(all_avg_times) / len(all_avg_times):.6f} seconds\n"
        )
        f.write(
            f"All experiment total times: min {min(all_total_times):.2f} seconds, max {max(all_total_times):.2f} seconds, mean {sum(all_total_times) / len(all_total_times):.2f} seconds\n"
        )
        f.write(
            f"Batch experiment total elapsed time: {sum(all_total_times):.2f} seconds ({sum(all_total_times) / 60:.2f} minutes)\n"
        )

    print(f"\nExperiment results saved to file: {filename}")


if __name__ == "__main__":

    if len(sys.argv) == 3:
        try:
            num_rounds = int(sys.argv[1])
            data_packet_size = int(sys.argv[2])
            if num_rounds <= 0 or data_packet_size <= 0:
                raise ValueError("Rounds and packet size must be positive integers.")

            print("Single-test mode")

            notebook = SimpleNotebook()

            client = MininetClient(MININET_HOST, MININET_PORT)
            client.connect()

            try:
                result = run_simulation_single_connection(
                    client, num_rounds, data_packet_size, "Single test", notebook
                )
                if result:
                    print(
                        "\nSingle test complete; total time has been recorded in the notebook"
                    )
                    notebook.read_all_times()
            finally:
                client.close()

        except ValueError as e:
            print(f"Invalid input parameter: {e}", file=sys.stderr)
            sys.exit(1)

    elif len(sys.argv) == 1:
        print("Batch-test mode: running 50 parameter combinations")
        print("Includes 10 algorithms x 5 configurations = 50 experiments")
        print(
            "Algorithms: FedAvg, FedProx, SCAFFOLD, pFedMe, Ditto, FedALA, PAGE, FedSampling, ClusteredSampling, Hone"
        )
        print(
            "This may take a long time. Make sure the network connection is stable..."
        )

        confirm = input("Continue batch testing? (y/N): ").strip().lower()
        if confirm in ["y", "yes"]:
            run_batch_experiments()
        else:
            print("Batch testing cancelled")

    else:
        print("Usage:")
        print("  Batch-test mode: python mininet.py")
        print("  Single-test mode: python mininet.py <rounds> <packet_size>")
        print("Example: python mininet.py 5 100000")
        sys.exit(1)
