import os
import subprocess


def get_tcp_interface_name(network_interface_type="ethernet"):
    """
    Return the name of the ethernet interface which is up
    """
    network_interfaces = os.listdir("/sys/class/net")

    process = subprocess.Popen("ip link show up".split(), stdout=subprocess.PIPE)
    out, err = process.communicate()

    prefix_list_map = {
        "ethernet": ("ens", "eth", "enp"),
        "infiniband": ("ib"),
    }

    for network_interface in network_interfaces:
        prefix_list = prefix_list_map[network_interface_type]
        if network_interface.startswith(
            prefix_list
        ) and network_interface in out.decode("utf-8"):
            print("Using network interface {}".format(network_interface))
            return network_interface
    print("List of network interfaces found:", network_interfaces)
    print("Prefix list being used to search:", prefix_list)
    raise Exception("No proper ethernet interface found")
