# NF-testkit

## Repository Structure

```console
.
├── Makefile
├── README.md
├── compression
│   ├── Makefile
│   ├── compression.c
│   ├── compression.h
│   ├── config.h
│   ├── doca.c
│   ├── doca.h
│   ├── ethernet.c
│   ├── ethernet.h
│   ├── ip.h
│   ├── ip4.c
│   ├── list.h
│   ├── main.c
│   ├── skbuff.c
│   └── skbuff.h
├── dns-filter
│   ├── Makefile
│   ├── dns_filter_rules.txt
│   └── main.c
├── file-verification
│   ├── Makefile
│   ├── config.h
│   ├── doca.c
│   ├── doca.h
│   ├── ethernet.c
│   ├── ethernet.h
│   ├── file_verification.c
│   ├── file_verification.h
│   ├── ip.h
│   ├── ip4.c
│   ├── list.h
│   ├── main.c
│   ├── skbuff.c
│   └── skbuff.h
├── nat-hw
│   ├── Makefile
│   ├── config.h
│   ├── dns_filter.c
│   ├── dns_filter.h
│   ├── dns_filter_rules_5.txt
│   ├── doca.c
│   ├── doca.h
│   ├── main.c
│   └── nat_hw
├── raw_echo
│   ├── Makefile
│   ├── config.h
│   ├── raw_echo
│   └── raw_echo.c
├── sha1-auth
│   ├── Makefile
│   ├── config.h
│   ├── doca.c
│   ├── doca.h
│   ├── ethernet.c
│   ├── ethernet.h
│   ├── ip.h
│   ├── ip4.c
│   ├── list.h
│   ├── main.c
│   ├── sha1_auth.c
│   ├── sha1_auth.h
│   ├── skbuff.c
│   └── skbuff.h
└── vxlan_fwd
    ├── app_vnf.h
    ├── common
    │   ├── dpdk_utils.c
    │   ├── dpdk_utils.h
    │   ├── offload_rules.c
    │   ├── offload_rules.h
    │   ├── utils.c
    │   └── utils.h
    ├── meson.build
    ├── run.sh
    ├── vxlan_fwd.c
    ├── vxlan_fwd.h
    ├── vxlan_fwd_ft.c
    ├── vxlan_fwd_ft.h
    ├── vxlan_fwd_pkt.c
    ├── vxlan_fwd_pkt.h
    ├── vxlan_fwd_port.c
    ├── vxlan_fwd_port.h
    ├── vxlan_fwd_vnf.c
    ├── vxlan_fwd_vnf_core.c
    └── vxlan_fwd_vnf_core.h
```

## Work Status of Subdirectories

Here is the current work status of the subdirectories within the NF-testkit project:

* `/compression`: `Complete`

    Deflate data via zlib or DOCA. Work with DOCA 2.2.

* `/file-verification`: `Complete`

    SHA-256 via OpenSSL or DOCA. Work with DOCA 2.2.

* `/raw-echo`: `Complete`

    Simple echo program based on DPDK.

* `/sha1-auth`: `Complete`

    SHA-1 via OpenSSL or DOCA. Work with DOCA 2.2.

* `/vxlan_fwd`: `Complete`

    VXLAN encapsulation via C or DOCA Flow. Work with DOCA 2.2.

## Usage

```console
sudo <app> -l <list-of-cores> -n <memory-channel-num> -a <pcie-addr> -- -c <number-of-cores>
```

### Example

To run `compression` application on 4 cores:

```console
sudo ./compression -l 0-3 -n 4 -a 03:00.1 -- -c 4
```
The redundant `-c` shall be fixed later.