#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

project('DOCA_SAMPLE', 'C', 'CPP',
	# Get version number from file.
	version: run_command(find_program('cat'),
		files('/opt/mellanox/doca/applications/VERSION'), check: true).stdout().strip(),
	license: 'Proprietary',
	default_options: ['buildtype=debug'],
	meson_version: '>= 0.61.2'
)

SAMPLE_NAME = 'vxlan_fwd'

# Comment this line to restore warnings of experimental DOCA features
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])

sample_dependencies = []
# Required for all DOCA programs
sample_dependencies += dependency('doca')
# 3rd Party dependencies
sample_dependencies += dependency('libdpdk')

sample_srcs = [
	SAMPLE_NAME + '.c',
	SAMPLE_NAME + '_vnf.c',
	SAMPLE_NAME + '_ft.c',
	SAMPLE_NAME + '_pkt.c',
	SAMPLE_NAME + '_port.c',
	SAMPLE_NAME + '_vnf_core.c',
	'common/utils.c',
	'common/dpdk_utils.c',
	'common/offload_rules.c',
]

sample_inc_dirs  = []
sample_inc_dirs += include_directories('common/')

executable('doca_' + SAMPLE_NAME, sample_srcs,
	c_args : '-Wno-missing-braces',
	dependencies : sample_dependencies,
	include_directories: sample_inc_dirs,
	install: false)
