project('VXLAN', 'C',
	license: 'Proprietary',
	default_options: ['buildtype=debug'],
	meson_version: '>= 0.61.2'
)

inc_dirs = [
	include_directories('.'),
]

libflexio_host = dependency('libflexio')

add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])

SAMPLE_NAME='vxlan'

flexio_build_device = '/opt/mellanox/doca/tools/build_flexio_device.sh'

device_builder = find_program(flexio_build_device, required: true)

source_file = meson.current_source_dir() + '/device/vxlan_dev.c'

# Build device code ./build_flexio_device.sh <app name> <source file> <output dir>
dpa_code = custom_target('VXLAN_DEV',
		command : [device_builder,
			SAMPLE_NAME + '_dev',
			source_file,
			meson.current_build_dir()],
		output : SAMPLE_NAME + '_dev.a')

dpa_dependency = static_library(SAMPLE_NAME + '_dev', link_with : dpa_code)

subdir('host')
