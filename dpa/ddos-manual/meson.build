project('VXLAN', 'C',
	license: 'Proprietary',
	default_options: ['buildtype=debug'],
	meson_version: '>= 0.61.2'
)

inc_dirs = [
	include_directories('.'),
]

ibverbs_dep = dependency('libibverbs', required: true)
mlx5_dep = dependency('libmlx5', required: true)
thread_dep = dependency('threads', required: true)
flexio_dep = dependency('libflexio', required: true)
dpdk_dep = dependency('libdpdk', required: true)
doca_common_dep = dependency('doca-common', required: true)
doca_flow_dep = dependency('doca-flow', required: true)

add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])

dev_app_name = 'ddos'
shared_include_files = [join_paths(meson.current_source_dir(),'common.h')]
subdir('dev')
subdir('host')
