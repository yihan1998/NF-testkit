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

dev_app_name = 'vxlan'
subdir('dev')
subdir('host')
