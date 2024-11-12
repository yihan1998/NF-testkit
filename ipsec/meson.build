project('DOCA_SAMPLE', 'C', 'CPP',
	# Get version number from file.
	version: run_command(find_program('cat'),
		files('/opt/mellanox/doca/applications/VERSION'), check: true).stdout().strip(),
	license: 'Proprietary',
	default_options: ['buildtype=debug'],
	meson_version: '>= 0.61.2'
)

APP_NAME = 'ipsec_security_gw'

app_dependencies = []
app_dependencies += dependency('doca')
app_dependencies += dependency('libdpdk')
app_dependencies += dependency('json-c')

common_dir_path = 'common'

app_inc_dirs = [
	include_directories(common_dir_path),
	include_directories('.'),
]

app_srcs = files([
	'config.c',
	'flow_common.c',
	'flow_decrypt.c',
	'flow_encrypt.c',
	'flow_encrypt.c',
	'ipsec_ctx.c',
	'policy.c',
	APP_NAME + '.c',
	common_dir_path + '/dpdk_utils.c',
	common_dir_path + '/offload_rules.c',
	common_dir_path + '/pack.c',
	common_dir_path + '/utils.c',
	common_dir_path + '/flow_parser.c',
	common_dir_path + '/common.c',
])

executable(APP_NAME,
	app_srcs,
	dependencies : app_dependencies,
	include_directories : app_inc_dirs)
