sample_dev_srcs =  ['ddos_dev.c']
app_out_name = dev_app_name + '.a'

build_dpacc = find_program('/opt/mellanox/flexio/samples/doca_build_dpacc.sh', required: true)

dpacc_devicecc_array = ['-DE_MODE_LE',
			'-Wall',
			'-Wextra',
			'-Wpedantic',
			'-Wdouble-promotion',
			'-Wno-empty-translation-unit',
			'-Wmissing-prototypes',
			'-Wstrict-prototypes',
			'-ffreestanding',
			'-mcmodel=medany',]
			
sample_dev_target = custom_target('apps_' + dev_app_name,
				  command: [build_dpacc, '--application',
					    '--app_name', dev_app_name,
					    '--srcs', '@INPUT@',
					    '--dpacc_build_dir', '@OUTDIR@',
					    '--external_cc_options',
					    ','.join(dpacc_devicecc_array),
                                            #'--additional_include_directories',
                                            #common_dev_include_dir,
                                            #'--additional_ld_libs', common_dev_name,
                                            #'--additional_lib_paths', common_dev_lib_path
                                  ],
				  output: app_out_name,
				  input: sample_dev_srcs,
                                  #depends: [common_dev_target],
				  depend_files: shared_include_files)

# Create dependency from DPA application for link with host side sample
sample_device_dep = declare_dependency(link_args :
                                       join_paths(meson.current_build_dir(), app_out_name))
