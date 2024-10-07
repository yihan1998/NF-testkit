/*
 * Copyright (c) 2023 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef FLOW_CT_COMMON_H_
#define FLOW_CT_COMMON_H_

#include <doca_dev.h>
#include <doca_argp.h>
#include <doca_flow.h>
#include <doca_flow_ct.h>

#include "flow_common.h"

#define FLOW_CT_COMMON_DEVARGS \
	"dv_flow_en=2,dv_xmeta_en=4,representor=pf[0-1],repr_matching_en=0," \
	"fdb_def_rule_en=0,vport_match=1"

#define DUP_FILTER_CONN_NUM 512
#define MAX_PORTS 4

struct ct_config {
	int n_ports;						     /* Number of ports configured */
	char ct_dev_pci_addr[MAX_PORTS][DOCA_DEVINFO_PCI_ADDR_SIZE]; /* Flow CT DOCA device PCI address */
};

/*
 * Register the command line parameters for the DOCA Flow CT samples
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_ct_register_params(void);

/*
 * Initialize DOCA Flow CT library
 *
 * @flags [in]: Flow CT flags
 * @nb_arm_queues [in]: Number of threads the sample will use
 * @nb_ctrl_queues [in]: Number of control queues
 * @nb_user_actions [in]: Number of CT user actions
 * @flow_log_cb [in]: Flow log callback
 * @nb_ipv4_sessions [in]: Number of IPv4 sessions
 * @nb_ipv6_sessions [in]: Number of IPv6 sessions
 * @dup_filter_sz [in]: Number of connections to cache in duplication filter
 * @o_match_inner [in]: Origin match inner
 * @o_zone_mask [in]: Origin zone mask
 * @o_modify_mask [in]: Origin modify mask
 * @r_match_inner [in]: Reply match inner
 * @r_zone_mask [in]: Reply zone mask
 * @r_modify_mask [in]: Reply modify mask
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t init_doca_flow_ct(uint32_t flags,
			       uint32_t nb_arm_queues,
			       uint32_t nb_ctrl_queues,
			       uint32_t nb_user_actions,
			       doca_flow_ct_flow_log_cb flow_log_cb,
			       uint32_t nb_ipv4_sessions,
			       uint32_t nb_ipv6_sessions,
			       uint32_t dup_filter_sz,
			       bool o_match_inner,
			       struct doca_flow_meta *o_zone_mask,
			       struct doca_flow_meta *o_modify_mask,
			       bool r_match_inner,
			       struct doca_flow_meta *r_zone_mask,
			       struct doca_flow_meta *r_modify_mask);

/*
 * Initialize DPDK environment for DOCA Flow CT
 *
 * @argc [in]: Number of program command line arguments
 * @dpdk_argv [in]: DPDK command line arguments create by argp library
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_ct_dpdk_init(int argc, char **dpdk_argv);

/*
 * Verify if DOCA device is ECPF by checking all supported capabilities
 *
 * @dev_info [in]: DOCA device info
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t flow_ct_capable(struct doca_devinfo *dev_info);

/*
 * Calculates a 6 tuple hash for the givin match
 *
 * @match [in]: Doca flow ct match struct that contains the 6 tuple data
 * @is_ipv6 [in]: Indicates ipv6 header match, otherwise ipv4 header match
 * @is_reply [in]: Indicates reply direction, otherwise origin direction
 * @return: hash value
 */
uint32_t flow_ct_hash_6tuple(const struct doca_flow_ct_match *match, bool is_ipv6, bool is_reply);

/*
 * Initialize DPDK environment for DOCA Flow CT
 *
 * @ct_pipe [in]: ct pipe to destroy
 * @nb_ports [in]: number of doca flow ports to close
 * @ports [in]: doca flow ports to close
 */
void cleanup_procedure(struct doca_flow_pipe *ct_pipe, int nb_ports, struct doca_flow_port *ports[]);

/*
 * Create root pipe with IP filter dst == 1.1.1.1
 *
 * @port [in]: Pipe port
 * @is_ipv4 [in]: allow IPv4 packets
 * @is_ipv6 [in]: allow IPv6 packets
 * @l4_type [in]: L4 type
 * @fwd_pipe [in]: Next pipe pointer
 * @status [in]: User context for adding entry
 * @pipe [out]: Created pipe pointer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise.
 */
doca_error_t create_ct_root_pipe(struct doca_flow_port *port,
				 bool is_ipv4,
				 bool is_ipv6,
				 enum doca_flow_l4_meta l4_type,
				 struct doca_flow_pipe *fwd_pipe,
				 struct entries_status *status,
				 struct doca_flow_pipe **pipe);

#endif /* FLOW_CT_COMMON_H_ */
