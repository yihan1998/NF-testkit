/*
 * Copyright (c) 2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#include <doca_dpa_dev.h>
#include <doca_dpa_dev_rdma.h>
#include <doca_dpa_dev_buf.h>
#include "../common/dpa_initiator_target_common_defs.h"

/**
 * @brief Number of expected receive completions
 */
#define EXPECTED_NUM_RECEIVES (4)

/**
 * @brief Array used to mark received data when completion is received.
 *
 * Expected received data are 1, 2, 3, 4.
 * When completion is received with data `i`, then received_values[i - 1] = 1.
 * At the end, this array should be [1, 1, 1, 1]
 */
static uint32_t received_values[EXPECTED_NUM_RECEIVES] = {0};

/**
 * @brief Number of current receive completions
 */
static uint32_t num_receive = 0;

/**
 * @brief Kernel function for DPA thread #1
 *
 * This kernel is triggered when a completion is received on attached RDMA context.
 * This kernel is triggered four times, on each, it receives data on DPA buffer with values 1, 2, 3 and 4.
 * On each completion, it gets and dumps completion info.
 * When number of received completions is 4, it verifies all data values (1, 2, 3, 4) were received.
 * if checker passed, DPA Thread #1 notifies DPA Thread #2 and finish, otherwise it errors and finish.
 *
 * @arg [in]: Kernel argument
 */
__dpa_global__ void thread1_kernel(uint64_t arg)
{
	DOCA_DPA_DEV_LOG_INFO("%s: Hello from Thread #1\n", __func__);

	struct dpa_thread1_arg *thread_arg = (struct dpa_thread1_arg *)arg;
	doca_dpa_dev_completion_element_t comp_element;
	int found = 0;
	uint64_t received_val = 0;
	uint32_t user_data;
	doca_dpa_dev_rdma_t rdma;
	doca_dpa_dev_mmap_t mmap_handle;
	uint64_t addr;

	DOCA_DPA_DEV_LOG_INFO("%s: Read completion info\n", __func__);
	found = doca_dpa_dev_get_completion(thread_arg->dpa_comp_handle, &comp_element);
	DOCA_DPA_DEV_LOG_INFO("%s: doca_dpa_dev_get_completion() returned %d\n", __func__, found);

	num_receive++;

	DOCA_DPA_DEV_LOG_INFO("%s: doca_dpa_dev_get_completion_type() returned %d\n",
			      __func__,
			      doca_dpa_dev_get_completion_type(comp_element));
	DOCA_DPA_DEV_LOG_INFO("%s: doca_dpa_dev_get_completion_immediate() returned %u\n",
			      __func__,
			      doca_dpa_dev_get_completion_immediate(comp_element));

	user_data = doca_dpa_dev_get_completion_user_data(comp_element);
	DOCA_DPA_DEV_LOG_INFO("%s: doca_dpa_dev_get_completion_user_data() returned %u\n", __func__, user_data);

	rdma = (user_data == TARGET_RDMA1_USER_DATA) ? thread_arg->target_rdma1_handle :
						       thread_arg->target_rdma2_handle;
	mmap_handle = (user_data == TARGET_RDMA1_USER_DATA) ? thread_arg->dpa_mmap1_handle :
							      thread_arg->dpa_mmap2_handle;
	addr = (user_data == TARGET_RDMA1_USER_DATA) ? thread_arg->local_buf1_addr : thread_arg->local_buf2_addr;

	received_val = *((uint64_t *)(addr));
	DOCA_DPA_DEV_LOG_INFO("%s: Value of DPA received buffer is %lu\n", __func__, received_val);
	received_values[received_val - 1] = 1;

	if (num_receive == EXPECTED_NUM_RECEIVES) {
		for (uint32_t i = 0; i < EXPECTED_NUM_RECEIVES; i++) {
			if (received_values[i] != 1) {
				DOCA_DPA_DEV_LOG_ERR("%s: Thread #1 didn't receive data with value %u\n",
						     __func__,
						     (i + 1));
				DOCA_DPA_DEV_LOG_INFO("%s: Finish\n", __func__);
				doca_dpa_dev_thread_finish();
			}
		}

		DOCA_DPA_DEV_LOG_INFO("%s: Received all data values, Notify Thread #2\n", __func__);
		doca_dpa_dev_thread_notify(thread_arg->notification_comp_handle);

		DOCA_DPA_DEV_LOG_INFO("%s: Finish\n", __func__);
		doca_dpa_dev_thread_finish();
	}

	DOCA_DPA_DEV_LOG_INFO("%s: Repost RDMA receive on DPA Mmap handle 0x%x, address 0x%lx, length %lu\n",
			      __func__,
			      mmap_handle,
			      addr,
			      thread_arg->length);
	doca_dpa_dev_rdma_post_receive(rdma, mmap_handle, addr, thread_arg->length);

	DOCA_DPA_DEV_LOG_INFO("%s: Acknowledge and request another notification\n", __func__);
	doca_dpa_dev_completion_ack(thread_arg->dpa_comp_handle, 1);
	doca_dpa_dev_completion_request_notification(thread_arg->dpa_comp_handle);

	DOCA_DPA_DEV_LOG_INFO("%s: Reschedule\n", __func__);
	doca_dpa_dev_thread_reschedule();
}

/**
 * @brief Kernel function for DPA thread #2
 *
 * This kernel is triggered when DPA thread #1 notifies DPA thread #2.
 * This kernel sets host sync event to let host application start destroying all resources and ending the application
 *
 * @arg [in]: Kernel argument
 */
__dpa_global__ void thread2_kernel(uint64_t arg)
{
	DOCA_DPA_DEV_LOG_INFO("%s: Hello from Thread #2\n", __func__);

	struct dpa_thread2_arg *thread2_arg = (struct dpa_thread2_arg *)arg;

	DOCA_DPA_DEV_LOG_INFO("%s: Set sync event value to %lu\n", __func__, thread2_arg->completion_count);
	doca_dpa_dev_sync_event_update_set(thread2_arg->sync_event_handle, thread2_arg->completion_count);

	DOCA_DPA_DEV_LOG_INFO("%s: Finish\n", __func__);
	doca_dpa_dev_thread_finish();
}

/**
 * @brief RPC function to post first RDMA receive operations on target RDMAs
 *
 * This RPC is used by target host application to post the first RDMA receive operations on DPA local buffers.
 * These buffers will be reused again after each completion
 *
 * @rdma1 [in]: Target RDMA #1 DPA handle
 * @local_buf1_addr [in]: address of received buffer used for Target RDMA #1
 * @dpa_mmap1_handle [in]: DOCA Mmap handle for local_buf1_addr
 * @rdma2 [in]: Target RDMA #2 DPA handle
 * @local_buf2_addr [in]: address of received buffer used for Target RDMA #2
 * @dpa_mmap2_handle [in]: DOCA Mmap handle for local_buf2_addr
 * @length [in]: length of received buffer
 * @return: RPC function always succeed and returns 0
 */
__dpa_rpc__ uint64_t rdma_post_receive_rpc(doca_dpa_dev_rdma_t rdma1,
					   doca_dpa_dev_uintptr_t local_buf1_addr,
					   doca_dpa_dev_mmap_t dpa_mmap1_handle,
					   doca_dpa_dev_rdma_t rdma2,
					   doca_dpa_dev_uintptr_t local_buf2_addr,
					   doca_dpa_dev_mmap_t dpa_mmap2_handle,
					   size_t length)
{
	DOCA_DPA_DEV_LOG_INFO("%s: Target RDMA #1 post receive on DPA Mmap handle 0x%x, address 0x%lx, length %lu\n",
			      __func__,
			      dpa_mmap1_handle,
			      local_buf1_addr,
			      length);
	doca_dpa_dev_rdma_post_receive(rdma1, dpa_mmap1_handle, local_buf1_addr, length);

	DOCA_DPA_DEV_LOG_INFO("%s: Target RDMA #2 post receive on DPA Mmap handle 0x%x, address 0x%lx, length %lu\n",
			      __func__,
			      dpa_mmap2_handle,
			      local_buf2_addr,
			      length);
	doca_dpa_dev_rdma_post_receive(rdma2, dpa_mmap2_handle, local_buf2_addr, length);
	return 0;
}

/**
 * @brief RPC function to post RDMA send with immediate operation
 *
 * This RPC is used by initiator host application to post RDMA send with immediate operation on host local buffer
 *
 * @rdma [in]: RDMA DPA handle
 * @local_buf_addr [in]: address of send buffer
 * @dpa_mmap_handle [in]: send DOCA Mmap handle
 * @length [in]: length of send buffer
 * @immediate [in]: immediate data
 * @return: RPC function always succeed and returns 0
 */
__dpa_rpc__ uint64_t rdma_post_send_imm_rpc(doca_dpa_dev_rdma_t rdma,
					    uintptr_t local_buf_addr,
					    doca_dpa_dev_mmap_t dpa_mmap_handle,
					    size_t length,
					    uint32_t immediate)
{
	DOCA_DPA_DEV_LOG_INFO(
		"%s: Post RDMA send with immediate %u on DPA Mmap handle 0x%x, address 0x%lx, length %lu\n",
		__func__,
		immediate,
		dpa_mmap_handle,
		local_buf_addr,
		length);
	doca_dpa_dev_rdma_post_send_imm(rdma, dpa_mmap_handle, local_buf_addr, length, immediate, 0);

	return 0;
}
