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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <doca_mmap.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dma.h>
#include <doca_types.h>
#include <doca_log.h>
#include <doca_pe.h>

#include <samples/common.h>
#include "pe_common.h"

DOCA_LOG_REGISTER(PE_POLLING::SAMPLE);

/**
 * This sample demonstrates how to use DOCA PE (progress engine) in polling mode.
 * This is the most basic PE sample. Diff between this sample and other sample to see their unique features.
 * The sample uses DOCA_DMA context as an example (DOCA PE can run any library that abides to the PE context API).
 * The sample runs 16 DMA memcpy tasks.
 */

/**
 * This macro is used to minimize code size.
 * The macro runs an expression and returns error if the expression status is not DOCA_SUCCESS
 */
#define EXIT_ON_FAILURE(_expression_) \
	{ \
		doca_error_t _status_ = _expression_; \
\
		if (_status_ != DOCA_SUCCESS) { \
			DOCA_LOG_ERR("%s failed with status %s", __func__, doca_error_get_descr(_status_)); \
			return _status_; \
		} \
	}

#define NUM_TASKS (16)
#define DMA_BUFFER_SIZE (1024)
#define BUFFER_SIZE (DMA_BUFFER_SIZE * 2 * NUM_TASKS)
#define BUF_INVENTORY_SIZE (NUM_TASKS * 2)

/**
 * This struct defines the program context.
 */
struct pe_polling_sample_state {
	struct pe_sample_state_base base;

	struct doca_dma *dma;
	struct doca_ctx *dma_ctx;
	struct doca_dma_task_memcpy *tasks[NUM_TASKS];
};

/*
 * DMA Memcpy task completed callback
 *
 * @dma_task [in]: Completed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void dma_memcpy_completed_callback(struct doca_dma_task_memcpy *dma_task,
					  union doca_data task_user_data,
					  union doca_data ctx_user_data)
{
	uint8_t expected_value = (uint8_t)task_user_data.u64;
	struct pe_polling_sample_state *state = (struct pe_polling_sample_state *)ctx_user_data.ptr;

	state->base.num_completed_tasks++;

	/**
	 * process_completed_dma_memcpy_task returns doca_error_t to be able to use EXIT_ON_FAILURE, but there is
	 * nothing to do with the return value.
	 */
	(void)process_completed_dma_memcpy_task(dma_task, expected_value);

	/* The task is no longer required, therefore it can be freed */
	(void)dma_task_free(dma_task);
}

/*
 * Memcpy task error callback
 *
 * @dma_task [in]: failed task
 * @task_user_data [in]: doca_data from the task
 * @ctx_user_data [in]: doca_data from the context
 */
static void dma_memcpy_error_callback(struct doca_dma_task_memcpy *dma_task,
				      union doca_data task_user_data,
				      union doca_data ctx_user_data)
{
	struct pe_polling_sample_state *state = (struct pe_polling_sample_state *)ctx_user_data.ptr;
	struct doca_task *task = doca_dma_task_memcpy_as_task(dma_task);

	(void)task_user_data;

	/* This sample defines that a task is completed even if it is completed with error */
	state->base.num_completed_tasks++;

	DOCA_LOG_ERR("Task failed with status %s", doca_error_get_descr(doca_task_get_status(task)));

	/* The task is no longer required, therefore it can be freed */
	(void)dma_task_free(dma_task);
}

/**
 * Create DMA
 *
 * @state [in]: sample state
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_dma(struct pe_polling_sample_state *state)
{
	union doca_data ctx_user_data = {0};

	DOCA_LOG_INFO("Creating DMA");

	EXIT_ON_FAILURE(doca_dma_create(state->base.device, &state->dma));
	state->dma_ctx = doca_dma_as_ctx(state->dma);

	/* A context can only be connected to one PE (PE can run multiple contexts) */
	EXIT_ON_FAILURE(doca_pe_connect_ctx(state->base.pe, state->dma_ctx));

	/**
	 * The ctx user data is received in the task completion callback.
	 * Setting the state to the user data binds the program to the callback.
	 * See dma_memcpy_completed_callback for usage.
	 */
	ctx_user_data.ptr = state;
	EXIT_ON_FAILURE(doca_ctx_set_user_data(state->dma_ctx, ctx_user_data));

	EXIT_ON_FAILURE(doca_dma_task_memcpy_set_conf(state->dma,
						      dma_memcpy_completed_callback,
						      dma_memcpy_error_callback,
						      NUM_TASKS));

	return DOCA_SUCCESS;
}

/**
 * This method cleans up the sample resources in reverse order of their creation.
 * This method does not check for destroy return values for simplify.
 * Real code should check the return value and act accordingly (e.g. if doca_ctx_stop failed with DOCA_ERROR_IN_PROGRESS
 * it means that some contexts are still added or even that there are still in flight tasks in the progress engine).
 *
 * @state [in]: sample state
 */
void cleanup(struct pe_polling_sample_state *state)
{
	/* A context must be stopped before it is destroyed */
	if (state->dma_ctx != NULL)
		(void)doca_ctx_stop(state->dma_ctx);

	/* All contexts must be destroyed before PE is destroyed. Context destroy disconnects it from the PE */
	if (state->dma != NULL)
		(void)doca_dma_destroy(state->dma);

	pe_sample_base_cleanup(&state->base);
}

/**
 * Run the sample
 * The method (and the method it calls) does not cleanup anything in case of failures.
 * It assumes that cleanup is called after it at any case.
 *
 * @state [in]: sample state
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t run(struct pe_polling_sample_state *state)
{
	memset(state, 0, sizeof(*state));

	state->base.buffer_size = BUFFER_SIZE;
	state->base.buf_inventory_size = BUF_INVENTORY_SIZE;

	EXIT_ON_FAILURE(allocate_buffer(&state->base));
	EXIT_ON_FAILURE(open_device(&state->base));
	EXIT_ON_FAILURE(create_mmap(&state->base));
	EXIT_ON_FAILURE(create_buf_inventory(&state->base));
	EXIT_ON_FAILURE(create_pe(&state->base));
	EXIT_ON_FAILURE(create_dma(state));
	EXIT_ON_FAILURE(doca_ctx_start(state->dma_ctx));
	EXIT_ON_FAILURE(allocate_dma_tasks(&state->base, state->dma, NUM_TASKS, DMA_BUFFER_SIZE, state->tasks));
	EXIT_ON_FAILURE(submit_dma_tasks(NUM_TASKS, state->tasks));
	EXIT_ON_FAILURE(poll_for_completion(&state->base, NUM_TASKS));

	return DOCA_SUCCESS;
}

/**
 * Run the PE polling sample
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t run_pe_polling_sample(void)
{
	struct pe_polling_sample_state state;
	doca_error_t status = run(&state);

	cleanup(&state);

	return status;
}
