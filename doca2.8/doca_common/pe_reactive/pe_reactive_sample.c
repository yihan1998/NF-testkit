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

#include <stdbool.h>
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

DOCA_LOG_REGISTER(PE_REACTIVE::SAMPLE);

/**
 * This sample demonstrates how to use DOCA PE (progress engine) in a reactive pattern.
 * The main loop does nothing but calling doca_pe_progress and the program is maintained in the callbacks.
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
struct pe_reactive_sample_state {
	struct pe_sample_state_base base;	       /* Base "state" */
	struct doca_dma *dma;			       /* DOCA DMA Context */
	struct doca_ctx *dma_ctx;		       /* DOCA Context */
	struct doca_dma_task_memcpy *tasks[NUM_TASKS]; /* Array of DMA memcpy tasks */
	bool run_pe_progress;			       /* Should we keep on progressing the PE? */
};

/**
 * Callback that reacts to DMA state changes
 *
 * @user_data [in]: user data associated with the DMA context. Will hold struct pe_reactive_sample_state
 * @ctx [in]: the DMA context that had a state change
 * @prev_state [in]: previous context state
 * @next_state [in]: next context state (context is already in this state when the callback is called)
 */
static void dma_state_changed_callback(const union doca_data user_data,
				       struct doca_ctx *ctx,
				       enum doca_ctx_states prev_state,
				       enum doca_ctx_states next_state)
{
	(void)ctx;
	(void)prev_state;

	struct pe_reactive_sample_state *state = (struct pe_reactive_sample_state *)user_data.ptr;
	doca_error_t status;

	switch (next_state) {
	case DOCA_CTX_STATE_IDLE:
		DOCA_LOG_INFO("DMA context has been stopped. Destroying context");
		/* The context has been stopped we can destroy it now */
		(void)doca_dma_destroy(state->dma);
		state->dma = NULL;
		state->dma_ctx = NULL;
		/* We can stop progressing the PE as well */
		state->run_pe_progress = false;
		break;
	case DOCA_CTX_STATE_STARTING:
		/**
		 * The context is in starting state, this is unexpected for DMA.
		 */
		DOCA_LOG_ERR("DMA context entered into starting state. Unexpected transition");
		break;
	case DOCA_CTX_STATE_RUNNING:
		DOCA_LOG_INFO("DMA context is running. Submitting tasks");
		/* The context is running, we can allocate and submit tasks now */
		status = allocate_dma_tasks(&state->base, state->dma, NUM_TASKS, DMA_BUFFER_SIZE, state->tasks);
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to allocate DMA memory copy tasks");
			state->run_pe_progress = false;
		}
		status = submit_dma_tasks(NUM_TASKS, state->tasks);
		if (status != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to submit DMA memory copy tasks");
			state->run_pe_progress = false;
		}
		break;
	case DOCA_CTX_STATE_STOPPING:
		/**
		 * doca_ctx_stop() has been called.
		 * In this sample, this happens either due to a failure encountered, in which case doca_pe_progress()
		 * will cause any inflight task to be flushed, or due to the successful compilation of the sample flow.
		 * In both cases, in this sample, doca_pe_progress() will eventually transition the context to idle
		 * state.
		 */
		DOCA_LOG_INFO("DMA context entered into stopping state. Any inflight tasks will be flushed");
		break;
	default:
		break;
	}
}

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
	struct pe_reactive_sample_state *state = (struct pe_reactive_sample_state *)ctx_user_data.ptr;

	state->base.num_completed_tasks++;

	/**
	 * process_completed_dma_memcpy_task returns doca_error_t to be able to use EXIT_ON_FAILURE, but there is
	 * nothing to do with the return value.
	 */
	(void)process_completed_dma_memcpy_task(dma_task, expected_value);

	/* The task is no longer required, therefore it can be freed */
	(void)dma_task_free(dma_task);

	/**
	 * The DMA context can be stopped when all tasks are completed. This section demonstrates that a context can be
	 * stopped during a completion callback, but it can be stopped at any other flow in the program.
	 */
	if (state->base.num_completed_tasks == NUM_TASKS) {
		DOCA_LOG_INFO("All tasks have completed. Stopping context");
		(void)doca_ctx_stop(state->dma_ctx);
	}
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
	struct pe_reactive_sample_state *state = (struct pe_reactive_sample_state *)ctx_user_data.ptr;
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
doca_error_t create_dma(struct pe_reactive_sample_state *state)
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

	/**
	 * This will allow sample to react to any state changes that occur during doca_pe_progress().
	 */
	EXIT_ON_FAILURE(doca_ctx_set_state_changed_cb(state->dma_ctx, dma_state_changed_callback));

	/**
	 * This will allow sample to allocate DMA tasks, while providing method that will react to completed tasks
	 * both in case task is successful or fails.
	 */
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
void cleanup(struct pe_reactive_sample_state *state)
{
	/* If all is successful then we don't enter this flow */
	if (state->dma_ctx != NULL)
		(void)doca_ctx_stop(state->dma_ctx);
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
doca_error_t run(struct pe_reactive_sample_state *state)
{
	memset(state, 0, sizeof(*state));

	state->run_pe_progress = true;
	state->base.buffer_size = BUFFER_SIZE;
	state->base.buf_inventory_size = BUF_INVENTORY_SIZE;

	EXIT_ON_FAILURE(allocate_buffer(&state->base));
	EXIT_ON_FAILURE(open_device(&state->base));
	EXIT_ON_FAILURE(create_mmap(&state->base));
	EXIT_ON_FAILURE(create_buf_inventory(&state->base));
	EXIT_ON_FAILURE(create_pe(&state->base));
	EXIT_ON_FAILURE(create_dma(state));
	EXIT_ON_FAILURE(doca_ctx_start(state->dma_ctx));

	DOCA_LOG_INFO("Polling until all tasks are completed");

	while (state->run_pe_progress) {
		/**
		 * This is the main loop of the sample. During these calls any can happen:
		 * - DMA state change callback is invoked due to a change in state.
		 * - DMA task completion callback is invoked, due to task ending in success/failure.
		 * In these callbacks the 'run_pe_progress' variable is set to false when everything is done.
		 */
		(void)doca_pe_progress(state->base.pe);
	}

	return DOCA_SUCCESS;
}

/**
 * Run the PE reactive sample
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t run_pe_reactive_sample(void)
{
	struct pe_reactive_sample_state state;
	doca_error_t status = run(&state);

	cleanup(&state);

	return status;
}
