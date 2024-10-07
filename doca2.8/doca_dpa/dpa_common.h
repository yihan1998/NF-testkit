/*
 * Copyright (c) 2022-2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#ifndef DPA_COMMON_H_
#define DPA_COMMON_H_

#include <stdlib.h>
#include <unistd.h>

#include <doca_dev.h>
#include <doca_dpa.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_types.h>
#include <doca_argp.h>
#include <doca_sync_event.h>
#include <doca_mmap.h>
#include <doca_rdma.h>
#include <doca_ctx.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Macro for sleep (wait) in seconds
 */
#define SLEEP(SECONDS) for (int i = 0; i < 1 + SECONDS * 30000; i++)

/**
 * @brief device default name
 */
#define DEVICE_DEFAULT_NAME "NOT_SET"

/**
 * @brief Mask for doca_sync_event_wait_gt() wait value
 */
#define SYNC_EVENT_MASK_FFS (0xFFFFFFFFFFFFFFFF)

/**
 * @brief A struct that includes all the resources needed for DPA
 */
struct dpa_resources {
	struct doca_dev *doca_device; /**< DOCA device for DPA */
	struct doca_dpa *doca_dpa;    /**< DOCA DPA context */
};

/**
 * @brief Configuration struct
 */
struct dpa_config {
	char device_name[DOCA_DEVINFO_IBDEV_NAME_SIZE]; /**< Buffer that holds the device name */
};

/**
 * @brief A struct that includes all the resources needed for DPA thread
 */
struct dpa_thread_obj {
	struct doca_dpa *doca_dpa;	    /**< DOCA DPA context */
	doca_dpa_func_t *func;		    /**< DPA thread entry point */
	uint64_t arg;			    /**< DPA thread entry point argument */
	doca_dpa_dev_uintptr_t tls_dev_ptr; /**< DPA thread local storage device memory pointer */
	struct doca_dpa_thread *thread;	    /**< Created DPA thread */
};

/**
 * @brief A struct that includes all the resources needed for DPA completion
 */
struct dpa_completion_obj {
	struct doca_dpa *doca_dpa;	      /**< DOCA DPA context */
	unsigned int queue_size;	      /**< DPA completion queue size */
	struct doca_dpa_thread *thread;	      /**< DPA completion attached thread */
	struct doca_dpa_completion *dpa_comp; /**< Created DPA completion */
	doca_dpa_dev_completion_t handle;     /**< Created DPA completion device handle */
};

/**
 * @brief A struct that includes all the resources needed for DPA notification completion
 */
struct dpa_notification_completion_obj {
	struct doca_dpa *doca_dpa;				    /**< DOCA DPA context */
	struct doca_dpa_thread *thread;				    /**< DPA notification completion attached thread */
	struct doca_dpa_notification_completion *notification_comp; /**< Created DPA notification completion */
	doca_dpa_dev_notification_completion_t handle; /**< Created DPA notification completion device handle*/
};

/**
 * @brief A struct that includes all the resources needed for DPA RDMA
 */
struct dpa_rdma_obj {
	struct doca_dev *doca_device;	/**< DOCA device for DPA */
	struct doca_dpa *doca_dpa;	/**< DOCA DPA context */
	uint32_t permissions;		/**< RDMA permissions */
	union doca_data user_data;	/**< RDMA user data */
	struct doca_rdma *rdma;		/**< Created RDMA */
	struct doca_ctx *rdma_as_ctx;	/**< Created RDMA context */
	doca_dpa_dev_rdma_t dpa_rdma;	/**< Created RDMA DPA device handle */
	const void *connection_details; /**< Created RDMA connection details from export */
	size_t conn_det_len;		/**< Created RDMA connection details length from export */
};

/**
 * @brief A struct that includes all the resources needed for DPA SRQ
 */
struct dpa_rdma_srq_obj {
	struct doca_dev *doca_device;	      /**< DOCA device for DPA */
	struct doca_dpa *doca_dpa;	      /**< DOCA DPA context */
	uint32_t srq_size;		      /**< SRQ queue size */
	uint32_t buf_list_len;		      /**< SRQ buffer list length */
	enum doca_rdma_srq_type srq_type;     /**< SRQ type */
	struct doca_rdma_srq *rdma_srq;	      /**< Created SRQ */
	struct doca_ctx *rdma_srq_as_ctx;     /**< Created SRQ context */
	doca_dpa_dev_rdma_srq_t dpa_rdma_srq; /**< Created SRQ DPA device handle */
};

/**
 * @brief DOCA Mmap type definition
 */
enum mmap_type {
	MMAP_TYPE_CPU,
	MMAP_TYPE_DPA,
};

/**
 * @brief A struct that includes all the resources needed for DOCA Mmap
 */
struct doca_mmap_obj {
	enum mmap_type mmap_type;	     /**< Mmap type */
	struct doca_dev *doca_device;	     /**< DOCA device for DPA */
	struct doca_dpa *doca_dpa;	     /**< DOCA DPA context */
	uint32_t permissions;		     /**< Mmap permissions */
	void *memrange_addr;		     /**< Mmap address */
	size_t memrange_len;		     /**< Mmap address length */
	struct doca_mmap *mmap;		     /**< Created Mmap */
	doca_dpa_dev_mmap_t dpa_mmap_handle; /**< Created Mmap DPA device handle */
	const void *rdma_export;	     /**< Exported Mmap to be used for RDMA operations */
	size_t export_len;		     /**< Exported Mmap length */
};

/**
 * @brief Register the command line parameters for the sample
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t register_dpa_params(void);

/**
 * @brief Create DOCA sync event to be published by the CPU and subscribed by the DPA
 *
 * @doca_dpa [in]: DOCA DPA context
 * @doca_device [in]: DOCA device
 * @wait_event [out]: Created DOCA sync event that is published by the CPU and subscribed by the DPA
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_doca_dpa_wait_sync_event(struct doca_dpa *doca_dpa,
					     struct doca_dev *doca_device,
					     struct doca_sync_event **wait_event);

/**
 * @brief Create DOCA sync event to be published by the DPA and subscribed by the CPU
 *
 * @doca_dpa [in]: DOCA DPA context
 * @doca_device [in]: DOCA device
 * @comp_event [out]: Created DOCA sync event that is published by the DPA and subscribed by the CPU
 * @handle [out]: Created DOCA sync event handle
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_doca_dpa_completion_sync_event(struct doca_dpa *doca_dpa,
						   struct doca_dev *doca_device,
						   struct doca_sync_event **comp_event,
						   doca_dpa_dev_sync_event_t *handle);

/**
 * @brief Create DOCA sync event to be published and subscribed by the DPA
 *
 * @doca_dpa [in]: DOCA DPA context
 * @kernel_event [out]: Created DOCA sync event that is published and subscribed by the DPA
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_doca_dpa_kernel_sync_event(struct doca_dpa *doca_dpa, struct doca_sync_event **kernel_event);

/**
 * @brief Create DOCA sync event to be published by a remote net and subscribed by the CPU
 *
 * @doca_device [in]: DOCA device
 * @remote_net_event [out]: Created DOCA sync event that is published by a remote net and subscribed by the CPU
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_doca_remote_net_sync_event(struct doca_dev *doca_device, struct doca_sync_event **remote_net_event);

/**
 * @brief Create DOCA sync event to be published by a remote net and subscribed by the CPU
 *
 * @doca_device [in]: DOCA device
 * @doca_dpa [in]: DOCA DPA context
 * @remote_net_event [in]: remote net DOCA sync event
 * @remote_net_exported_event [out]: Created from export remote net DOCA sync event
 * @remote_net_event_dpa_handle [out]: DPA handle of the created from export remote net DOCA sync event
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t export_doca_remote_net_sync_event_to_dpa(struct doca_dev *doca_device,
						      struct doca_dpa *doca_dpa,
						      struct doca_sync_event *remote_net_event,
						      struct doca_sync_event_remote_net **remote_net_exported_event,
						      doca_dpa_dev_sync_event_remote_net_t *remote_net_event_dpa_handle);

/**
 * @brief Allocate DOCA DPA resources
 *
 * @cfg [in]: DOCA DPA configurations
 * @resources [out]: DOCA DPA resources to allocate
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t allocate_dpa_resources(struct dpa_config *cfg, struct dpa_resources *resources);

/**
 * @brief Destroy DOCA DPA resources
 *
 * @resources [in]: DOCA DPA resources to destroy
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t destroy_dpa_resources(struct dpa_resources *resources);

/**
 * @brief Initialize DPA thread
 *
 * @dpa_thread_obj [in/out]: DPA thread object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_thread_obj_init(struct dpa_thread_obj *dpa_thread_obj);

/**
 * @brief Destroy DPA thread
 *
 * @dpa_thread_obj [in]: DPA thread object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_thread_obj_destroy(struct dpa_thread_obj *dpa_thread_obj);

/**
 * @brief Initialize DPA completion
 *
 * @dpa_completion_obj [in/out]: DPA completion object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_completion_obj_init(struct dpa_completion_obj *dpa_completion_obj);

/**
 * @brief Destroy DPA completion
 *
 * @dpa_completion_obj [in]: DPA completion object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_completion_obj_destroy(struct dpa_completion_obj *dpa_completion_obj);

/**
 * @brief Initialize DPA notification completion
 *
 * @notification_completion_obj [in/out]: DPA notification completion object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_notification_completion_obj_init(struct dpa_notification_completion_obj *notification_completion_obj);

/**
 * @brief Destroy DPA notification completion
 *
 * @notification_completion_obj [in]: DPA notification completion object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_notification_completion_obj_destroy(
	struct dpa_notification_completion_obj *notification_completion_obj);

/**
 * @brief Initialize DPA RDMA SRQ
 *
 * @dpa_rdma_srq_obj [in/out]: DPA RDMA SRQ object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_rdma_srq_obj_init(struct dpa_rdma_srq_obj *dpa_rdma_srq_obj);

/**
 * @brief Destroy DPA RDMA SRQ
 *
 * @dpa_rdma_srq_obj [in]: DPA RDMA SRQ object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_rdma_srq_obj_destroy(struct dpa_rdma_srq_obj *dpa_rdma_srq_obj);

/**
 * @brief Initialize DPA RDMA without starting it
 *
 * This function creates DPA RDMA object.
 * This function attaches RDMA to SRQ in case of DPA RDMA SRQ is provided.
 * Please note that this function doesn't start the created DPA RDMA object, this need to be done
 * using dpa_rdma_obj_start() API
 *
 * @dpa_rdma_srq_obj [in]: DPA RDMA SRQ object
 * @dpa_rdma_obj [in/out]: DPA RDMA object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_rdma_obj_init(struct dpa_rdma_srq_obj *dpa_rdma_srq_obj, struct dpa_rdma_obj *dpa_rdma_obj);

/**
 * @brief Start DPA RDMA
 *
 * @dpa_rdma_obj [in]: DPA RDMA object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_rdma_obj_start(struct dpa_rdma_obj *dpa_rdma_obj);

/**
 * @brief Destroy DPA RDMA
 *
 * @dpa_rdma_obj [in]: DPA RDMA object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpa_rdma_obj_destroy(struct dpa_rdma_obj *dpa_rdma_obj);

/**
 * @brief Initialize DOCA Mmap
 *
 * @doca_mmap_obj [in/out]: DOCA Mmap object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t doca_mmap_obj_init(struct doca_mmap_obj *doca_mmap_obj);

/**
 * @brief Destroy DOCA Mmap
 *
 * @doca_mmap_obj [in]: DOCA Mmap object
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t doca_mmap_obj_destroy(struct doca_mmap_obj *doca_mmap_obj);

#ifdef __cplusplus
}
#endif

#endif /* DPA_COMMON_H_ */
