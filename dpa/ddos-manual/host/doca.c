#include <string.h>
#include <unistd.h>

#include <doca_log.h>
#include <doca_flow.h>

#include "flow_common.h"

doca_error_t doca_init(int nb_queues)
{
	int nb_ports = 2;
#ifdef ENABLE_COUNTER
	struct flow_resources resource = {.nr_counters = 64};
#else
	struct flow_resources resource = {0};
#endif	/* ENABLE_COUNTER */
	uint32_t nr_shared_resources[SHARED_RESOURCE_NUM_VALUES] = {0};
	struct doca_flow_port *ports[nb_ports];
	struct doca_dev *dev_arr[nb_ports];
	struct doca_flow_pipe *pipe;
	struct entries_status status_ingress;
	int num_of_entries_ingress = 1;
	struct entries_status status_egress;
	int num_of_entries_egress = 1;
	doca_error_t result;
	int port_id;

	result = init_doca_flow(nb_queues, "vnf,hws", &resource, nr_shared_resources);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA Flow: %s\n", doca_error_get_descr(result));
		return result;
	}

	memset(dev_arr, 0, sizeof(struct doca_dev *) * nb_ports);
	result = init_doca_flow_ports(nb_ports, ports, true, dev_arr);
	if (result != DOCA_SUCCESS) {
		printf("Failed to init DOCA ports: %s\n", doca_error_get_descr(result));
		doca_flow_destroy();
		return result;
	}

	for (port_id = 0; port_id < nb_ports; port_id++) {
		memset(&status_ingress, 0, sizeof(status_ingress));
		memset(&status_egress, 0, sizeof(status_egress));

		result = create_match_pipe(ports[port_id], port_id, &pipe);
		if (result != DOCA_SUCCESS) {
			printf("Failed to create match pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = add_match_pipe_entry(pipe, &status_ingress, &match_entry[port_id]);
		if (result != DOCA_SUCCESS) {
			printf("Failed to add entry to match pipe: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		result = doca_flow_entries_process(ports[port_id], 0, DEFAULT_TIMEOUT_US, num_of_entries_ingress);
		if (result != DOCA_SUCCESS) {
			printf("Failed to process entries: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return result;
		}

		if (status_ingress.nb_processed != num_of_entries_ingress || status_ingress.failure) {
			printf("Failed to process entries: %s\n", doca_error_get_descr(result));
			stop_doca_flow_ports(nb_ports, ports);
			doca_flow_destroy();
			return DOCA_ERROR_BAD_STATE;
		}
	}

	return result;
}
