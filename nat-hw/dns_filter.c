#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "doca.h"
#include "dns_filter.h"

/* DOCA compatible */
__thread regex_t compiled_rules[MAX_RULES];
__thread int rule_count = 0;

int load_regex_rules(void) {
    FILE *file = fopen("/home/ubuntu/Nutcracker/apps/nf_dns/dns_filter_rules_5.txt", "r");
    if (!file) {
        perror("Failed to open file");
        return -1;
    }

    char regex[MAX_REGEX_LENGTH];
    int ret;
    while (fgets(regex, MAX_REGEX_LENGTH, file)) {
        if (regex[strlen(regex) - 1] == '\n') {
            regex[strlen(regex) - 1] = '\0';  // Remove newline character
        }

        ret = regcomp(&compiled_rules[rule_count], regex, REG_EXTENDED);
        if (ret) {
            fprintf(stderr, "Could not compile regex: %s\n", regex);
            continue;
        }
        rule_count++;
        if (rule_count >= MAX_RULES) break;
    }

    fclose(file);
    return 0;
}

struct dns_header {
    uint16_t id; // Transaction ID
    uint16_t flags; // DNS flags
    uint16_t qdcount; // Number of questions
    uint16_t ancount; // Number of answers
    uint16_t nscount; // Number of authority records
    uint16_t arcount; // Number of additional records
};

int report_results(struct doca_event *event) {
	struct doca_regex_match_metadata * const meta = (struct doca_regex_match_metadata *)event->user_data.ptr;
	struct doca_regex_search_result * const result = &(meta->result);
	struct doca_regex_match *match;
    printf("Detected matches: %d\n", result->detected_matches);

	if (result->detected_matches > 0)
		printf("Job complete. Detected %d match(es), num matched: %d\n", result->detected_matches, result->num_matches);
	if (result->num_matches == 0)
		return 0;

	for (match = result->matches; match != NULL;) {
		printf("Matched rule Id: %12d\n", match->rule_id);
		match = match->next;
	}

	result->matches = NULL;
	return 0;
}

// Function to print a domain name from a DNS query
void print_domain_name(const unsigned char * buffer, int* position, unsigned char * domain_name) {
    int len = buffer[(*position)++];
    while (len > 0) {
        for (int i = 0; i < len; i++) {
            *(domain_name++) = buffer[(*position)++];
        }
        len = buffer[(*position)++];
        if (len > 0) {
            *(domain_name++) = '.';
        }
    }
}

// Parse and print details from a DNS query
int parse_dns_query(const unsigned char * buffer, int size) {
    unsigned char domain_name[256] = {0};
    // Cast the buffer to the DNS header struct
    // struct dns_header* dns = (struct dns_header*)buffer;
    int position = sizeof(struct dns_header); // Position in the buffer
    print_domain_name(buffer, &position, domain_name);
    if (find_matching_rule((const char *)domain_name) < 0) {
        return -1;
    }
    return 0;
}

int find_matching_rule(const char * domain_name) {
#ifdef CONFIG_DOCA
    struct doca_regex_match_metadata meta, *ret;
    int res;
	struct doca_event event = {0};
    void *mbuf_data;
	struct doca_regex_ctx * regex_ctx = &ctx->regex_ctx;
    // double elapsed_time;

    memcpy(regex_ctx->data_buffer, domain_name, strlen(domain_name));

    doca_buf_get_data(regex_ctx->buf, &mbuf_data);
    doca_buf_set_data(regex_ctx->buf, mbuf_data, strlen(domain_name));

    struct doca_regex_job_search const job = {
        .base = {
            .ctx = doca_regex_as_ctx(regex_ctx->doca_reg),
            .type = DOCA_REGEX_JOB_SEARCH,
            .user_data = { .ptr = &meta },
        },
        .rule_group_ids = { 1, 0, 0, 0 },
        .buffer = regex_ctx->buf,
        .result = &(meta.result),
        .allow_batching = 0
    };

    // struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    res = doca_workq_submit(ctx->workq, (struct doca_job *)&job);
    if (res == DOCA_SUCCESS) {
        /* store ref to job data so it can be released once a result is obtained */
        meta.job_data = regex_ctx->buf;
    } else if (res == DOCA_ERROR_NO_MEMORY) {
        printf("Unable to enqueue job. [%s]", doca_get_error_string(res));
        return -1;
    } else {
        printf("Unable to enqueue job. [%s]", doca_get_error_string(res));
        return -1;
    }

    do {
		res = doca_workq_progress_retrieve(ctx->workq, &event, DOCA_WORKQ_RETRIEVE_FLAGS_NONE);
		if (res == DOCA_SUCCESS) {
			/* Handle the completed jobs */
			ret = (struct doca_regex_match_metadata *)event.user_data.ptr;
			if (ret->result.status_flags & DOCA_REGEX_STATUS_SEARCH_FAILED) {
				printf("RegEx search failed\n");
				if (ret->result.status_flags & DOCA_REGEX_STATUS_MAX_MATCH)
					printf("DOCA RegEx engine reached maximum number of matches, should reduce job size by using \"chunk-size\" flag\n");
				/* In case there are other jobs in workq, need to dequeue them and then to exit */
				return -1;
			} else {
				// report_results(&event);
				struct doca_regex_match_metadata * search_meta = (struct doca_regex_match_metadata *)event.user_data.ptr;
				struct doca_regex_search_result * search_result = &(search_meta->result);
				if (search_result->detected_matches > 0) {
                    // clock_gettime(CLOCK_MONOTONIC, &end);
                    // elapsed_time = (end.tv_sec - start.tv_sec) * 1e9;    // seconds to nanoseconds
                    // elapsed_time += (end.tv_nsec - start.tv_nsec);       // add nanoseconds
                    // fprintf(stderr, "%.9f\n", elapsed_time);
                    return 0;
                } else return -1; 
            }
			doca_buf_refcount_rm(ret->job_data, NULL);
		} else if (res == DOCA_ERROR_AGAIN) {
			continue;
		} else {
			printf("Unable to dequeue results. [%s]\n", doca_get_error_string(res));
			return -1;
		}
	} while (res != DOCA_SUCCESS);

	return 0;
#else
    int result;
    // double elapsed_time;
    // struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    // Iterate through all compiled rules
    for (int i = 0; i < rule_count; i++) {
        result = regexec(&compiled_rules[i], domain_name, 0, NULL, 0);
        if (result == 0) {
            // printf("Match found with rule %d: %s\n", i, domain_name);
            // clock_gettime(CLOCK_MONOTONIC, &end);
            // elapsed_time = (end.tv_sec - start.tv_sec) * 1e9;    // seconds to nanoseconds
            // elapsed_time += (end.tv_nsec - start.tv_nsec);       // add nanoseconds
            // fprintf(stderr, "%.9f\n", elapsed_time);
            
            return i;  // Return the index of the first matching rule
        }
    }

    // printf("No match found for: %s\n", domain_name);
    return -1;  // Return -1 if no match is found
#endif
}