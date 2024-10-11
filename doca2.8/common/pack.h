/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#ifndef COMMON_PACK_H_
#define COMMON_PACK_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Get LSB at position N from logical value V */
#define GET_BYTE(V, N) ((uint8_t)((V) >> ((N)*8) & 0xFF))
/* Set byte value V at the LSB position N */
#define SET_BYTE(V, N) (((V)&0xFF) << ((N)*8))

/*
 * 64-bit extensions to regular host-to-network/network-to-host functions
 *
 * @value [in]: value to convert
 * @return: host byte order/network byte order
 */
uint64_t ntohq(uint64_t value);
#define htonq ntohq

/*
 * Pack an 8-bit numeric value into a work buffer, and advance the write head.
 *
 * @buffer [in]: pointer to a write-head to write into, and to increment
 * @value [in]: value to pack
 */
void pack_uint8(uint8_t **buffer, uint8_t value);

/*
 * Pack a 16-bit numeric value into a work buffer in Big Endian, and advance the write head.
 *
 * @buffer [in]: pointer to a write-head to write into, and to increment
 * @value [in]: value to pack
 */
void pack_uint16(uint8_t **buffer, uint16_t value);

/*
 * Pack a 32-bit numeric value into a work buffer in Big Endian, and advance the write head.
 *
 * @buffer [in]: pointer to a write-head to write into, and to increment
 * @value [in]: value to pack
 */
void pack_uint32(uint8_t **buffer, uint32_t value);

/*
 * Pack a 64-bit numeric value into a work buffer in Big Endian, and advance the write head.
 *
 * @buffer [in]: pointer to a write-head to write into, and to increment
 * @value [in]: value to pack
 */
void pack_uint64(uint8_t **buffer, uint64_t value);

/*
 * Pack a binary large object into a work buffer, and advance the write head.
 *
 * @buffer [in]: pointer to a write-head to write into, and to increment
 * @length [in]: object size to pack
 * @object [in]: pointer to byte array to be packed
 */
void pack_blob(uint8_t **buffer, size_t length, uint8_t *object);

/*
 * Unpack an 8-bit numeric value from a work buffer, and advance the read head.
 *
 * @buffer [in]: pointer to a read-head to read from, and to increment
 * @return: unpacked numerical value
 */
uint8_t unpack_uint8(uint8_t **buffer);

/*
 * Unpack a 16-bit Big Endian numeric value from a work buffer, and advance the read head.
 *
 * @buffer [in]: pointer to a read-head to read from, and to increment
 * @return: unpacked numerical value
 */
uint16_t unpack_uint16(uint8_t **buffer);

/*
 * Unpack a 32-bit Big Endian numeric value from a work buffer, and advance the read head.
 *
 * @buffer [in]: pointer to a read-head to read from, and to increment
 * @return: unpacked numerical value
 */
uint32_t unpack_uint32(uint8_t **buffer);

/*
 * Unpack a 64-bit Big Endian numeric value from a work buffer, and advance the read head.
 *
 * @buffer [in]: pointer to a read-head to read from, and to increment
 * @return: unpacked numerical value
 */
uint64_t unpack_uint64(uint8_t **buffer);

/*
 * Unpack a binary large object from a work buffer, and advance the read head.
 *
 * @buffer [in]: pointer to a read-head to read from, and to increment
 * @length [in]: object size to unpack
 * @object [out]: pointer to hold received byte array
 */
void unpack_blob(uint8_t **buffer, size_t length, uint8_t *object);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COMMON_PACK_H_ */
