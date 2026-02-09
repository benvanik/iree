// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for zero-copy socket operations.
//
// These tests verify that zerocopy options can be enabled (when available) and
// that data transfer works correctly with zerocopy enabled.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

// Tests for SO_ZEROCOPY socket option (MSG_ZEROCOPY on Linux 4.14+).
// Zero-copy send avoids kernel buffer copies but requires specific kernel
// support. These tests verify the option can be enabled (when available) and
// that data integrity is maintained.
class ZeroCopyTest : public SocketTestBase<> {
 protected:
  // Checks if zero-copy is available on this system.
  bool IsZeroCopyAvailable() {
    iree_async_socket_t* test_socket = nullptr;
    iree_status_t status = iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_ZERO_COPY, &test_socket);
    if (iree_status_is_ok(status)) {
      iree_async_socket_release(test_socket);
      return true;
    }
    iree_status_ignore(status);
    return false;
  }
};

// Creating a socket with ZERO_COPY option should succeed on supported kernels
// or fail gracefully with UNAVAILABLE on older kernels.
TEST_P(ZeroCopyTest, ZeroCopy_CreateSocket) {
  iree_async_socket_t* socket = nullptr;
  iree_status_t status =
      iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                               IREE_ASYNC_SOCKET_OPTION_ZERO_COPY, &socket);

  if (iree_status_is_ok(status)) {
    // Success: zero-copy is available.
    ASSERT_NE(socket, nullptr);
    EXPECT_EQ(iree_async_socket_query_state(socket),
              IREE_ASYNC_SOCKET_STATE_CREATED);
    iree_async_socket_release(socket);
  } else {
    // Expected on older kernels: should be UNAVAILABLE.
    EXPECT_TRUE(iree_status_is_unavailable(status))
        << "Expected UNAVAILABLE for unsupported zero-copy, got: "
        << iree_status_code(status);
    iree_status_ignore(status);
    EXPECT_EQ(socket, nullptr);
  }
}

// When zero-copy is available, verify data integrity with a send/recv.
TEST_P(ZeroCopyTest, ZeroCopy_SendRecv) {
  if (!IsZeroCopyAvailable()) {
    GTEST_SKIP() << "SO_ZEROCOPY not available on this kernel";
  }

  // Create listener without zero-copy (server doesn't need it for recv).
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create client with zero-copy enabled.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(
      proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
      IREE_ASYNC_SOCKET_OPTION_ZERO_COPY | IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
      &client));

  // Connect.
  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_NE(accept_op.accepted_socket, nullptr);
  iree_async_socket_t* server = accept_op.accepted_socket;

  // Send data from zero-copy client.
  const char* send_data = "Zero-copy test data for verification!";
  iree_host_size_t send_length = strlen(send_data);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, send_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive on server.
  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, send_length);

  // Verify data integrity.
  EXPECT_EQ(memcmp(recv_buffer, send_data, send_length), 0)
      << "Zero-copy data corruption detected";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Combining ZERO_COPY with other options should work.
TEST_P(ZeroCopyTest, ZeroCopy_CombinedOptions) {
  iree_async_socket_t* socket = nullptr;
  iree_status_t status = iree_async_socket_create(
      proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
      IREE_ASYNC_SOCKET_OPTION_ZERO_COPY | IREE_ASYNC_SOCKET_OPTION_NO_DELAY |
          IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
      &socket);

  if (iree_status_is_ok(status)) {
    ASSERT_NE(socket, nullptr);
    // Should be able to bind with combined options.
    iree_async_address_t address;
    IREE_ASSERT_OK(iree_async_address_from_ipv4(
        iree_make_cstring_view("127.0.0.1"), 0, &address));
    IREE_EXPECT_OK(iree_async_socket_bind(socket, &address));
    iree_async_socket_release(socket);
  } else {
    // Zero-copy not available - that's fine.
    EXPECT_TRUE(iree_status_is_unavailable(status));
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE_WITH_TAGS(ZeroCopyTest, /*required=*/{"zerocopy"},
                                  /*excluded=*/{});

}  // namespace iree::async::cts
