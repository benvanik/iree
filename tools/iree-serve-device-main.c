// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-serve-device: Exposes local HAL devices to remote clients.
//
// Usage:
//   iree-serve-device --device=hip://3 --bind=tcp://0.0.0.0:5000
//   iree-serve-device --device=cuda://0 --bind=tcp://[::]:5000
//
// This tool creates a server that wraps a local HAL device and allows remote
// clients to execute operations on it. Clients connect using the remote HAL
// driver:
//
//   iree-run-module --device=remote://server:5000 --module=model.vmfb
//
// Multiple clients can connect to a single server. Each client gets an
// independent session with its own resource namespace.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/tooling/device_util.h"

// Note: We use the standard --device flag from tooling/device_util.h.
// Users specify the device as: iree-serve-device --device=hip://0

IREE_FLAG(string, bind, "tcp://0.0.0.0:5000",
          "Address to bind the server to.\n"
          "Carrier prefixes:\n"
          "  tcp://host:port       TCP sockets (default)\n"
          "  quic://host:port      QUIC/UDP (future)\n"
          "  ws://host:port/path   WebSocket (future)\n"
          "  shm:///path           Shared memory (testing)");

IREE_FLAG(int32_t, max_connections, 16,
          "Maximum number of concurrent client connections.");

IREE_FLAG(bool, rdma, false, "Enable RDMA for bulk transfers when available.");

IREE_FLAG(bool, trace, false, "Enable server operation tracing for debugging.");

// Returns a display name for a carrier type.
static const char* iree_serve_device_carrier_name(
    iree_hal_remote_server_carrier_t carrier) {
  switch (carrier) {
    case IREE_HAL_REMOTE_SERVER_CARRIER_TCP:
      return "TCP";
    case IREE_HAL_REMOTE_SERVER_CARRIER_QUIC:
      return "QUIC";
    case IREE_HAL_REMOTE_SERVER_CARRIER_WEBSOCKET:
      return "WebSocket";
    case IREE_HAL_REMOTE_SERVER_CARRIER_SHM:
      return "SHM";
    default:
      return "unknown";
  }
}

// Parses the carrier prefix from a bind URI and returns the address portion.
// Supported prefixes: tcp://, quic://, ws://, shm://
static iree_status_t iree_serve_device_parse_bind_uri(
    iree_string_view_t bind_uri, iree_hal_remote_server_carrier_t* out_carrier,
    iree_string_view_t* out_address) {
  // Try each carrier prefix.
  if (iree_string_view_consume_prefix(&bind_uri, IREE_SV("tcp://"))) {
    *out_carrier = IREE_HAL_REMOTE_SERVER_CARRIER_TCP;
    *out_address = bind_uri;
    return iree_ok_status();
  }
  if (iree_string_view_consume_prefix(&bind_uri, IREE_SV("quic://"))) {
    *out_carrier = IREE_HAL_REMOTE_SERVER_CARRIER_QUIC;
    *out_address = bind_uri;
    return iree_ok_status();
  }
  if (iree_string_view_consume_prefix(&bind_uri, IREE_SV("ws://"))) {
    *out_carrier = IREE_HAL_REMOTE_SERVER_CARRIER_WEBSOCKET;
    *out_address = bind_uri;
    return iree_ok_status();
  }
  if (iree_string_view_consume_prefix(&bind_uri, IREE_SV("shm://"))) {
    *out_carrier = IREE_HAL_REMOTE_SERVER_CARRIER_SHM;
    *out_address = bind_uri;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "bind URI must have a carrier prefix (tcp://, quic://, ws://, shm://), "
      "got: '%.*s'",
      (int)bind_uri.size, bind_uri.data);
}

static iree_status_t iree_serve_device_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();

  // Create the local device to serve.
  // Uses the --device flag from tooling/device_util.h.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_create_device_from_flags(
      iree_hal_available_driver_registry(),
      /*default_device=*/iree_string_view_empty(), host_allocator, &device));

  iree_string_view_t device_id = iree_hal_device_id(device);
  fprintf(stdout, "Created device: %.*s\n", (int)device_id.size,
          device_id.data);

  // Configure server options.
  iree_hal_remote_server_options_t options;
  iree_hal_remote_server_options_initialize(&options);

  // Parse carrier and address from --bind flag.
  iree_hal_remote_server_carrier_t carrier;
  iree_string_view_t bind_address;
  IREE_RETURN_IF_ERROR(iree_serve_device_parse_bind_uri(
      iree_make_cstring_view(FLAG_bind), &carrier, &bind_address));
  options.carrier = carrier;
  options.bind_address = bind_address;

  options.max_connections = (uint32_t)FLAG_max_connections;
  if (FLAG_rdma) {
    options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
  }
  if (FLAG_trace) {
    options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS;
  }

  // Create the server.
  fprintf(stdout, "Creating server: carrier=%s address=%.*s\n",
          iree_serve_device_carrier_name(carrier), (int)bind_address.size,
          bind_address.data);
  iree_hal_remote_server_t* server = NULL;
  iree_status_t status =
      iree_hal_remote_server_create(&options, device, host_allocator, &server);

  // Run the server event loop.
  // The server is built on iree/async/proactor.h and handles shutdown
  // through the proactor's cancellation mechanism rather than signals.
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Server starting...\n");
    status = iree_hal_remote_server_run(server);
  }

  // Cleanup.
  iree_hal_remote_server_release(server);
  iree_hal_device_release(device);

  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-serve-device",
      "Exposes local HAL devices to remote clients over the network.\n"
      "\n"
      "Examples:\n"
      "  # Serve a HIP device on port 5000 over TCP\n"
      "  iree-serve-device --device=hip://0 --bind=tcp://0.0.0.0:5000\n"
      "\n"
      "  # Serve over shared memory (for local testing)\n"
      "  iree-serve-device --device=hip://0 --bind=shm:///dev/shm/iree-gpu\n"
      "\n"
      "  # Connect from another machine (client uses matching carrier)\n"
      "  iree-run-module --device=remote-tcp://server:5000 "
      "--module=model.vmfb\n"
      "\n"
      "  # Connect via shared memory\n"
      "  iree-run-module --device=remote-shm:///dev/shm/iree-gpu "
      "--module=model.vmfb\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_serve_device_run();

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
