// SPDX-License-Identifier: Apache-2.0
//
// Shared-token connection authentication for the internal planes (discoveryd,
// control plane, data plane). When a token is configured (DYN_AUTH_TOKEN or
// set_token()), every new connection starts with one auth frame carrying the
// token; servers verify it (constant-time) before serving anything. Without a
// token both helpers are no-ops, so mixed configurations fail closed on the
// side that has the token. The HTTP frontend is not covered (standard HTTP
// auth belongs in a proxy).
//
// The token authenticates cluster membership, not confidentiality — pair it
// with TLS (tls.h) on untrusted networks or the token travels in cleartext.

#pragma once

#include <optional>
#include <string>

namespace dynamo::transports {

class Socket;

namespace auth {

/// Overrides the DYN_AUTH_TOKEN environment configuration (nullopt disables).
void set_token(std::optional<std::string> token);

/// True when a token is configured.
bool required();

/// Client side: sends the auth frame when a token is configured. Returns
/// false when the write fails.
bool send(Socket& sock);

/// Server side: when a token is configured, reads the connection's first
/// frame (bounded to 5s) and verifies the token; restores unbounded reads on
/// success. False = reject the connection.
bool expect(Socket& sock);

}  // namespace auth
}  // namespace dynamo::transports
