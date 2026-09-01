#!/usr/bin/env python3
"""A stdio MCP server that can be told to misbehave.

One script, many roles, so the Rust tests drive the *real* `McpClient` against
a real child process instead of a hand-mocked pipe. The role is argv[1] (or
$MOCK_MCP_ROLE); every frame the client sends is appended to the file named by
argv[2] (or $MOCK_MCP_RECORD), which is how the tests assert the exact JSON-RPC
on the wire without racing on process-wide environment.

Roles are documented next to their branch in `dispatch()`. Every role that
would otherwise block forever has a hard self-destruct so a wedged test cannot
leave a python process behind.
"""

import json
import os
import sys
import time

ROLE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MOCK_MCP_ROLE", "ok")
RECORD = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MOCK_MCP_RECORD")

# No role may outlive the test that spawned it, even if the client wedges.
SELF_DESTRUCT_SECS = 30

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the arguments back as text",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "what to echo"}},
            "required": ["text"],
        },
    },
    {
        # No `description` key at all: the client must supply a placeholder
        # rather than register a tool the registry will reject.
        "name": "undescribed",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "explode",
        "description": "Always fails in-band",
        "inputSchema": {"type": "object"},
    },
]


def emit(text):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def emit_json(obj):
    emit(json.dumps(obj))


def note(text):
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def read_request():
    """Next non-blank frame from stdin, or None at EOF."""
    while True:
        line = sys.stdin.readline()
        if line == "":
            return None
        if RECORD:
            with open(RECORD, "a") as fh:
                fh.write(line if line.endswith("\n") else line + "\n")
        if line.strip() == "":
            continue
        return json.loads(line)


def dispatch(req):
    """The well-behaved answer to one request, or None for a notification."""
    method = req.get("method")
    if method == "initialize":
        return {
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mock-mcp", "version": "0.1.0"},
            }
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"result": {"tools": TOOLS}}
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments")
        if name == "echo":
            return {
                "result": {
                    "content": [{"type": "text", "text": json.dumps(args, sort_keys=True)}],
                    "isError": False,
                }
            }
        if name == "undescribed":
            return {"result": {"content": [{"type": "text", "text": "ok"}]}}
        if name == "explode":
            return {
                "result": {
                    "content": [{"type": "text", "text": "the tool ran and failed"}],
                    "isError": True,
                }
            }
        return {"error": {"code": -32602, "message": "unknown tool: %s" % name}}
    return {"error": {"code": -32601, "message": "method not found: %s" % method}}


def respond(req, body):
    if body is None:
        return
    frame = {"jsonrpc": "2.0", "id": req.get("id")}
    frame.update(body)
    emit_json(frame)


def serve(transform=None):
    """Answer every request, optionally rewriting the frame first."""
    deadline = time.time() + SELF_DESTRUCT_SECS
    while time.time() < deadline:
        req = read_request()
        if req is None:
            return
        body = dispatch(req)
        if transform is not None:
            transform(req, body)
        else:
            respond(req, body)


def main():
    if ROLE == "ok":
        serve()

    elif ROLE == "exit-immediately":
        # Never reads a byte: the client's very first write may land on a
        # closed pipe, or may succeed and then hit EOF on the read.
        return

    elif ROLE == "die-after-initialize":
        req = read_request()
        if req is not None:
            respond(req, dispatch(req))
        return

    elif ROLE == "malformed-json":
        read_request()
        emit('{"jsonrpc": "2.0", "id": 1, "result": {')

    elif ROLE == "not-json":
        read_request()
        emit("mock-mcp: starting up, please hold")
        serve()

    elif ROLE == "silent":
        # Drains stdin and answers nothing. The client's timeout is the only
        # thing that ends this conversation.
        time.sleep(SELF_DESTRUCT_SECS)

    elif ROLE == "rpc-error":
        # Handshakes and lists normally; every call comes back as a JSON-RPC
        # error object rather than a result.
        def transform(req, body):
            if req.get("method") == "tools/call":
                body = {
                    "error": {
                        "code": -32603,
                        "message": "the server is having a bad day",
                        "data": {"detail": "disk on fire"},
                    }
                }
            respond(req, body)

        serve(transform)

    elif ROLE == "wrong-id":
        read_request()
        emit_json({"jsonrpc": "2.0", "id": 9999, "result": {"protocolVersion": "2024-11-05"}})
        time.sleep(SELF_DESTRUCT_SECS)

    elif ROLE == "noise-before-answer":
        # A mismatched id, a notification carrying no id, and blank lines all
        # precede the real answer; only the correlated frame may be taken.
        def transform(req, body):
            emit("")
            emit_json({"jsonrpc": "2.0", "id": 9999, "result": {"protocolVersion": "0.0.0"}})
            emit_json({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
            emit("   ")
            respond(req, body)

        serve(transform)

    elif ROLE == "no-result-no-error":
        req = read_request()
        emit_json({"jsonrpc": "2.0", "id": req.get("id")})

    elif ROLE == "extra-fields":
        # Unknown members at every level, which a forward-compatible client
        # must ignore rather than reject.
        def transform(req, body):
            if body is None:
                return
            frame = {"jsonrpc": "2.0", "id": req.get("id"), "meta": {"trace": "abc"}}
            frame.update(body)
            if "result" in frame and isinstance(frame["result"], dict):
                frame["result"]["_experimental"] = [1, 2, 3]
                for tool in frame["result"].get("tools", []):
                    tool["annotations"] = {"readOnlyHint": True}
            emit_json(frame)

        serve(transform)

    elif ROLE == "huge":
        def transform(req, body):
            if req.get("method") == "tools/call":
                body = {
                    "result": {
                        "content": [{"type": "text", "text": "z" * (2 * 1024 * 1024)}],
                        "isError": False,
                    }
                }
            respond(req, body)

        serve(transform)

    elif ROLE == "no-tools":
        def transform(req, body):
            if req.get("method") == "tools/list":
                body = {"result": {"tools": []}}
            respond(req, body)

        serve(transform)

    elif ROLE == "silent-after-list":
        # Handshakes and lists normally, then stops answering — the shape of a
        # server that wedges on one particular tool.
        def transform(req, body):
            if req.get("method") == "tools/call":
                time.sleep(SELF_DESTRUCT_SECS)
                return
            respond(req, body)

        serve(transform)

    elif ROLE == "die-after-list":
        def transform(req, body):
            if req.get("method") == "tools/call":
                sys.exit(0)
            respond(req, body)

        serve(transform)

    elif ROLE == "noisy-stderr":
        def transform(req, body):
            note("mock-mcp: handling %s" % req.get("method"))
            note("mock-mcp: still here")
            respond(req, body)
            note("mock-mcp: done")

        serve(transform)

    else:
        note("mock-mcp: unknown role %r" % ROLE)
        sys.exit(2)


if __name__ == "__main__":
    main()
