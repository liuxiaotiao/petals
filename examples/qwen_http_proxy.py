"""A minimal CONNECT proxy, so nodes with no egress can reach the Hub through one that has it.

Petals servers download weights over HTTPS, and both requests and urllib honour HTTPS_PROXY,
so one node with egress is enough to serve the whole cluster without touching the firewall
or changing how blocks are assigned.

Only CONNECT is implemented. That is all an HTTPS client needs, and refusing everything else
keeps this from quietly becoming a general-purpose open relay. Clients are restricted to the
cluster's own addresses and destinations to the ports the Hub actually uses: an open proxy on
a lab network is a real liability, and here the client set is known exactly, so the allowlist
costs nothing.

  python examples/qwen_http_proxy.py --port 8899 --allow 192.168.1.2,192.168.1.3
"""
import argparse
import logging
import select
import socket
import socketserver

BUFFER = 65536
# A shard download that has gone quiet for this long is wedged; do not pin a thread on it.
IDLE_TIMEOUT = 300
CONNECT_TIMEOUT = 30

ALLOWED_CLIENTS = set()
ALLOWED_PORTS = {80, 443}


def relay(left, right):
    """Shuttle bytes both ways until either side closes or goes quiet."""
    sockets = [left, right]
    try:
        while True:
            readable, _, errored = select.select(sockets, [], sockets, IDLE_TIMEOUT)
            if errored or not readable:
                return
            for source in readable:
                data = source.recv(BUFFER)
                if not data:
                    return
                (right if source is left else left).sendall(data)
    except OSError:
        return
    finally:
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass


class Handler(socketserver.StreamRequestHandler):
    timeout = 60

    def deny(self, status, explanation):
        """Answer a refusal in full, so the client reports the reason and not a bare close."""
        body = explanation.encode()
        try:
            self.wfile.write(
                b"HTTP/1.1 %d Forbidden\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
                % (status, len(body), body)
            )
            self.wfile.flush()
        except OSError:
            pass

    def handle(self):
        client = self.client_address[0]
        if ALLOWED_CLIENTS and client not in ALLOWED_CLIENTS:
            logging.warning("refused %s: not in the client allowlist", client)
            self.deny(403, "client %s is not in this proxy's allowlist" % client)
            return
        try:
            line = self.rfile.readline(BUFFER).decode("latin-1").strip()
        except OSError:
            return
        while True:  # drain the request headers; CONNECT carries nothing we need
            try:
                more = self.rfile.readline(BUFFER)
            except OSError:
                return
            if not more or more in (b"\r\n", b"\n"):
                break

        parts = line.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            logging.warning("refused %r from %s: this proxy only speaks CONNECT", line[:80], client)
            return

        host, _, port_text = parts[1].rpartition(":")
        try:
            port = int(port_text)
        except ValueError:
            host, port = parts[1], 443
        if port not in ALLOWED_PORTS:
            logging.warning("refused %s -> %s:%s: port not in %s", client, host, port, sorted(ALLOWED_PORTS))
            self.deny(403, "port %s is not allowed; this proxy permits %s" % (port, sorted(ALLOWED_PORTS)))
            return

        try:
            upstream = socket.create_connection((host, port), timeout=CONNECT_TIMEOUT)
        except OSError as error:
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            logging.warning("%s -> %s:%d failed: %s", client, host, port, error)
            return

        self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        self.wfile.flush()
        logging.info("%s -> %s:%d", client, host, port)
        # The relay does its own waiting; the handler's read timeout would abort a long
        # transfer that is simply large rather than stuck.
        self.connection.settimeout(None)
        upstream.settimeout(None)
        relay(self.connection, upstream)


class Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    # 15 servers fetching shards in parallel open a lot of connections at once.
    request_queue_size = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--allow", default="", help="comma-separated client IPs; empty means any")
    parser.add_argument("--ports", default="80,443", help="comma-separated destination ports")
    args = parser.parse_args()

    ALLOWED_CLIENTS.update(item for item in args.allow.split(",") if item)
    ALLOWED_PORTS.clear()
    ALLOWED_PORTS.update(int(item) for item in args.ports.split(",") if item)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(
        "CONNECT proxy on %s:%d, %s, ports %s",
        args.host,
        args.port,
        "%d allowed client(s)" % len(ALLOWED_CLIENTS) if ALLOWED_CLIENTS else "ANY client",
        sorted(ALLOWED_PORTS),
    )
    Proxy((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
