import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeRobot Joint States</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Arial, Helvetica, sans-serif;
      background: #181a1f;
      color: #f4f4f5;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: #181a1f;
    }

    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 700;
    }

    .status {
      min-width: 150px;
      padding: 8px 10px;
      border: 1px solid #343842;
      border-radius: 6px;
      text-align: center;
      color: #cbd5e1;
      background: #20232b;
    }

    .status.live {
      color: #b9f6ca;
      border-color: #2e6f45;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border: 1px solid #343842;
      border-radius: 8px;
      background: #20232b;
    }

    th,
    td {
      padding: 12px 14px;
      border-bottom: 1px solid #343842;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    th:first-child,
    td:first-child {
      text-align: left;
    }

    th {
      color: #aab3c2;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    .empty {
      padding: 28px;
      border: 1px solid #343842;
      border-radius: 8px;
      color: #aab3c2;
      background: #20232b;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>LeRobot Joint States</h1>
      <div id="status" class="status">waiting</div>
    </header>

    <div id="empty" class="empty">Waiting for /joint_states...</div>
    <table id="table" hidden>
      <thead>
        <tr>
          <th>Joint</th>
          <th>Position rad</th>
          <th>Position deg</th>
          <th>Velocity</th>
          <th>Effort</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>

  <script>
    const statusEl = document.getElementById("status");
    const emptyEl = document.getElementById("empty");
    const tableEl = document.getElementById("table");
    const rowsEl = document.getElementById("rows");

    function formatNumber(value, digits = 4) {
      return typeof value === "number" && Number.isFinite(value)
        ? value.toFixed(digits)
        : "";
    }

    const events = new EventSource("/events");

    events.onopen = () => {
      statusEl.textContent = "connected";
      statusEl.classList.add("live");
    };

    events.onerror = () => {
      statusEl.textContent = "reconnecting";
      statusEl.classList.remove("live");
    };

    events.onmessage = (event) => {
      const state = JSON.parse(event.data);
      statusEl.textContent = `live ${state.age_ms} ms`;
      statusEl.classList.add("live");
      emptyEl.hidden = true;
      tableEl.hidden = false;

      rowsEl.innerHTML = state.joints.map((joint) => `
        <tr>
          <td>${joint.name}</td>
          <td>${formatNumber(joint.position)}</td>
          <td>${formatNumber(joint.position_deg, 2)}</td>
          <td>${formatNumber(joint.velocity)}</td>
          <td>${formatNumber(joint.effort)}</td>
        </tr>
      `).join("");
    };
  </script>
</body>
</html>
"""


class JointStateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None

    def update(self, message: JointState):
        joints = []
        for index, name in enumerate(message.name):
            position = self._value_at(message.position, index)
            velocity = self._value_at(message.velocity, index)
            effort = self._value_at(message.effort, index)
            joints.append(
                {
                    "name": name,
                    "position": position,
                    "position_deg": math.degrees(position) if position is not None else None,
                    "velocity": velocity,
                    "effort": effort,
                }
            )

        with self._lock:
            self._latest = {
                "received_at": time.time(),
                "joints": joints,
            }

    def snapshot(self):
        with self._lock:
            if self._latest is None:
                return None

            snapshot = dict(self._latest)
            snapshot["age_ms"] = int((time.time() - snapshot["received_at"]) * 1000)
            return snapshot

    @staticmethod
    def _value_at(values, index):
        return float(values[index]) if index < len(values) else None


class JointStateNode(Node):
    def __init__(self, store: JointStateStore):
        super().__init__("joint_state_gui")
        self._store = store
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.get_logger().info("Listening to /joint_states")

    def _on_joint_state(self, message: JointState):
        self._store.update(message)


def make_request_handler(store: JointStateStore):
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._send_html()
            elif self.path == "/events":
                self._send_events()
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            return

        def _send_html(self):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            while True:
                snapshot = store.snapshot()
                if snapshot is not None:
                    payload = json.dumps(snapshot)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                time.sleep(0.1)

    return RequestHandler


def main():
    parser = argparse.ArgumentParser(description="Show /joint_states in a browser.")
    parser.add_argument("--port", type=int, default=3000, help="HTTP port for the web GUI.")
    args = parser.parse_args()

    rclpy.init()

    store = JointStateStore()
    node = JointStateNode(store)

    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), make_request_handler(store))
    except OSError as error:
        if error.errno == 98:
            node.get_logger().error(
                f"Port {args.port} is already in use. Stop the old GUI or use --port <port>."
            )
            node.destroy_node()
            rclpy.shutdown()
            return
        raise

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    node.get_logger().info(f"Joint-state GUI available on http://localhost:{args.port}")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
