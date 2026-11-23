"""MujocoDriver: the ArmDriver for `pixi run sim`.

The bridge node (moveit1_interface, airbot_play_ros_interface.py) loads it with
`~driver:=sim`. It implements the same methods as arm_driver.ArmDriver there; it does
not import that module, so the simulation does not depend on the bridge package.
"""

from __future__ import annotations

import itertools
import socket
import threading
import time

from sim import protocol


class DriverError(RuntimeError):
    pass


class MujocoDriver:
    name = "sim"

    def __init__(self, path: str | None = None, rate: float = 500.0, **_ignored):
        self.path = str(path or protocol.socket_path())
        self.rate = rate
        self._sock: socket.socket | None = None
        self._wlock = threading.Lock()
        self._state: dict | None = None
        self._state_event = threading.Event()
        self._replies: dict[int, dict] = {}
        self._reply_cond = threading.Condition()
        self._ids = itertools.count(1)
        self._alive = False
        self._info = {"name": "mujoco", "is_sim": True}

    # --- ArmDriver ----------------------------------------------------------------
    def connect(self, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        last = None
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(max(0.1, deadline - time.monotonic()))
                s.connect(self.path)
                break
            except OSError as e:
                last = e
                s.close()
                if time.monotonic() >= deadline:
                    raise DriverError(
                        f"连不上仿真进程（{self.path}：{last}）。请先在另一个终端运行：pixi run sim"
                    ) from None
                time.sleep(0.2)
        s.settimeout(None)
        self._sock = s
        self._alive = True
        threading.Thread(target=self._reader, daemon=True).start()
        hello = self._request("hello", timeout=max(0.5, deadline - time.monotonic()))
        if hello.get("version") != protocol.VERSION:
            self.disconnect()
            raise DriverError(f"仿真进程的协议版本是 {hello.get('version')}，驱动是 {protocol.VERSION}：请重启 pixi run sim")
        self._info = {"name": hello.get("name", "mujoco"), "is_sim": bool(hello.get("is_sim", True))}
        self._request("subscribe", rate=self.rate)
        if not self._state_event.wait(max(0.5, deadline - time.monotonic())):
            self.disconnect()
            raise DriverError("仿真进程没有发来状态")

    def disconnect(self) -> None:
        self._alive = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def info(self) -> dict:
        return dict(self._info)

    def ok(self) -> bool:
        return self._alive

    def enter_servo_mode(self) -> None:
        self._request("enter_servo_mode")

    def servo_joint_pos(self, q) -> None:
        self._send({"op": "servo_joint", "q": [float(v) for v in q]})

    def servo_eef_pos(self, width: float) -> None:
        self._send({"op": "servo_eef", "width": float(width)})

    def get_joint_state(self):
        st = self._latest()
        return st["q"], st["dq"], st["tau"]

    def get_eef_state(self):
        st = self._latest()
        return st["eef"], st["eef_v"], st["eef_eff"]

    # --- extra (sim-ctl, camera) ------------------------------------------------
    def request(self, op: str, timeout: float = 3.0, **fields) -> dict:
        return self._request(op, timeout=timeout, **fields)

    # --- internals --------------------------------------------------------------
    def _latest(self) -> dict:
        if not self._alive:
            raise DriverError("和仿真进程的连接已断开（pixi run sim 被关掉了？）")
        if self._state is None:
            raise DriverError("还没有收到仿真状态")
        return self._state

    def _send(self, msg: dict) -> None:
        if not self._alive or self._sock is None:
            raise DriverError("和仿真进程的连接已断开（pixi run sim 被关掉了？）")
        data = protocol.encode(msg)
        try:
            with self._wlock:
                self._sock.sendall(data)
        except OSError as e:
            self._alive = False
            raise DriverError(f"和仿真进程的连接已断开：{e}") from None

    def _request(self, op: str, timeout: float = 3.0, **fields) -> dict:
        rid = next(self._ids)
        self._send({"op": op, "id": rid, **fields})
        deadline = time.monotonic() + timeout
        with self._reply_cond:
            while rid not in self._replies:
                left = deadline - time.monotonic()
                if left <= 0 or not self._alive:
                    raise DriverError(f"仿真进程没有回应 {op}")
                self._reply_cond.wait(left)
            reply = self._replies.pop(rid)
        if not reply.get("ok"):
            raise DriverError(reply.get("error", f"{op} 失败"))
        return reply

    def _reader(self) -> None:
        buf = b""
        sock = self._sock
        try:
            while self._alive and sock is not None:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if not line:
                        continue
                    msg = protocol.decode(line)
                    if msg.get("type") == "state":
                        self._state = msg
                        self._state_event.set()
                    elif "id" in msg:
                        with self._reply_cond:
                            self._replies[msg["id"]] = msg
                            self._reply_cond.notify_all()
        except OSError:
            pass
        finally:
            self._alive = False
            with self._reply_cond:
                self._reply_cond.notify_all()


def probe(timeout: float = 1.0) -> bool:
    """Is a simulation process answering on the socket?"""
    d = MujocoDriver(rate=10)
    try:
        d.connect(timeout)
        return True
    except DriverError:
        return False
    finally:
        d.disconnect()
