"""A stand-in for airbot_server (the arm container's gRPC service), for testing AirbotPyDriver.

Commanded joint and gripper positions are fed back as the state right away. Start it in a
test with `server, state = serve(port)`, or by hand with `python tests/fake_airbot_server.py
[port]` and point the bridge node at it.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent import futures

import grpc
from airbot_proto import types_pb2
from airbot_proto.motion import arm_pb2, arm_pb2_grpc

NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_mapping_controller"]


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.q = [0.0] * 6
        self.eef = 0.0
        self.mode = None
        self.commands = 0
        self.stopped = threading.Event()


class _Feedback(arm_pb2_grpc.FeedbackServiceServicer):
    def __init__(self, state: State):
        self.s = state

    def GetState(self, request, context):
        while context.is_active() and not self.s.stopped.is_set():
            with self.s.lock:
                js = types_pb2.JointStates(
                    name=NAMES, position=self.s.q + [self.s.eef], velocity=[0.0] * 7, effort=[0.0] * 7
                )
            yield arm_pb2.GetStateResponse(request_id=request.request_id, component_names=["arm"], state=js)
            time.sleep(0.005)

    def GetMode(self, request, context):
        return arm_pb2.GetModeResponse(request_id=request.request_id, component_names=["arm"], modes=[self.s.mode or 0])


class _Control(arm_pb2_grpc.ControlServiceServicer):
    def __init__(self, state: State):
        self.s = state

    def SetMode(self, request, context):
        self.s.mode = request.modes[0]
        return arm_pb2.SetModeResponse(
            request_id=request.request_id, component_names=request.component_names, modes=request.modes
        )

    def SetState(self, request_iterator, context):
        for req in request_iterator:
            with self.s.lock:
                for comp, cmd in zip(req.component_names, req.commands):
                    pos = list(cmd.joint_states.position)
                    if comp == "arm" and len(pos) == 6:
                        self.s.q = pos
                    elif comp == "eef" and pos:
                        self.s.eef = pos[0]
                self.s.commands += 1
            yield arm_pb2.SetStateResponse(request_id=req.request_id, status=arm_pb2.Status.RUNNING)


def serve(port: int = 50051, host: str = "localhost"):
    """Start the fake service; returns (grpc server, State). Stop it with stop(server, state)."""
    state = State()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    arm_pb2_grpc.add_FeedbackServiceServicer_to_server(_Feedback(state), server)
    arm_pb2_grpc.add_ControlServiceServicer_to_server(_Control(state), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server, state


def stop(server, state: State) -> None:
    state.stopped.set()  # end the feedback streams
    server.stop(0)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    srv, st = serve(port)
    print(f"fake airbot_server listening on localhost:{port}", flush=True)
    try:
        while True:
            time.sleep(5)
            with st.lock:
                print(f"commands={st.commands} q={[round(x, 4) for x in st.q]} eef={st.eef:.4f}", flush=True)
    except KeyboardInterrupt:
        stop(srv, st)
