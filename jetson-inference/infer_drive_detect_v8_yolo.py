import os
import sys
import json
import time
import signal
import logging
import multiprocessing as mp

import cv2
import zmq
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("racer")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATA_PORT = 5555
DRIVING_ENGINE_PATH = "./models/resnet18_merged_v3.engine"
DETECTION_ENGINE_PATH = "./models/yolo11.engine"
SIGN_LABELS_PATH = "labels.txt"

CAMERA_W, CAMERA_H = 640, 480
MODEL_INPUT_SIZE = 224

CONF_THRES = 0.25
IOU_THRES = 0.45

MAX_THROTTLE = 0.375
THROTTLE_GAIN = 0.375
STEERING_GAIN = -0.65
STEERING_OFFSET = 0.20

DURATION_STOP = 2.0
DURATION_SLOW = 2.0
DURATION_SPEED_LIMIT = 7.0 

TRACKER_IOU_THRES = 0.3
TRACKER_EVICT_FRAMES = 90

FACTOR_CHILD = 0.95
FACTOR_ADULT = 1.0 
FACTOR_SPEED_LIMIT = 0.95

MIN_AREA_PERSON = 3000
MIN_AREA_SIGN = 4000

PERSON_CLASSES = {"child", "adult"}
SIGN_CLASSES = {
    "stop sign", "speed limit zone", "end speed limit zone",
    "turn right", "yield", "priority road", "general danger",
}

# ──────────────────────────────────────────────
# TensorRT Inference Engines
# ──────────────────────────────────────────────


class TRTInference:
    """Generic TensorRT inference wrapper for the driving model (ResNet)."""

    def __init__(self, engine_path: str):
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self._logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = (
            self._allocate_buffers()
        )

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for b in self.engine:
            size = trt.volume(self.engine.get_binding_shape(b))
            dtype = trt.nptype(self.engine.get_binding_dtype(b))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            entry = {
                "host": host_mem,
                "device": device_mem,
                "shape": self.engine.get_binding_shape(b),
            }
            if self.engine.binding_is_input(b):
                inputs.append(entry)
            else:
                outputs.append(entry)
        return inputs, outputs, bindings, stream

    def infer(self, img_np: np.ndarray) -> list:
        np.copyto(self.inputs[0]["host"], img_np.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        return [out["host"] for out in self.outputs]


class YoloTRT:
    """TensorRT inference wrapper for YOLOv8/v11 detection."""

    def __init__(self, engine_path: str):
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        log.info("Loading YOLO engine: %s", engine_path)
        with open(engine_path, "rb") as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)

            nptype = np.float16 if dtype == trt.float16 else np.float32
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, nptype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            binding = {
                "index": i,
                "name": name,
                "dtype": nptype,
                "shape": list(shape),
                "host": host_mem,
                "device": device_mem,
            }

            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
                self.input_h = shape[2]
                self.input_w = shape[3]
                log.info(
                    "YOLO input detected: expecting %dx%d", self.input_w, self.input_h
                )
            else:
                self.outputs.append(binding)

        self.stream = cuda.Stream()

    def _preprocess(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        target_h, target_w = self.input_h, self.input_w
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        image_resized = cv2.resize(img, (nw, nh))

        image_padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dw, dh = (target_w - nw) // 2, (target_h - nh) // 2
        image_padded[dh : nh + dh, dw : nw + dw, :] = image_resized

        blob = image_padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)
        return blob, scale, (dw, dh)

    def infer(self, img: np.ndarray) -> list:
        blob, scale, (dw, dh) = self._preprocess(img)
        np.copyto(self.inputs[0]["host"], blob.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        output = np.transpose(output[0], (1, 0))

        return self._postprocess(output, scale, (dw, dh))

    def _postprocess(self, prediction: np.ndarray, scale: float, pad: tuple) -> list:
        dw, dh = pad

        max_scores = np.max(prediction[:, 4:], axis=1)
        mask = max_scores > CONF_THRES
        prediction = prediction[mask]

        if len(prediction) == 0:
            return []

        xc, yc, w, h = (
            prediction[:, 0],
            prediction[:, 1],
            prediction[:, 2],
            prediction[:, 3],
        )
        cls_scores = prediction[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        confidences = np.max(cls_scores, axis=1)

        x1 = (xc - w / 2) - dw
        y1 = (yc - h / 2) - dh
        x2 = (xc + w / 2) - dw
        y2 = (yc + h / 2) - dh

        x1 /= scale
        y1 /= scale
        x2 /= scale
        y2 /= scale

        w_scaled = w / scale
        h_scaled = h / scale

        nms_boxes = np.stack([x1, y1, w_scaled, h_scaled], axis=1)
        result_boxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), confidences.tolist(), CONF_THRES, IOU_THRES
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(
                    {
                        "box": result_boxes[i],
                        "conf": confidences[i],
                        "class_id": cls_ids[i],
                    }
                )
        return results


# ──────────────────────────────────────────────
# Kalman Filter
# ──────────────────────────────────────────────


class KalmanSteering:
    def __init__(self, process_noise: float = 0.005, measurement_noise: float = 0.1):
        self.x = np.array([[0.0], [0.0]])
        self.P = np.eye(2)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.I = np.eye(2)
        self.R = np.array([[measurement_noise]])
        self.Q = np.array([[process_noise, 0.0], [0.0, process_noise]])

    def predict(self, dt: float):
        self.F[0, 1] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: float) -> float:
        z = np.array([[measurement]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        IKH = self.I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        return self.x[0, 0]


# ──────────────────────────────────────────────
# Detection Tracker
# ──────────────────────────────────────────────


class DetectionTracker:
    """IoU-based tracker that assigns persistent IDs to detections across frames."""

    def __init__(self, iou_threshold: float = 0.3, evict_frames: int = 90):
        self._iou_threshold = iou_threshold
        self._evict_frames = evict_frames
        self._next_id = 0
        self._tracks = {}

    @staticmethod
    def _iou(box_a: list, box_b: list) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix = max(ax, bx)
        iy = max(ay, by)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix) * max(0, iy2 - iy)
        if inter == 0:
            return 0.0

        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list) -> list:
        unmatched = list(range(len(detections)))
        matched_track_ids = set()

        for det_idx in list(unmatched):
            det = detections[det_idx]
            best_iou = 0.0
            best_tid = None

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                if track["class"] != det["class"]:
                    continue
                iou = self._iou(track["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self._iou_threshold:
                self._tracks[best_tid]["box"] = det["box"]
                self._tracks[best_tid]["missed"] = 0
                det["track_id"] = best_tid
                det["acted"] = self._tracks[best_tid]["acted"]
                matched_track_ids.add(best_tid)
                unmatched.remove(det_idx)

        for det_idx in unmatched:
            det = detections[det_idx]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "class": det["class"],
                "box": det["box"],
                "acted": False,
                "missed": 0,
            }
            det["track_id"] = tid
            det["acted"] = False

        for tid in list(self._tracks):
            if tid not in matched_track_ids:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self._evict_frames:
                    del self._tracks[tid]

        return detections

    def mark_acted(self, track_id: int):
        if track_id in self._tracks:
            self._tracks[track_id]["acted"] = True


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────


def preprocess_driving(image: np.ndarray) -> np.ndarray:
    crop_img = image[:, 80:560]
    resized = cv2.resize(crop_img, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_img = ((gray.astype(np.float32) / 255.0) - 0.5) / 0.5
    return np.ascontiguousarray(norm_img)


# ──────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────


def main_loop(stop_event: mp.Event):
    labels = []
    if os.path.exists(SIGN_LABELS_PATH):
        with open(SIGN_LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    else:
        log.warning("Labels file not found at %s", SIGN_LABELS_PATH)

    log.info("Loaded %d class labels", len(labels))

    ctx = zmq.Context()
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.bind(f"tcp://*:{DATA_PORT}")

    car = NvidiaRacecar()
    car.throttle_gain = THROTTLE_GAIN
    car.steering_gain = STEERING_GAIN
    car.steering_offset = STEERING_OFFSET

    log.info("Loading driving engine (ResNet)...")
    driving_ai = TRTInference(DRIVING_ENGINE_PATH)

    log.info("Loading detection engine (YOLOv11)...")
    detection_ai = YoloTRT(DETECTION_ENGINE_PATH)

    kf = KalmanSteering(0.05, 0.2)
    tracker = DetectionTracker(TRACKER_IOU_THRES, TRACKER_EVICT_FRAMES)
    camera = CSICamera(width=CAMERA_W, height=CAMERA_H, capture_fps=30)
    camera.running = True

    log.info("System ready — entering main drive loop")

    speed_limit_active = False
    speed_limit_end_time = 0.0
    incident_end_time = 0.0
    incident_type = None

    log.info("Waiting for camera feed...")
    while camera.value is None:
        time.sleep(0.01)

    last_time = time.perf_counter()

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            dt = loop_start - last_time
            last_time = loop_start
            current_time = time.time()

            frame = camera.value
            if frame is None:
                time.sleep(0.01)
                continue

            # Detection
            yolo_results = detection_ai.infer(frame)

            detections = []
            for item in yolo_results:
                cid = item["class_id"]
                label_name = labels[cid] if cid < len(labels) else f"ID_{cid}"
                x1, y1, x2, y2 = item["box"]
                bw, bh = int(x2 - x1), int(y2 - y1)
                area = bw * bh
                active = True
                if label_name in PERSON_CLASSES and area < MIN_AREA_PERSON:
                    active = False
                if label_name in SIGN_CLASSES and area < MIN_AREA_SIGN:
                    active = False
                detections.append(
                    {
                        "class": label_name,
                        "score": float(item["conf"]),
                        "box": [int(x1), int(y1), bw, bh],
                        "area": area,
                        "active": active,
                    }
                )

            seen_labels = [d["class"] for d in detections if d["active"]]
            tracker.update(detections)

            # State machine — speed limit zone
            if "speed limit zone" in seen_labels:
                speed_limit_active = True
                speed_limit_end_time = current_time + DURATION_SPEED_LIMIT
                log.info("Entering speed-limit zone (latched)")
            elif "end speed limit zone" in seen_labels:
                speed_limit_active = False
                log.info("Leaving speed-limit zone (latched)")

            if speed_limit_active and current_time > speed_limit_end_time:
                speed_limit_active = False
                log.info("Speed-limit zone expired (timeout)")

            # State machine — incident handling
            stop_dets = [
                d for d in detections
                if d["class"] in ("stop_sign", "stop sign")
                and d["active"]
                and not d.get("acted", False)
            ]
            if stop_dets:
                if incident_type != "STOP":
                    incident_type = "STOP"
                    incident_end_time = current_time + DURATION_STOP
                    for d in stop_dets:
                        tracker.mark_acted(d["track_id"])
                    log.info("STOP triggered for %.1fs (track IDs: %s)",
                             DURATION_STOP,
                             [d["track_id"] for d in stop_dets])
            elif "child" in seen_labels:
                if incident_type != "STOP":
                    incident_type = "CHILD"
                    incident_end_time = current_time + DURATION_SLOW
            elif "adult" in seen_labels:
                if incident_type not in ["STOP", "CHILD"]:
                    incident_type = "ADULT"
                    incident_end_time = current_time + DURATION_SLOW

            # Driving control
            drive_input = preprocess_driving(frame)
            drive_out = driving_ai.infer(drive_input)[0]
            raw_steer = float(drive_out[0])

            kf.predict(dt)
            smooth_steer = kf.update(raw_steer)
            car.steering = np.clip(smooth_steer, -1.0, 1.0)

            # Throttle
            target_throttle = MAX_THROTTLE
            if speed_limit_active:
                target_throttle *= FACTOR_SPEED_LIMIT

            is_incident_active = current_time < incident_end_time

            if is_incident_active:
                if incident_type == "STOP":
                    target_throttle = 0.0
                elif incident_type == "CHILD":
                    target_throttle *= FACTOR_CHILD
                    log.debug(
                        "Slowing (child) — %.1fs remaining",
                        incident_end_time - current_time,
                    )
                elif incident_type == "ADULT":
                    target_throttle *= FACTOR_ADULT
                    log.debug(
                        "Slowing (adult) — %.1fs remaining",
                        incident_end_time - current_time,
                    )
            else:
                incident_type = None

            car.throttle = target_throttle

            # Telemetry
            vis_frame = frame.copy()
            for d in detections:
                x, y, w, h = d["box"]
                color = (
                    (0, 0, 255)
                    if d["class"] in ["stop sign", "child"]
                    else (0, 255, 0)
                )
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    vis_frame,
                    f"{d['class']} {d['score']:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            ret, jpg = cv2.imencode(
                ".jpg", vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            )

            telemetry = {
                "raw_steer": raw_steer,
                "smooth_steer": smooth_steer,
                "throttle": car.throttle,
                "fps": 1.0 / (time.perf_counter() - loop_start),
                "mode": "LIMIT" if speed_limit_active else "NORMAL",
                "incident": incident_type if is_incident_active else "NONE",
                "detections": detections,
            }
            pub_socket.send_multipart(
                [b"dashboard", json.dumps(telemetry).encode("utf-8"), jpg.tobytes()]
            )

    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        camera.running = False
        car.throttle = 0.0
        car.steering = 0.0
        log.info("Cleanup complete — system stopped")


if __name__ == "__main__":
    stop_event = mp.Event()
    signal.signal(signal.SIGINT, lambda s, f: stop_event.set())
    main_loop(stop_event)
