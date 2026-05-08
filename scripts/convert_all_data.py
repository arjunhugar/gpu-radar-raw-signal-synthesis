"""
convert_all_data.py
===================

Standalone ROS1 bag writer for radar simulation output conversion.

This script converts:
- ADC binary frames
- vehicle odometry text output
- ground-truth object text output

into a ROS1 bag file.

Main features:
1. Reads binary ADC frames from a local folder
2. Reads odometry text output
3. Reads ground-truth object output
4. Serializes radar FFT, odometry, and reference objects
5. Writes a ROS1-compatible .bag file without requiring a full ROS installation

Expected local input structure:
- data/adc/
- data/odometry/Car_output.txt
- data/ground_truth/GroundTruth_output.txt

Output:
- out/<timestamp>_converted.bag

Important:
- Do not upload raw simulation files to GitHub.
- Do not upload generated .bag files to GitHub.
"""

import argparse
import collections
import csv
import datetime as dt
import math
import multiprocessing
import os
import struct
import time
import traceback
from typing import Optional

import numpy as np
from rosbags.rosbag1 import Writer

from virtual_validation.readers import OdometryDataReader
from virtual_validation.processing import RadarProcessing
from virtual_validation.data_types import RadarRawData


# =============================================================================
# Default local paths
# =============================================================================
DEFAULT_ADC_DIRECTORY = os.path.join("data", "adc")
DEFAULT_ODOMETRY_FILE = os.path.join("data", "odometry", "Car_output.txt")
DEFAULT_GROUND_TRUTH_FILE = os.path.join("data", "ground_truth", "GroundTruth_output.txt")
DEFAULT_OUTPUT_DIR = "out"
# =============================================================================


# ---------------------------------------------------------------------------
# Binary ADC reader
# ---------------------------------------------------------------------------

class BinAdcFrame:
    def __init__(self, frame_idx, mode, data):
        self.frame = frame_idx
        self.mode = mode
        self.data = data


class AnsysBinReader:
    MAGIC = b"ADC1"
    HEADER_SIZE = 32

    def __init__(self):
        self._files = []
        self._idx = 0
        self._file_time = None
        self.total_frames = 0

    @property
    def file_time(self):
        return self._file_time

    def open(self, directory):
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"ADC directory not found: {directory}")

        entries = []

        for name in os.listdir(directory):
            if not name.endswith("_adcsamples.bin"):
                continue

            stem = name[:-len("_adcsamples.bin")]
            parts = stem.split("_")

            if len(parts) < 2:
                continue

            try:
                frame = int(parts[-2])
                mode = int(parts[-1].replace("Mode", ""))
            except ValueError:
                continue

            entries.append((frame, mode, os.path.join(directory, name)))

        if not entries:
            return False

        entries.sort(key=lambda x: (x[0], x[1]))
        self._files = entries
        self._idx = 0
        self.total_frames = len(entries)
        self._file_time = dt.datetime.fromtimestamp(os.path.getmtime(entries[0][2]))

        return True

    def close(self):
        self._files = []
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._files):
            raise StopIteration

        frame_idx, mode, path = self._files[self._idx]
        self._idx += 1

        return self._read_bin(frame_idx, mode, path)

    def _read_bin(self, frame_idx, mode, path):
        with open(path, "rb") as f:
            hdr = f.read(self.HEADER_SIZE)

            if len(hdr) < self.HEADER_SIZE:
                raise IOError(f"ADC file too small: {path}")

            if hdr[:4] != self.MAGIC:
                raise ValueError(f"Bad ADC magic header: {path}")

            _v, _ntx, nrx, nc, ns, _d, _r = struct.unpack_from("<7i", hdr, 4)

            n = nrx * nc * ns
            raw = np.frombuffer(f.read(n * 4), dtype=np.float32)

            if len(raw) != n:
                raise IOError(f"Truncated ADC file: {path}")

        data = np.moveaxis(raw.reshape(nrx, nc, ns), 1, 2)
        return BinAdcFrame(frame_idx, mode, data)


# ---------------------------------------------------------------------------
# Ground Truth reader
# ---------------------------------------------------------------------------

class GTObject:
    def __init__(
        self,
        timestamp_s,
        obj_id,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        yaw,
        length,
        width,
        height,
    ):
        self.timestamp = dt.timedelta(seconds=timestamp_s)
        self.object_id = obj_id

        self.x_m = x
        self.y_m = y
        self.z_m = z

        self.velocity_x_mps = vx
        self.velocity_y_mps = vy
        self.velocity_z_mps = vz

        self.acceleration_x_mpss = float("nan")
        self.acceleration_y_mpss = float("nan")
        self.acceleration_z_mpss = float("nan")

        self.yaw_rad = yaw
        self.yaw_rate_radps = float("nan")

        self.length_m = length
        self.width_m = width
        self.height_m = height


class GroundTruthReader:
    def __init__(self):
        self._rows = []
        self._cursor = 0
        self._next_obj = None
        self._file_time = None

    @property
    def file_time(self):
        return self._file_time

    def open(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Ground-truth file not found: {filename}")

        self._file_time = dt.datetime.fromtimestamp(os.path.getmtime(filename))
        self._rows = []
        self._cursor = 0
        self._next_obj = None

        with open(filename, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=",", skipinitialspace=True):
                self._rows.append(row)

        print(f"[GT] Loaded {len(self._rows)} rows from {os.path.basename(filename)}")

    def close(self):
        self._rows = []
        self._cursor = 0
        self._next_obj = None

    def _parse(self, row):
        return GTObject(
            timestamp_s=float(row["TimeStamp[s]"]),
            obj_id=int(row["ObjID[-]"]),
            x=float(row["Pos_x[m]"]),
            y=float(row["Pos_y[m]"]),
            z=float(row["Pos_z[m]"]),
            vx=float(row["Vel_x[m/s]"]),
            vy=float(row["Vel_y[m/s]"]),
            vz=float(row["Vel_z[m/s]"]),
            yaw=float(row["rot_rz[rad]"]),
            length=float(row["Dim_l[m]"]),
            width=float(row["Dim_w[m]"]),
            height=float(row["Dim_h[m]"]),
        )

    def __iter__(self):
        return self

    def __next__(self):
        current = []

        if self._next_obj is not None:
            current.append(self._next_obj)
            self._next_obj = None

        while self._cursor < len(self._rows):
            obj = self._parse(self._rows[self._cursor])
            self._cursor += 1

            if not current or current[0].timestamp == obj.timestamp:
                current.append(obj)
            else:
                self._next_obj = obj
                break

        if not current:
            raise StopIteration

        return current


# ---------------------------------------------------------------------------
# ROS1 primitive serialization helpers
# ---------------------------------------------------------------------------

def _i8(v):
    return struct.pack("<b", int(v))


def _u8(v):
    return struct.pack("<B", int(v) & 0xFF)


def _i16(v):
    return struct.pack("<h", int(v))


def _u16(v):
    return struct.pack("<H", int(v) & 0xFFFF)


def _i32(v):
    return struct.pack("<i", int(v))


def _u32(v):
    return struct.pack("<I", int(v) & 0xFFFFFFFF)


def _i64(v):
    try:
        return struct.pack("<q", int(v))
    except Exception:
        return struct.pack("<q", 0)


def _f32(v):
    return struct.pack("<f", float(v))


def _f64(v):
    return struct.pack("<d", float(v))


def _bool(v):
    return struct.pack("<B", 1 if v else 0)


def _str(s):
    b = s.encode("utf-8")
    return _u32(len(b)) + b


def _ros_time(sec, nsec):
    return struct.pack("<II", int(sec), int(nsec))


def _header(seq, sec, nsec, frame_id):
    return _u32(seq) + _ros_time(sec, nsec) + _str(frame_id)


def _dt_to_sec_nsec(ts):
    unix = ts.timestamp()
    sec = int(unix)
    return sec, int((unix - sec) * 1e9)


def _dt_to_ns(ts):
    return int(ts.timestamp() * 1e9)


def _nan_f64():
    return _f64(float("nan"))


# ---------------------------------------------------------------------------
# CompressedFFT serialization
# ---------------------------------------------------------------------------

def _int8_to_bytes(values):
    if isinstance(values, (bytes, bytearray, memoryview)):
        return bytes(values)

    out = bytearray()
    for v in values:
        out.append(int(v) & 0xFF)

    return bytes(out)


def serialise_compressed_fft(compressed_data, base_time, sequence, frame_id):
    timestamp_us = int(compressed_data.timestamp_us)

    ts = base_time + dt.timedelta(microseconds=timestamp_us)
    sec, nsec = _dt_to_sec_nsec(ts)
    ts_ns = _dt_to_ns(ts)

    d = _header(int(sequence), sec, nsec, frame_id)

    d += _u32(int(compressed_data.cycle_number))
    d += _u32(int(compressed_data.beam_number))
    d += _i64(timestamp_us)
    d += _u32(int(compressed_data.chirp_to_chirp_distance_us))
    d += _u32(int(compressed_data.bandwidth_khz))
    d += _u32(int(compressed_data.carrier_frequency_khz))
    d += _u32(int(compressed_data.num_range_bins))
    d += _u32(int(compressed_data.num_chirps))
    d += _u32(int(compressed_data.num_samples_per_chirp))
    d += _u32(int(compressed_data.rx_flags))

    fft_bytes = _int8_to_bytes(compressed_data.compressed_fft)

    d += _u32(0)
    d += _u32(2)

    d += _str("doppler_bins")
    d += _u32(320)
    d += _u32(448 * 320 * 8)

    d += _str("range_bins")
    d += _u32(448)
    d += _u32(448 * 8)

    d += _u32(len(fft_bytes)) + fft_bytes

    d += _i64(getattr(compressed_data, "udp_timestamp", 0))

    d += _u32(0)
    d += _u32(1)

    d += _str("pckg_rx_states")
    d += _u32(0)
    d += _u32(0)

    d += _u32(0)
    d += _u32(3)
    d += _u32(1200)

    return ts_ns, d


# ---------------------------------------------------------------------------
# ODOCInput serialization
# ---------------------------------------------------------------------------

_odo_prev_x: Optional[float] = None
_odo_prev_t: Optional[float] = None


def serialise_odometry(odometry_data, base_time, sequence):
    global _odo_prev_x, _odo_prev_t

    ts = base_time + odometry_data.timestamp
    sec, nsec = _dt_to_sec_nsec(ts)
    ts_ns = _dt_to_ns(ts)

    d = _header(sequence, sec, nsec, "ego")

    vx = odometry_data.velocity_x_mps
    vy = odometry_data.velocity_y_mps
    vel_mps = math.sqrt(vx**2 + vy**2)

    cur_t = odometry_data.timestamp.total_seconds()
    cur_x = odometry_data.x_m

    if vel_mps < 0.01 and _odo_prev_x is not None and _odo_prev_t is not None:
        dt_s = cur_t - _odo_prev_t
        if dt_s > 1e-6:
            vel_from_pos = abs(cur_x - _odo_prev_x) / dt_s
            if vel_from_pos > vel_mps:
                vel_mps = vel_from_pos

    _odo_prev_x = cur_x
    _odo_prev_t = cur_t

    vel_kmh100 = int(round(100.0 * vel_mps / 3.6))

    acc_x_mm = (
        0
        if math.isnan(odometry_data.acceleration_x_mpss)
        else int(round(odometry_data.acceleration_x_mpss * 1000.0))
    )

    acc_y_mm = (
        0
        if math.isnan(odometry_data.acceleration_y_mpss)
        else int(round(odometry_data.acceleration_y_mpss * 1000.0))
    )

    yaw_centideg = int(round(odometry_data.yaw_rate_radps * 5729.578))
    gear = max(1, int(odometry_data.gear_position))

    wheel_rpm = int(round(vel_mps / (2 * math.pi * 0.316) * 60.0))

    d += _i32(0)
    d += _u32(0)
    d += _bool(False)

    d += _i16(0)
    d += _u32(0)
    d += _bool(False)

    d += _i16(0)
    d += _u32(0)
    d += _bool(False)

    d += _i16(yaw_centideg)
    d += _u32(0)
    d += _bool(True)

    d += _i16(vel_kmh100)
    d += _u32(0)
    d += _bool(True)

    d += _i8(20)
    d += _bool(True)
    d += _u8(gear)

    d += _i16(acc_x_mm)
    d += _u32(0)
    d += _bool(True)

    d += _i16(acc_y_mm)
    d += _u32(0)
    d += _bool(True)

    d += _i16(0)
    d += _u32(0)
    d += _bool(False)

    d += _bool(False)
    d += _bool(False)

    for _ in range(4):
        d += _u32(0)
        d += _i8(1)
        d += _i16(wheel_rpm)
        d += _bool(True)
        d += _bool(True)
        d += _bool(False)

    d += _u32(0)
    d += _u32(0)
    d += _u32(0)

    assert len(d) - 19 == 115, f"HostOdometry payload = {len(d) - 19}, expected 115"

    return ts_ns, d


# ---------------------------------------------------------------------------
# ReferenceObjects serialization
# ---------------------------------------------------------------------------

def serialise_reference_objects(gt_list, base_time, sequence):
    if not gt_list:
        raise ValueError("Empty ground-truth object list")

    ts = base_time + gt_list[0].timestamp
    sec, nsec = _dt_to_sec_nsec(ts)
    ts_ns = _dt_to_ns(ts)

    d = _header(sequence, sec, nsec, "ego")

    d += _u32(sequence)
    d += _u8(3)
    d += _u32(len(gt_list))

    for obj in gt_list:
        d += _u32(obj.object_id)

        d += _f64(obj.x_m)
        d += _nan_f64()

        d += _f64(obj.y_m)
        d += _nan_f64()

        d += _f64(obj.z_m)
        d += _nan_f64()

        d += _f64(obj.velocity_x_mps)
        d += _nan_f64()

        d += _f64(obj.velocity_y_mps)
        d += _nan_f64()

        d += _f64(obj.velocity_z_mps)
        d += _nan_f64()

        d += _f64(obj.acceleration_x_mpss)
        d += _f64(obj.acceleration_y_mpss)
        d += _f64(obj.acceleration_z_mpss)

        d += _u32(5)

        d += _f64(obj.length_m)
        d += _f64(obj.width_m)
        d += _f64(obj.height_m)

        d += _f64(obj.yaw_rad)
        d += _nan_f64()

        d += _f64(obj.yaw_rate_radps)

        d += _nan_f64()
        d += _nan_f64()
        d += _f64(1.0)

        d += _u32(0xFFFFFFFF)

    return ts_ns, d


# ---------------------------------------------------------------------------
# ADC worker
# ---------------------------------------------------------------------------

def _adc_worker(adc_frame, cycle_number, current_timestamp, file_time, sequence, sensor_name):
    try:
        t0 = time.time()

        cycle_number = int(cycle_number)
        current_timestamp = int(current_timestamp)
        sequence = int(sequence)

        data = adc_frame.data.astype(np.float64)
        max_val = np.abs(data).max()

        if max_val > 1e-10:
            data = data * (20000.0 / (max_val * 1e8))

        scaled = BinAdcFrame(
            int(adc_frame.frame),
            int(adc_frame.mode),
            data.astype(np.float32),
        )

        radar_raw = RadarRawData(scaled.frame, scaled.mode, scaled.data)

        compressed = RadarProcessing.process_data(
            radar_raw,
            cycle_number,
            0,
            current_timestamp,
        )

        elapsed = time.time() - t0

        ts_ns, raw = serialise_compressed_fft(
            compressed,
            file_time,
            sequence,
            sensor_name,
        )

        return ts_ns, raw, elapsed

    except Exception as e:
        print(f"[WARN] ADC worker failed seq={sequence}: {e}")
        print(traceback.format_exc())
        return None, None, 0.0


# ---------------------------------------------------------------------------
# Message definitions
# ---------------------------------------------------------------------------

FFT_MSGDEF = (
    "std_msgs/Header header\n"
    "uint32 cycle_number\nuint32 beam_number\nint64 timestamp\n"
    "uint32 chirp_to_chirp_distance\nuint32 bandwidth\nuint32 carrier_freq_khz\n"
    "uint32 num_range_bins\nuint32 num_chirps\nuint32 num_samples_per_chirp\n"
    "uint32 rx_flags\nstd_msgs/Int8MultiArray compressed_fft\nint64 udp_timestamp\n"
    "std_msgs/Int32MultiArray pckg_rx_states\nuint32 num_pckgs_per_chirp\nuint32 std_pckg_size\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
    "================================================================================\n"
    "MSG: std_msgs/Int8MultiArray\nstd_msgs/MultiArrayLayout layout\nint8[] data\n"
    "================================================================================\n"
    "MSG: std_msgs/MultiArrayLayout\nstd_msgs/MultiArrayDimension[] dim\nuint32 data_offset\n"
    "================================================================================\n"
    "MSG: std_msgs/MultiArrayDimension\nstring label\nuint32 size\nuint32 stride\n"
    "================================================================================\n"
    "MSG: std_msgs/Int32MultiArray\nstd_msgs/MultiArrayLayout layout\nint32[] data\n"
)

ODO_MSGDEF = (
    "std_msgs/Header header\nHostOdometry odometry\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
    "================================================================================\n"
    "MSG: rad_gen_messages/HostOdometry\n"
    "int32 steeringWheelAngle\nuint32 steeringWheelAngleTimestamp\nbool steeringWheelAngleValid\n"
    "int16 rearAxleAngle\nuint32 rearAxleAngleTimestamp\nbool rearAxleAngleValid\n"
    "int16 steeringMoment\nuint32 steeringMomentTimestamp\nbool SteeringMomentValid\n"
    "int16 yawRate\nuint32 yawRateTimestamp_2us\nbool yawValid\n"
    "int16 velocity\nuint32 velocity_2us\nbool velocityValid\n"
    "int8 airTemperature\nbool airTemperatureValid\nuint8 gearLeverPosition\n"
    "int16 AccLon\nuint32 AccLonTimestamp\nbool AccLonValid\n"
    "int16 AccLat\nuint32 AccLatTimestamp\nbool AccLatValid\n"
    "int16 AccZ\nuint32 AccZTimestamp\nbool AccZValid\n"
    "bool isActiveABS\nbool isActiveABSValid\n"
    "uint32 wheelTickFL\nint8 wheelDirFL\nint16 wheelSpeedFL\nbool wheelDirValidFL\nbool wheelSpeedValidFL\nbool wheelTickValidFL\n"
    "uint32 wheelTickFR\nint8 wheelDirFR\nint16 wheelSpeedFR\nbool wheelDirValidFR\nbool wheelSpeedValidFR\nbool wheelTickValidFR\n"
    "uint32 wheelTickRL\nint8 wheelDirRL\nint16 wheelSpeedRL\nbool wheelDirValidRL\nbool wheelSpeedValidRL\nbool wheelTickValidRL\n"
    "uint32 wheelTickRR\nint8 wheelDirRR\nint16 wheelSpeedRR_rpm\nbool wheelDirValidRR\nbool wheelSpeedValidRR\nbool wheelTickValidRR\n"
    "uint32 wheelDirTimestamp\nuint32 wheelSpeedTimestamp\nuint32 wheelTickTimestamp\n"
)

REF_MSGDEF = (
    "std_msgs/Header header\nuint32 measurement_id\n"
    "uint8 UNKNOWN=0\nuint8 LASER=1\nuint8 RADAR=2\nuint8 SYNTHETIC=3\nuint8 DGPS=4\n"
    "uint8 sensor_type\nReferenceObject[] objects\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
    "================================================================================\n"
    "MSG: rad_gen_messages/ReferenceObject\n"
    "uint32 id\n"
    "float64 x_m\nfloat64 x_m_variance\nfloat64 y_m\nfloat64 y_m_variance\n"
    "float64 z_m\nfloat64 z_m_variance\nfloat64 vx_m_s\nfloat64 vx_m_s_variance\n"
    "float64 vy_m_s\nfloat64 vy_m_s_variance\nfloat64 vz_m_s\nfloat64 vz_m_s_variance\n"
    "float64 ax_m_s_s\nfloat64 ay_m_s_s\nfloat64 az_m_s_s\n"
    "uint32 UNCLASSIFIED=0\nuint32 UNKNOWN_SMALL=1\nuint32 UNKNOWN_BIG=2\n"
    "uint32 PEDESTRIAN=3\nuint32 BIKE=4\nuint32 CAR=5\nuint32 TRUCK=6\n"
    "uint32 OVERDRIVABLE=10\nuint32 UNDERDRIVABLE=12\nuint32 MOTORBIKE=15\n"
    "uint32 INFRASTRUCTURE=16\nuint32 BICYCLE=17\n"
    "uint32 classification\n"
    "float64 length_m\nfloat64 width_m\nfloat64 height_m\n"
    "float64 orientation_rad\nfloat64 orientation_variance\nfloat64 yaw_rate_rad_s\n"
    "float64 rcs\nfloat64 prediction_age_s\nfloat64 existence_probability\n"
    "uint32 MOTION_TYPE_STATIC=0\nuint32 MOTION_TYPE_DYNAMIC=1\n"
    "uint32 MOTION_TYPE_STOPPED=2\nuint32 MOTION_TYPE_UNKNOWN=4294967295\n"
    "uint32 motion_type\n"
)

FFT_MD5 = "a" * 32
ODO_MD5 = "b" * 32
REF_MD5 = "c" * 32


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_all_data(output_filename, adc_directory, ground_truth_filename, odometry_filename):
    global _odo_prev_x, _odo_prev_t

    _odo_prev_x = None
    _odo_prev_t = None

    print(f"[INFO] ADC directory : {adc_directory}")
    print(f"[INFO] Odometry file : {odometry_filename}")
    print(f"[INFO] Ground truth  : {ground_truth_filename}")
    print(f"[INFO] Output bag    : {output_filename}")
    print()

    adc_reader = AnsysBinReader()

    if not adc_reader.open(adc_directory):
        raise RuntimeError(f"No ADC .bin files found in: {adc_directory}")

    print(f"[INFO] ADC frames    : {adc_reader.total_frames}")

    odo_reader = OdometryDataReader()
    odo_reader.open(odometry_filename)
    print("[INFO] Odometry reader: OK")

    gt_reader = GroundTruthReader()
    gt_reader.open(ground_truth_filename)
    print("[INFO] GT reader      : OK")
    print()

    first_ts = None
    cycle_number = 0

    fft_seq = 0
    odo_seq = 0
    ref_seq = 0

    all_cnt = 0
    loss_cnt = 0
    fft_success = 0
    fft_failed = 0
    comp_time = 0.0

    fft_ts = None
    fft_raw = None

    odo_ts = None
    odo_raw = None

    ref_ts = None
    ref_raw = None

    futures = collections.deque()
    max_q = max(1, multiprocessing.cpu_count() * 2)

    FFT_TOPIC = "/Radar_FL/HeadUnit/CompressedFFT"
    ODO_TOPIC = "/OdocInput"
    REF_TOPIC = "/ReferenceObjects"

    with Writer(output_filename) as writer, multiprocessing.Pool() as pool:
        fft_conn = writer.add_connection(
            FFT_TOPIC,
            "rad_gen_messages/msg/CompressedFFT",
            msgdef=FFT_MSGDEF,
            md5sum=FFT_MD5,
        )

        odo_conn = writer.add_connection(
            ODO_TOPIC,
            "rad_gen_messages/msg/ODOCInput",
            msgdef=ODO_MSGDEF,
            md5sum=ODO_MD5,
        )

        ref_conn = writer.add_connection(
            REF_TOPIC,
            "rad_gen_messages/msg/ReferenceObjects",
            msgdef=REF_MSGDEF,
            md5sum=REF_MD5,
        )

        adc_done = False

        while True:
            while len(futures) < max_q and not adc_done:
                try:
                    frame = next(adc_reader)

                    cur_ts = (frame.frame + 50 * frame.mode) * 1000

                    if first_ts is None:
                        first_ts = cur_ts
                        cycle_number = frame.mode

                    cur_ts -= first_ts

                    fut = pool.apply_async(
                        _adc_worker,
                        (
                            frame,
                            cycle_number,
                            cur_ts,
                            adc_reader.file_time,
                            fft_seq,
                            "radar_sensor_001",
                        ),
                    )

                    futures.append(fut)

                    fft_seq += 1
                    cycle_number += 1

                except StopIteration:
                    adc_done = True
                    break

            if fft_raw is None and futures and futures[0].ready():
                fft_ts, fft_raw, elapsed = futures.popleft().get()

                if fft_raw:
                    fft_success += 1
                    comp_time += elapsed
                    print(f"  FFT seq={fft_seq - len(futures) - 1} ({elapsed:.3f}s)")
                else:
                    fft_failed += 1
                    loss_cnt += 1

            if odo_raw is None:
                nxt = next(odo_reader, None)

                if nxt is not None:
                    odo_ts, odo_raw = serialise_odometry(
                        nxt,
                        adc_reader.file_time,
                        odo_seq,
                    )
                    odo_seq += 1

            if ref_raw is None:
                try:
                    nxt = next(gt_reader)

                    if nxt:
                        ref_ts, ref_raw = serialise_reference_objects(
                            nxt,
                            adc_reader.file_time,
                            ref_seq,
                        )
                        ref_seq += 1

                except StopIteration:
                    pass

            if not futures and fft_raw is None and odo_raw is None and ref_raw is None:
                break

            candidates = [
                (t, tag)
                for t, tag in [
                    (fft_ts, "fft"),
                    (odo_ts, "odo"),
                    (ref_ts, "ref"),
                ]
                if t is not None
            ]

            if not candidates:
                time.sleep(0.01)
                continue

            _, tag = min(candidates, key=lambda x: x[0])
            all_cnt += 1

            if tag == "fft":
                writer.write(fft_conn, fft_ts, fft_raw)
                print(f"  BAG <- FFT  ts_ns={fft_ts}")
                fft_ts = None
                fft_raw = None

            elif tag == "odo":
                writer.write(odo_conn, odo_ts, odo_raw)
                print(f"  BAG <- ODO  ts_ns={odo_ts}")
                odo_ts = None
                odo_raw = None

            elif tag == "ref":
                writer.write(ref_conn, ref_ts, ref_raw)
                print(f"  BAG <- REF  ts_ns={ref_ts}")
                ref_ts = None
                ref_raw = None

    print()

    if all_cnt > 0:
        print(f"[INFO] Loss ratio        : {loss_cnt / all_cnt:.4f}")

    print(f"[INFO] FFT success       : {fft_success}")
    print(f"[INFO] FFT failed        : {fft_failed}")

    if fft_success + fft_failed > 0:
        print(f"[INFO] FFT loss ratio    : {fft_failed / (fft_success + fft_failed):.4f}")

    print(f"[INFO] Total compression : {comp_time:.2f} s")
    print(f"[INFO] Bag written       : {output_filename}")

    adc_reader.close()
    odo_reader.close()
    gt_reader.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert radar ADC, odometry, and ground-truth files into a ROS1 bag."
    )

    parser.add_argument(
        "--adc-dir",
        default=DEFAULT_ADC_DIRECTORY,
        help="Directory containing *_adcsamples.bin files.",
    )

    parser.add_argument(
        "--odometry",
        default=DEFAULT_ODOMETRY_FILE,
        help="Path to Car_output.txt.",
    )

    parser.add_argument(
        "--ground-truth",
        default=DEFAULT_GROUND_TRUTH_FILE,
        help="Path to GroundTruth_output.txt.",
    )

    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated .bag file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    now = dt.datetime.now()
    bag_path = os.path.join(
        args.out_dir,
        now.strftime("%Y-%m-%d_%H-%M-%S") + "_converted.bag",
    )

    t0 = time.time()

    convert_all_data(
        output_filename=bag_path,
        adc_directory=args.adc_dir,
        ground_truth_filename=args.ground_truth,
        odometry_filename=args.odometry,
    )

    print(f"\n[DONE] Elapsed : {time.time() - t0:.1f} s")
    print(f"[DONE] Bag     : {bag_path}")
