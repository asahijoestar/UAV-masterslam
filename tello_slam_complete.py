#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tello_slam_complete.py (robust)
- TelloのH.264(UDP)を低遅延で受信し、MASt3R-SLAMに直結する完成版（PyAV例外耐性強化）
"""

import os, sys, time, socket, threading, argparse, subprocess, yaml
import numpy as np
import cv2
import torch, lietorch
import torch.multiprocessing as mp
from contextlib import nullcontext

# ========= MASt3R-SLAM imports =========
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.visualization import WindowMsg, run_visualization

# ========= Low-latency options for OpenCV/FFmpeg =========
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "protocol_whitelist;file,udp,rtp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|reorder_queue_size;0"
    "|max_delay;0"
    "|min_delay;1"
    "|use_wallclock_as_timestamps;1"
    "|avioflags;direct"
    "|flush_packets;1"
    "|probesize;32"
    "|analyzeduration;0"
)

TELLO_ADDR = ("192.168.10.1", 8889)

# ========= Utilities =========
def get_local_ip_on_192_168_10() -> str | None:
    try:
        out = subprocess.check_output(
            "ip -o -4 addr show | awk '{print $4}'",
            shell=True
        ).decode()
        for cidr in out.split():
            ip = cidr.split('/')[0]
            if ip.startswith("192.168.10."):
                return ip
    except Exception:
        pass
    return None

def reset_stream():
    """Iフレーム誘発: command→streamoff→streamon"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        for cmd, wait in ((b"command",0.2),(b"streamoff",0.4),(b"streamon",0.6)):
            s.sendto(cmd, TELLO_ADDR); time.sleep(wait)
        s.close()
    except Exception:
        pass

# ========= Receivers =========
class _PyAVReceiver:
    """
    PyAVでUDP生H.264を受信し、デコード済みBGRの最新1枚のみ提供
    - av.error.* を正しく捕捉し、InvalidData等は無視して再同期
    - 入力停止時はコンテナを再オープン
    """
    def __init__(self, local_ip: str, warmup_sec: float = 3.0, reset_on_stall: bool = True):
        import av
        from av import error as av_error
        self.av = av
        self.av_error = av_error

        self.options = {
            "fflags":"nobuffer","flags":"low_delay",
            "probesize":"32","analyzeduration":"0",
            "max_delay":"0","reorder_queue_size":"0",
            "avioflags":"direct","use_wallclock_as_timestamps":"1",
        }
        # 複数候補URL
        self.urls = [
            f"udp://0.0.0.0:11111?localaddr={local_ip}",
            "udp://@0.0.0.0:11111",
            "udp://0.0.0.0:11111",
        ]
        self._lock = threading.Lock()
        self._latest = None  # (ts, bgr ndarray)
        self._running = True
        self._reset_on_stall = reset_on_stall

        # 最初のオープン（format指定あり/なしの順に試す）
        self._open_any()

        # スレッド起動
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

        # Warmup: 実フレームが溜まるまで待機（最大 warmup_sec）
        t0 = time.time()
        while time.time() - t0 < warmup_sec:
            with self._lock:
                if self._latest is not None:
                    break
            time.sleep(0.02)

    def _open_any(self):
        self.container = None
        self.stream = None
        last_exc = None
        for url in self.urls:
            # 1) autodetect
            try:
                self.container = self.av.open(url, options=self.options)
                self.stream = next((s for s in self.container.streams if s.type=="video"), None)
                if self.stream:
                    # 成功
                    return
                else:
                    try: self.container.close()
                    except: pass
                    self.container = None
            except Exception as e:
                last_exc = e
            # 2) format="h264"
            try:
                self.container = self.av.open(url, format="h264", options=self.options)
                self.stream = next((s for s in self.container.streams if s.type=="video"), None)
                if self.stream:
                    return
                else:
                    try: self.container.close()
                    except: pass
                    self.container = None
            except Exception as e:
                last_exc = e
        raise RuntimeError(f"PyAVでTelloストリームを開けません: {last_exc}")

    def _reopen(self):
        # コンテナを閉じて再オープン
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass
        # たまにIフレーム誘発が必要
        if self._reset_on_stall:
            reset_stream()
        time.sleep(0.2)
        self._open_any()

    def _loop(self):
        stall_t0 = time.time()
        while self._running:
            try:
                for pkt in self.container.demux(self.stream):
                    if not self._running:
                        break
                    try:
                        frames = pkt.decode()
                    except (self.av_error.FFmpegError, self.av_error.InvalidDataError, Exception):
                        # 壊れたパケット（no frame! など）は捨てて次へ
                        continue
                    got_frame = False
                    for frm in frames:
                        img_bgr = frm.to_ndarray(format="bgr24")
                        ts = time.time()
                        with self._lock:
                            self._latest = (ts, img_bgr)
                        got_frame = True
                    if got_frame:
                        stall_t0 = time.time()

                # デマックスが途切れて一定時間無音なら再オープン
                if time.time() - stall_t0 > 3.0:
                    self._reopen()
                    stall_t0 = time.time()

            except (self.av_error.FFmpegError, self.av_error.InvalidDataError, OSError, Exception):
                # デコード/IOエラーは再オープンで復帰
                try: time.sleep(0.05)
                except: pass
                self._reopen()
                stall_t0 = time.time()

    def read(self):
        with self._lock:
            return self._latest  # (ts, bgr) or None

    def release(self):
        self._running = False
        try: self._t.join(timeout=0.5)
        except: pass
        try: self.container.close()
        except: pass

class _OpenCVReceiver:
    """OpenCV/FFmpegで受信。SPS/PPS次第で不安定な環境あり。"""
    def __init__(self, warmup_sec: float = 3.0):
        url = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000&reuse=1"
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            raise RuntimeError("OpenCVでTello UDPストリームを開けません。")

        t0 = time.time(); ok=False; fr=None
        while time.time() - t0 < warmup_sec:
            r, f = self.cap.read()
            if r and f is not None and f.size > 0:
                ok, fr = True, f
                break
            time.sleep(0.01)
        if not ok:
            raise RuntimeError("OpenCVで有効なフレームが来ません。")

        self._latest = (time.time(), fr)
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            r, f = self.cap.read()
            if not r or f is None or f.size == 0:
                time.sleep(0.001); continue
            ts = time.time()
            with self._lock:
                self._latest = (ts, f)

    def read(self):
        with self._lock:
            return self._latest

    def release(self):
        self._running = False
        try: self._t.join(timeout=0.5)
        except: pass
        try: self.cap.release()
        except: pass

class TelloSource:
    """PyAV優先→OpenCVフォールバックの複合受信クラス（最新フレームのみ）"""
    def __init__(self, warmup_sec=3.0, do_reset=False):
        if do_reset:
            reset_stream()
        local_ip = get_local_ip_on_192_168_10()
        if not local_ip:
            raise RuntimeError("192.168.10.x のIPが見つかりません。TelloのWi-Fiに接続してください。")

        # PyAV優先
        self.mode = None
        self.rx = None
        try:
            import av  # 存在確認
            self.rx = _PyAVReceiver(local_ip, warmup_sec=warmup_sec)
            self.mode = "pyav"
            return
        except Exception as e:
            print(f"[WARN] PyAV受信に失敗: {e}\n[INFO] OpenCVへフォールバックします。", file=sys.stderr)

        # OpenCVフォールバック
        self.rx = _OpenCVReceiver(warmup_sec=warmup_sec)
        self.mode = "opencv"

    def get(self):
        latest = self.rx.read()
        if latest is None:
            return time.time(), None
        return latest  # (ts, bgr)

    def close(self):
        if self.rx:
            self.rx.release()

# ========= Backend =========
def relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = retrieval_database.update(
            frame, add_after_query=False,
            k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
        )
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            frame_idx = [n_kf - 1] * len(kf_idx)
            if factor_graph.add_factors(
                frame_idx, list(kf_idx),
                config["reloc"]["min_match_frac"], is_reloc=config["reloc"]["strict"]
            ):
                retrieval_database.update(
                    frame, add_after_query=True,
                    k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
                )
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[list(kf_idx)[0]].clone()
                if config["use_calib"]: factor_graph.solve_GN_calib()
                else:                   factor_graph.solve_GN_rays()
                return True
            else:
                keyframes.pop_last()
        return False

def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    while states.get_mode() is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01); continue

        if mode == Mode.RELOC:
            frame = states.get_frame()
            if relocalization(frame, keyframes, factor_graph, retrieval_database):
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue

        idx = -1
        with states.lock:
            if states.global_optimizer_tasks:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01); continue

        frame = keyframes[idx]
        kf_idx = retrieval_database.update(
            frame, add_after_query=True,
            k=config["retrieval"]["k"], min_thresh=config["retrieval"]["min_thresh"]
        )
        if kf_idx:
            factor_graph.add_factors(
                list(kf_idx), [idx]*len(kf_idx), config["local_opt"]["min_match_frac"]
            )

        if config["use_calib"]: factor_graph.solve_GN_calib()
        else:                   factor_graph.solve_GN_rays()

        with states.lock:
            if states.global_optimizer_tasks:
                states.global_optimizer_tasks.pop(0)

# ========= Main =========
def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="config/base.yaml")
    ap.add_argument("--calib",   default="")
    ap.add_argument("--proc_size", type=int, default=224, choices=[224, 512])
    ap.add_argument("--no-viz",  action="store_true")
    ap.add_argument("--reset_stream", type=int, default=1,
                    help="1で開始時に streamoff→on を送ってIフレーム誘発")
    # Latency/負荷チューニング
    ap.add_argument("--frame_skip", type=int, default=0,
                    help="N>0で(N+1)枚中1枚のみ処理（例:1→半分）")
    ap.add_argument("--viz_interval", type=int, default=2,
                    help="可視化フレーム間引き（1=毎フレーム）")
    ap.add_argument("--go_every_kf", type=int, default=2,
                    help="グローバル最適化は M 枚に1回に間引き")
    ap.add_argument("--stale_thresh_ms", type=float, default=45.0,
                    help="受信→今の差がこの閾値より古いフレームは捨てる")
    args = ap.parse_args()

    load_config(args.config)

    # 受信開始
    src = TelloSource(warmup_sec=5.0, do_reset=bool(args.reset_stream))
    print(f"[Live/Tello] receiver={getattr(src,'mode','unk')} proc_size={args.proc_size}")

    # 最初の有効フレームでネット入力サイズ確定（最大5秒待つ）
    t0 = time.time()
    first = None
    while time.time() - t0 < 5.0:
        ts0, fr0 = src.get()
        if fr0 is not None:
            first = fr0
            break
        time.sleep(0.01)
    if first is None:
        # 最後の手段：ストリーム再起動
        reset_stream()
        time.sleep(0.5)
        t1 = time.time()
        while time.time() - t1 < 3.0:
            ts0, fr0 = src.get()
            if fr0 is not None:
                first = fr0
                break
            time.sleep(0.01)
    if first is None:
        raise RuntimeError("初期フレームを取得できません。")

    def resize_bgr(fr, size):
        return cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)

    img_bgr0 = resize_bgr(first, args.proc_size)
    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img_rgb0, T_WC0, img_size=args.proc_size, device=device)
    H_real, W_real = frame0.img.shape[-2:]
    print(f"[Init] Shared buffers: {H_real}x{W_real}")

    # 共有
    manager  = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)
    keyframes = SharedKeyframes(manager, H_real, W_real)
    states    = SharedStates(manager,  H_real, W_real)
    states.unpause()
    states.set_mode(Mode.INIT)

    # 可視化
    if not args.no_viz:
        viz = mp.Process(target=run_visualization, args=(config, states, keyframes, main2viz, viz2main))
        viz.start()
        print(f"[Viz] pid={viz.pid} started")

    # モデル
    model = load_mast3r(device=device)
    model.share_memory()

    # キャリブ（任意）
    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intr = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        intrinsics = Intrinsics.from_calib((H_real, W_real), intr["width"], intr["height"], intr["calibration"])
        K = torch.from_numpy(intrinsics.K_frame).to(device, dtype=torch.float32)
        keyframes.set_intrinsics(K)

    tracker  = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    # FP16コンテキスト
    amp = torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()

    def make_uimg_rgb_from_bgr(img_bgr, h, w):
        u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if u.shape[0] != h or u.shape[1] != w:
            u = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
        u = (u.astype(np.float32) / 255.0).copy()
        return torch.from_numpy(u)

    # メインループ
    i = 0
    cap_time_acc = inf_time_acc = tot_time_acc = 0.0
    go_mod = max(1, args.go_every_kf)

    try:
        while True:
            loop_start = time.time()

            # 可視化メッセージ
            mode = states.get_mode()
            msg  = try_get_msg(viz2main)
            if msg: last_msg = msg
            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED); break
            if last_msg.is_paused and not last_msg.next:
                states.pause(); time.sleep(0.01); continue
            else:
                states.unpause()

            # 取り込み（最新のみ）
            cap_start = time.time()
            ts, img_bgr = src.get()
            if img_bgr is None:
                time.sleep(0.001); continue
            cap_time_acc += (time.time() - cap_start)

            # ステイル破棄
            stale_ms = (time.time() - ts) * 1000.0
            if stale_ms > args.stale_thresh_ms:
                retried = 0
                replaced = False
                while retried < 5:
                    ts2, fr2 = src.get()
                    if fr2 is not None and (time.time() - ts2) * 1000.0 <= args.stale_thresh_ms:
                        ts, img_bgr = ts2, fr2
                        stale_ms = (time.time() - ts) * 1000.0
                        replaced = True
                        break
                    retried += 1
                    time.sleep(0.001)
                if not replaced:
                    continue

            # フレームスキップ
            if args.frame_skip > 0 and (i % (args.frame_skip + 1)) != 0:
                i += 1
                continue

            # 前処理
            img_bgr = cv2.resize(img_bgr, (args.proc_size, args.proc_size), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            prev_for_T = states.get_frame()
            T_WC = lietorch.Sim3.Identity(1, device=device) if (i == 0 or prev_for_T is None) else prev_for_T.T_WC
            frame = create_frame(i, img_rgb, T_WC, img_size=args.proc_size, device=device)
            frame.uimg = make_uimg_rgb_from_bgr(img_bgr, H_real, W_real)

            # INIT復帰ガード
            if len(keyframes) == 0 and mode != Mode.INIT:
                states.set_mode(Mode.INIT)
                mode = Mode.INIT

            # 推論・追跡
            inf_start = time.time()
            did_geometry_update = False

            if mode == Mode.INIT:
                with amp:
                    X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                if (len(keyframes) % go_mod) == 0:
                    states.queue_global_optimization(len(keyframes)-1)
                states.set_mode(Mode.TRACKING)
                did_geometry_update = True

            elif mode == Mode.TRACKING:
                with amp:
                    add_new_kf, _, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                if add_new_kf:
                    keyframes.append(frame)
                    if (len(keyframes) % go_mod) == 0:
                        states.queue_global_optimization(len(keyframes)-1)

            elif mode == Mode.RELOC:
                with amp:
                    X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                states.queue_reloc()
                did_geometry_update = True

            inf_time_acc += (time.time() - inf_start)
            inf_last_ms = (time.time() - inf_start) * 1000.0

            # 幾何未更新なら前フレームから継承
            if not did_geometry_update:
                prev = states.get_frame()
                if prev is not None:
                    for attr in ("X_canon", "C_canon", "feat", "pos", "pointmap"):
                        if getattr(prev, attr, None) is not None and getattr(frame, attr, None) is None:
                            setattr(frame, attr, getattr(prev, attr).clone())

            # 可視化への投入（間引き）
            if args.viz_interval <= 1 or (i % args.viz_interval) == 0:
                states.set_frame(frame)

            # 統計
            tot_time_acc += (time.time() - loop_start)
            if i > 0 and (i % 60 == 0):
                def avg_ms_fps(total_s, n=60):
                    avg_ms = 1000.0 * (total_s / n)
                    fps = 1000.0 / avg_ms if avg_ms > 0 else float('inf')
                    return avg_ms, fps
                cap_ms, cap_fps = avg_ms_fps(cap_time_acc)
                inf_ms_avg, inf_fps = avg_ms_fps(inf_time_acc)
                tot_ms, tot_fps = avg_ms_fps(tot_time_acc)
                e2e_ms = (time.time() - ts) * 1000.0
                print(f"[Last 60] Capture: {cap_ms:.2f} ms ({cap_fps:.1f} FPS) | "
                      f"Infer(avg): {inf_ms_avg:.2f} ms ({inf_fps:.1f} FPS) | "
                      f"Loop: {tot_ms:.2f} ms ({tot_fps:.1f} FPS) | "
                      f"[Latency] recv->now(last): {e2e_ms:.1f} ms | stale_in: {stale_ms:.1f} ms | infer(last): {inf_last_ms:.1f} ms")
                cap_time_acc = inf_time_acc = tot_time_acc = 0.0

            i += 1

    finally:
        try: states.set_mode(Mode.TERMINATED)
        except Exception: pass
        if not args.no_viz:
            try: viz.join(timeout=1.0)
            except: pass
        try: src.close()
        except: pass

if __name__ == "__main__":
    main()
