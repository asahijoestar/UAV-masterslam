#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tello.py — TelloのUDP映像を直接取り込み、INIT→KF生成→TRACKING を確実に実行

import os, time, cv2, yaml, argparse, numpy as np, torch, lietorch, socket
import torch.multiprocessing as mp

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.visualization import WindowMsg, run_visualization

# ---- 低遅延：古いフレームを溜めないFFmpeg設定 ----
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "protocol_whitelist;file,udp,rtp|fflags;nobuffer|flags;low_delay|max_delay;0|min_delay;1|reorder_queue_size;0"
)

TELLO_UDP_URL = "udp://0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000"
TELLO_ADDR = ("192.168.10.1", 8889)

# -----------------------------
# Tello Live camera -> BGR
# -----------------------------
class LiveRGBDataset:
    def __init__(self, proc_size=512, src=TELLO_UDP_URL, warmup_sec=3.0, reset_stream=False):
        self.proc_size = int(proc_size)

        if reset_stream:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(1.0)
                s.sendto(b"command", TELLO_ADDR);  time.sleep(0.2)
                s.sendto(b"streamoff", TELLO_ADDR); time.sleep(0.4)
                s.sendto(b"streamon", TELLO_ADDR)
                s.close()
                time.sleep(0.5)
            except Exception:
                pass

        # 👇ここで self.src = ... を削除し、直接 OpenCV VideoCapture を使う
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass


        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError("Tello UDPストリームを開けません。setuzoku.py で streamon 後に実行してください。")

        # ウォームアップ：有効フレームを待つ（PPSエラーが消えるまで）
        t0 = time.time()
        ok, fr = False, None
        while time.time() - t0 < warmup_sec:
            ok, fr = self.cap.read()
            if ok and fr is not None and fr.size > 0:
                break
            time.sleep(0.01)
        if not ok or fr is None or fr.size == 0:
            raise RuntimeError("Telloから有効なフレームを取得できません（電波/streamonを確認）。")

        fr = cv2.resize(fr, (self.proc_size, self.proc_size), interpolation=cv2.INTER_AREA)
        self._shape = ((self.proc_size, self.proc_size, 3),)
        self.img_size = (self.proc_size, self.proc_size)

    def __getitem__(self, i):
        ok, fr = self.cap.read()
        if not ok or fr is None or fr.size == 0:
            return time.time(), None
        fr = cv2.resize(fr, (self.proc_size, self.proc_size), interpolation=cv2.INTER_AREA)
        return time.time(), fr

    def __len__(self): return 1 << 60
    def get_img_shape(self): return self._shape
    def subsample(self, n): pass
    def has_calib(self): return False


# -----------------------------
# Backend（既存ロジック準拠）
# -----------------------------
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


# -----------------------------
# main
# -----------------------------
def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="config/base.yaml")
    ap.add_argument("--save-as", default="default")
    ap.add_argument("--no-viz",  action="store_true")
    ap.add_argument("--calib",   default="")
    ap.add_argument("--proc_size", type=int, default=512, choices=[224, 512])
    ap.add_argument("--reset_stream", type=int, default=0,
                    help="1で開始時に command→streamoff→streamon を送ってIフレームを促す")
    args = ap.parse_args()

    load_config(args.config)

    manager  = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # 1) Tello ライブ入力
    dataset = LiveRGBDataset(proc_size=args.proc_size, reset_stream=bool(args.reset_stream))
    print(f"[Live/Tello] url={TELLO_UDP_URL} size={args.proc_size}")

    # 2) 最初のフレームでネット入力サイズを確定
    ts0, img_bgr0 = dataset[0]
    if img_bgr0 is None:
        raise RuntimeError("最初のフレームを取得できませんでした。")
    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img_rgb0, T_WC0, img_size=args.proc_size, device=device)
    H_real, W_real = frame0.img.shape[-2:]
    print(f"[Init] Shared buffers: {H_real}x{W_real}")

    # 3) 共有バッファと状態
    keyframes = SharedKeyframes(manager, H_real, W_real)
    states    = SharedStates(manager,  H_real, W_real)
    states.unpause()
    states.set_mode(Mode.INIT)  # ★必ずINIT開始

    # 4) 可視化
    if not args.no_viz:
        viz = mp.Process(target=run_visualization, args=(config, states, keyframes, main2viz, viz2main))
        viz.start()
        print(f"[Viz] pid={viz.pid} started")

    # 5) モデル
    model = load_mast3r(device=device)
    model.share_memory()

    # 6) キャリブ（任意）
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

    # 7) uimg 作成ヘルパ（書き込み可な配列にして警告回避）
    def make_uimg_rgb_from_bgr(img_bgr, h, w):
        u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if u.shape[0] != h or u.shape[1] != w:
            u = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
        u = (u.astype(np.float32) / 255.0).copy()  # ★ writable
        return torch.from_numpy(u)

    # 8) メインループ
    i = 0
    cap_time_acc = inf_time_acc = tot_time_acc = 0.0

    while True:
        loop_start = time.time()

        # 可視化からのメッセージ処理
        mode = states.get_mode()
        msg  = try_get_msg(viz2main)
        if msg: last_msg = msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED); break
        if last_msg.is_paused and not last_msg.next:
            states.pause(); time.sleep(0.01); continue
        else:
            states.unpause()

        # 取り込み
        cap_start = time.time()
        ts, img_bgr = dataset[i]
        if img_bgr is None:
            time.sleep(0.002); continue
        cap_time_acc += (time.time() - cap_start)

        # Frame 準備（T初期値は直前のTを継承）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        prev_for_T = states.get_frame()
        T_WC = lietorch.Sim3.Identity(1, device=device) if (i == 0 or prev_for_T is None) else prev_for_T.T_WC
        frame = create_frame(i, img_rgb, T_WC, img_size=args.proc_size, device=device)
        frame.uimg = make_uimg_rgb_from_bgr(img_bgr, H_real, W_real)

        # ★ KFが0なら必ずINITへ（復帰ガード）
        if len(keyframes) == 0 and mode != Mode.INIT:
            states.set_mode(Mode.INIT)
            mode = Mode.INIT

        # 推論／追跡（★ 必ず幾何を確定してから set_frame を呼ぶ）
        inf_start = time.time()
        did_geometry_update = False

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes)-1)
            states.set_mode(Mode.TRACKING)
            did_geometry_update = True

        elif mode == Mode.TRACKING:
            add_new_kf, _, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            if add_new_kf:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes)-1)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.queue_reloc()
            did_geometry_update = True

        inf_time_acc += (time.time() - inf_start)

        # 幾何未更新なら前フレームの属性を継承して、Noneを避ける
        if not did_geometry_update:
            prev = states.get_frame()
            if prev is not None:
                for attr in ("X_canon", "C_canon", "feat", "pos", "pointmap"):
                    if getattr(prev, attr, None) is not None and getattr(frame, attr, None) is None:
                        setattr(frame, attr, getattr(prev, attr).clone())

        # ★ 最後に共有へ反映（幾何が揃ってから）
        states.set_frame(frame)

        # 60フレームごとに平均時間＆FPSを出力
        tot_time_acc += (time.time() - loop_start)
        if i > 0 and (i % 60 == 0):
            def avg_ms_fps(total_s, n=60):
                avg_ms = 1000.0 * (total_s / n)
                fps = 1000.0 / avg_ms if avg_ms > 0 else float('inf')
                return avg_ms, fps
            cap_ms, cap_fps = avg_ms_fps(cap_time_acc)
            inf_ms, inf_fps = avg_ms_fps(inf_time_acc)
            tot_ms, tot_fps = avg_ms_fps(tot_time_acc)
            print(f"[Last 60 frames] "
                  f"Capture: {cap_ms:.2f} ms ({cap_fps:.2f} FPS) | "
                  f"Inference: {inf_ms:.2f} ms ({inf_fps:.2f} FPS) | "
                  f"Total loop: {tot_ms:.2f} ms ({tot_fps:.2f} FPS)")
            cap_time_acc = inf_time_acc = tot_time_acc = 0.0

        i += 1

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()


if __name__ == "__main__":
    main()
