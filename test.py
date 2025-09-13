#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, cv2, torch, yaml, lietorch
import numpy as np
import torch.multiprocessing as mp
from contextlib import nullcontext

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.visualization import WindowMsg, run_visualization


# ---------- Camera ----------
class LiveRGBDataset:
    def __init__(self, src, proc_size=512, fps=0.0, yuv_mode="auto"):
        self.proc_size = int(proc_size)
        self.yuv_mode = yuv_mode.upper() if isinstance(yuv_mode, str) else "AUTO"

        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            try:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # 実質BGR
            except Exception:
                pass
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(self.proc_size))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.proc_size))
            if fps and fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, float(fps))
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")

        ok, fr = self.cap.read()
        if not ok or fr is None:
            raise RuntimeError("No frames from camera")

        fr = self._to_bgr(fr)
        fr = cv2.resize(fr, (self.proc_size, self.proc_size), interpolation=cv2.INTER_AREA)
        self._shape = ((self.proc_size, self.proc_size, 3),)
        self.img_size = (self.proc_size, self.proc_size)

    def _to_bgr(self, fr):
        if fr.ndim == 2:
            return cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        if fr.ndim == 3 and fr.shape[2] == 2:
            if self.yuv_mode == "UYVY":
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_UYVY)
            elif self.yuv_mode in ("YUY2", "YUYV"):
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_YUY2)
            elif self.yuv_mode == "NV12":
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_NV12)
            else:
                return cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_YUY2)
        return fr

    def __getitem__(self, i):
        ok, fr = self.cap.read()
        if not ok or fr is None:
            return time.time(), None
        fr = self._to_bgr(fr)
        fr = cv2.resize(fr, (self.proc_size, self.proc_size), interpolation=cv2.INTER_AREA)
        return time.time(), fr

    def __len__(self): return 1 << 60
    def get_img_shape(self): return self._shape
    def subsample(self, n): pass
    def has_calib(self): return False


# ---------- backend ----------
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


# ---------- helpers ----------
def make_uimg_rgb_from_bgr(img_bgr, h, w):
    u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if u.shape[0] != h or u.shape[1] != w:
        u = cv2.resize(u, (w, h), interpolation=cv2.INTER_AREA)
    u = (u.astype(np.float32) / 255.0).copy()
    return torch.from_numpy(u)  # HWC float32 RGB


def borrow_prev_for_viz(dst, prev):
    """vizに必要な属性を前フレームから借りる（存在するものだけ）。"""
    if prev is None:
        return
    for name in ["X_canon", "C", "pointmap", "feat", "desc", "pos", "ind"]:
        val = getattr(prev, name, None)
        if val is not None:
            try:
                setattr(dst, name, val.clone())
            except Exception:
                setattr(dst, name, val)


# ---------- main ----------
if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config",  default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz",  action="store_true")
    parser.add_argument("--calib",   default="")
    parser.add_argument("--camera",   type=str,  default=None,
                        help="Live source. '0' for /dev/video0, 'udp://0.0.0.0:11111?...', or video path")
    parser.add_argument("--proc_size", type=int, default=512, choices=[224, 512],
                        help="Model input size (square). 224 or 512")
    parser.add_argument("--cam_fps",  type=float, default=0.0,
                        help="Request camera FPS (0 = auto)")
    parser.add_argument("--yuv",      type=str,  default="auto",
                        choices=["auto", "YUY2", "YUYV", "UYVY", "NV12"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--proc-hz", type=float, default=0.0,
                        help="Run heavy SLAM step at this Hz (0=every frame).")
    args = parser.parse_args()

    load_config(args.config)

    manager  = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # dataset
    if args.camera is not None:
        dataset = LiveRGBDataset(src=args.camera, proc_size=args.proc_size, fps=args.cam_fps, yuv_mode=args.yuv)
        print(f"[Live] source={args.camera} size={args.proc_size} fps={args.cam_fps} yuv={args.yuv}")
    else:
        dataset = load_dataset(args.dataset)
        dataset.subsample(config["dataset"]["subsample"])

    # first frame
    ts0, img_bgr0 = dataset[0]
    if img_bgr0 is None:
        raise RuntimeError("Failed to grab the first frame.")
    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    T_WC0 = lietorch.Sim3.Identity(1, device=device)
    frame0 = create_frame(0, img_rgb0, T_WC0, img_size=args.proc_size, device=device)
    H_real, W_real = frame0.img.shape[-2:]

    print(f"[Init] Shared buffers: {H_real}x{W_real}")
    keyframes = SharedKeyframes(manager, H_real, W_real)
    states    = SharedStates(manager,  H_real, W_real)
    states.unpause()

    if not args.no_viz:
        viz = mp.Process(target=run_visualization, args=(config, states, keyframes, main2viz, viz2main))
        viz.start()
        print(f"[Viz] pid={viz.pid} started")

    model = load_mast3r(device=device)
    model.share_memory()

    # calib
    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intr = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        if hasattr(dataset, "use_calibration"): dataset.use_calibration = True
        if hasattr(dataset, "camera_intrinsics"):
            dataset.camera_intrinsics = Intrinsics.from_calib(
                (H_real, W_real), intr["width"], intr["height"], intr["calibration"]
            )
            K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(device, dtype=torch.float32)
            keyframes.set_intrinsics(K)

    tracker  = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    amp_ctx = nullcontext() if args.no_amp else torch.autocast(device_type="cuda", dtype=torch.float16)

    # loop
    i = 0
    t0 = time.time()
    cap_cnt, proc_cnt, skipped_proc = 0, 0, 0
    last_log_t = t0
    proc_period = (1.0 / args.proc_hz) if args.proc_hz > 0 else 0.0
    last_proc_t = 0.0
    inited = False

    while True:
        mode = states.get_mode()
        msg  = try_get_msg(viz2main)
        last_msg = msg if msg else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED); break
        if last_msg.is_paused and not last_msg.next:
            states.pause(); time.sleep(0.005); continue
        else:
            states.unpause()

        ts, img_bgr = dataset[i]
        if img_bgr is None:
            time.sleep(0.002); continue
        cap_cnt += 1

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        T_WC = lietorch.Sim3.Identity(1, device=device) if i == 0 else states.get_frame().T_WC
        frame = create_frame(i, img_rgb, T_WC, img_size=args.proc_size, device=device)
        frame.uimg = make_uimg_rgb_from_bgr(img_bgr, H_real, W_real)

        now = time.time()
        do_heavy = (proc_period == 0.0) or (now - last_proc_t >= proc_period) or (not inited)

        if do_heavy:
            if mode == Mode.INIT:
                with amp_ctx:
                    X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes)-1)
                states.set_mode(Mode.TRACKING)
            elif mode == Mode.TRACKING:
                with amp_ctx:
                    add_new_kf, _, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                    # --- 強制的に Nフレームおきに KF を追加 ---
                #if i % 5 == 0:   # 5フレームごとに必ず KF
                    #add_new_kf = True
                if add_new_kf:
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes)-1)
            elif mode == Mode.RELOC:
                with amp_ctx:
                    X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                states.queue_reloc()

            last_proc_t = time.time()
            proc_cnt += 1
            inited = True
        else:
            prev = states.get_frame()
            borrow_prev_for_viz(frame, prev)
            skipped_proc += 1

        # 安全に set_frame
        try:
            states.set_frame(frame)
        except TypeError:
            prev = states.get_frame()
            if prev is not None:
                states.set_frame(prev)

        if now - last_log_t >= 1.0:
            elapsed = now - t0
            print(f"Capture FPS: {cap_cnt/elapsed:.2f}, Throughput FPS: {i/elapsed:.2f}, "
                  f"Proc FPS: {proc_cnt/elapsed:.2f} (skipped {skipped_proc})")
            last_log_t = now

        i += 1

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
