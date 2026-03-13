# Features:
#   1. Zone-based goods theft detection (GoodsZoneGuard)
#      - Only goods zones (--goods_zone_names) trigger theft logic
#      - Median baseline item count when zone is clear
#      - GUEST enters OR reaches wrist near item in zone → freeze count
#      - STAFF presence also freezes count but does NOT trigger alert
#      - If staff also touched zone in same event → suppress alert (avoid false positive)
#      - Guest exits → cooldown → recount with RAW (non-ghost) items
#      - If items decreased → ALERT "GOODS TAKEN"
#   2. Concealment detection (pick up & hide)
#      - Item tracking IDs via ByteTrack (det_model.track)
#      - Item center near wrist keypoints for N frames → "being carried"
#      - Carried item disappears → ALERT "ITEM CONCEALED" (guests only)
#   3. Staff/Guest role labeling
#      - det_model has 3 classes: staff, guest, item
#      - staff/guest boxes matched to pose persons by IoU + center distance
#      - Label confirmed after 2-3 consecutive detections; upgrades if
#        a different role appears consistently
#      - Label sticks on pose person even when det_model loses detection

import os, json, time, argparse, subprocess, threading, re, collections
import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def clamp(v, a, b): return max(a, min(b, v))

def point_in_poly(pt, poly_pts):
    poly = np.array(poly_pts, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def xyxy_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def poly_label_anchor(pts):
    return int(min(p[0] for p in pts)), int(min(p[1] for p in pts))

def dist2(a, b): return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def load_zones(zones_path):
    with open(zones_path, "r", encoding="utf-8") as f:
        zdata = json.load(f)
    zones = zdata.get("zones", [])
    if not isinstance(zones, list) or not zones:
        raise ValueError("zones.json: expecting {'zones':[{name,points},...]}")
    for z in zones:
        if "name" not in z or "points" not in z or not z["points"]:
            raise ValueError("Each zone needs 'name' and non-empty 'points'")
    return zones

def is_rtsp(src): return src.lower().startswith("rtsp://")

def probe_size_with_ffprobe(src, ffprobe_path="ffprobe", timeout_s=6):
    try:
        cmd = [ffprobe_path, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "json", src]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if p.returncode != 0 or not p.stdout: return None, None
        s = json.loads(p.stdout).get("streams", [])
        if not s: return None, None
        w, h = int(s[0].get("width", 0)), int(s[0].get("height", 0))
        return (w, h) if w > 0 and h > 0 else (None, None)
    except Exception:
        return None, None

def bbox_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def bbox_contains_point(bbox, pt):
    x1, y1, x2, y2 = bbox
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def norm_name(s):
    return str(s).strip().lower().replace(" ", "")

def point_valid(pt):
    return pt[0] > 1 and pt[1] > 1

def items_inside_zone(items, zone_pts):
    return [it for it in items if point_in_poly(it["center"], zone_pts)]

def person_role_is(person, role):
    return person.get("role", "unknown") == role

# ─────────────────────────────────────────────────────────────
# Pose skeleton  (COCO-17)
# ─────────────────────────────────────────────────────────────
COCO17_EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

KP_L_ELBOW,  KP_R_ELBOW  = 7, 8
KP_L_WRIST,  KP_R_WRIST  = 9, 10
_ARM_KPS = (KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST)

ROLE_COLORS = {
    "staff":   (0,   200, 80),
    "guest":   (200, 80,  0),
    "unknown": (255, 80,  80),
}

def draw_pose_skeleton(vis, kpts, color=(255, 80, 80)):
    for a, b in COCO17_EDGES:
        xa, ya = kpts[a]; xb, yb = kpts[b]
        if xa > 1 and ya > 1 and xb > 1 and yb > 1:
            cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for x, y in kpts:
        if x > 1 and y > 1:
            cv2.circle(vis, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# Person-zone interaction helper
# ─────────────────────────────────────────────────────────────
def person_interacts_with_goods_zone(person, zone_pts, zone_items, reach_dist=55):
    """
    True if:
      - body center inside zone, OR
      - any arm/wrist keypoint inside zone, OR
      - wrist is within reach_dist px of any item inside the zone
        (handles leaning-in / arm-stretch from outside zone boundary)
    """
    if point_in_poly(xyxy_center(person["bbox"]), zone_pts):
        return True
    for ki in _ARM_KPS:
        kx, ky = person["kpts"][ki]
        if point_valid((kx, ky)) and point_in_poly((kx, ky), zone_pts):
            return True
    for ki in (KP_L_WRIST, KP_R_WRIST):
        wx, wy = person["kpts"][ki]
        if not point_valid((wx, wy)):
            continue
        for it in zone_items:
            if dist2((wx, wy), it["center"]) <= reach_dist ** 2:
                return True
    return False

# ─────────────────────────────────────────────────────────────
# Person Role Tracker
#
# Pose is the authoritative "person source".
# det_model only supplies staff/guest labels to stick onto pose persons.
#
# Matching pose<->track   : IoU + center distance (robust to partial occlusion)
# Matching role det<->pose: IoU OR role-box center inside person box
# Confirmation            : same role seen >= confirm_frames consecutively
# Label persistence       : confirmed_role never cleared when det_model loses
#                           detection; only changes if new role streaks enough
# ─────────────────────────────────────────────────────────────
class PersonRoleTracker:
    def __init__(self,
                 confirm_frames: int        = 3,
                 history_len: int           = 10,
                 iou_thresh: float          = 0.25,
                 max_age_s: float           = 3.0,
                 person_match_iou: float    = 0.15,
                 person_match_center: float = 120.0,
                 candidate_ttl_s: float     = 1.0):
        self.confirm_frames       = confirm_frames
        self.history_len          = history_len
        self.iou_thresh           = iou_thresh
        self.max_age_s            = max_age_s
        self.person_match_iou     = person_match_iou
        self.person_match_center  = person_match_center
        self.candidate_ttl_s      = candidate_ttl_s
        self._tracks: list        = []
        self._next_track_id: int  = 1

    def _person_track_score(self, person_bbox, track_bbox):
        iou = bbox_iou(person_bbox, track_bbox)
        pcx, pcy = xyxy_center(person_bbox)
        tcx, tcy = xyxy_center(track_bbox)
        cd = float(np.hypot(pcx - tcx, pcy - tcy))
        if iou < self.person_match_iou and cd > self.person_match_center:
            return None
        dist_score = max(0.0, 1.0 - cd / max(1.0, self.person_match_center))
        return iou + 0.35 * dist_score

    def _match_persons_to_tracks(self, persons):
        assignments = [None] * len(persons)
        scores = []
        for pi, p in enumerate(persons):
            for ti, t in enumerate(self._tracks):
                s = self._person_track_score(p["bbox"], t["bbox"])
                if s is not None:
                    scores.append((s, pi, ti))
        scores.sort(key=lambda x: -x[0])
        used_p, used_t = set(), set()
        for s, pi, ti in scores:
            if pi not in used_p and ti not in used_t:
                assignments[pi] = ti
                used_p.add(pi); used_t.add(ti)
        return assignments

    def _role_person_score(self, person_bbox, role_bbox):
        iou = bbox_iou(person_bbox, role_bbox)
        if iou >= self.iou_thresh:
            return iou
        rc = xyxy_center(role_bbox)
        if bbox_contains_point(person_bbox, rc):
            return 0.35
        return None

    def _match_role_dets_to_persons(self, persons, role_detections):
        assignments = [None] * len(persons)
        scores = []
        for ri, rd in enumerate(role_detections):
            for pi, p in enumerate(persons):
                s = self._role_person_score(p["bbox"], rd["bbox"])
                if s is not None:
                    scores.append((s, rd["conf"], pi, ri))
        scores.sort(key=lambda x: (-x[0], -x[1]))
        used_p, used_r = set(), set()
        for s, conf, pi, ri in scores:
            if pi not in used_p and ri not in used_r:
                assignments[pi] = ri
                used_p.add(pi); used_r.add(ri)
        return assignments

    def update(self, persons: list, role_detections: list, now: float):
        """
        persons        : list of {bbox, kpts} from pose model
        role_detections: list of {bbox, name, conf} from det_model (staff/guest only)
        now            : time.time()

        Adds "role" and "role_conf" keys to each person dict in-place.
        """
        track_assign = self._match_persons_to_tracks(persons)

        for pi, p in enumerate(persons):
            ti = track_assign[pi]
            if ti is None:
                self._tracks.append({
                    "track_id":           self._next_track_id,
                    "bbox":               p["bbox"],
                    "confirmed_role":     "unknown",
                    "confirmed_conf":     0.0,
                    "candidate_role":     None,
                    "candidate_count":    0,
                    "candidate_conf_sum": 0.0,
                    "candidate_last_ts":  0.0,
                    "last_seen_ts":       now,
                })
                self._next_track_id += 1
                ti = len(self._tracks) - 1
            else:
                self._tracks[ti]["bbox"]         = p["bbox"]
                self._tracks[ti]["last_seen_ts"] = now
            p["_track_idx"] = ti

        role_assign = self._match_role_dets_to_persons(persons, role_detections)

        for pi, p in enumerate(persons):
            ti    = p["_track_idx"]
            track = self._tracks[ti]
            ri    = role_assign[pi]

            if ri is not None:
                rd        = role_detections[ri]
                best_role = rd["name"]
                best_conf = float(rd["conf"])

                same_candidate = (
                    track["candidate_role"] == best_role and
                    (now - track["candidate_last_ts"]) <= self.candidate_ttl_s
                )
                if same_candidate:
                    track["candidate_count"]    += 1
                    track["candidate_conf_sum"] += best_conf
                else:
                    track["candidate_role"]      = best_role
                    track["candidate_count"]     = 1
                    track["candidate_conf_sum"]  = best_conf
                track["candidate_last_ts"] = now

                # Confirm / upgrade once streak is long enough
                if track["candidate_count"] >= self.confirm_frames:
                    track["confirmed_role"] = best_role
                    track["confirmed_conf"] = (
                        track["candidate_conf_sum"] /
                        max(1, track["candidate_count"])
                    )
            else:
                # No role detection this frame.
                # Do NOT clear confirmed_role — label sticks on pose person.
                # Only expire the in-progress candidate if too much time passed.
                if (track["candidate_role"] is not None and
                        (now - track["candidate_last_ts"]) > self.candidate_ttl_s):
                    track["candidate_role"]      = None
                    track["candidate_count"]     = 0
                    track["candidate_conf_sum"]  = 0.0

            p["role"]      = track["confirmed_role"]
            p["role_conf"] = track["confirmed_conf"]
            del p["_track_idx"]

        self._tracks = [t for t in self._tracks
                        if (now - t["last_seen_ts"]) < self.max_age_s]
        return persons

# ─────────────────────────────────────────────────────────────
# Goods Zone Guard  (ROLE-AWARE, MEDIAN BASELINE)
#
# Alert only when:
#   - guest interacted with zone (body/arm/wrist near zone items)
#   - guest left, cooldown elapsed
#   - RAW visible item count dropped vs median baseline
#   - staff did NOT also touch zone during same event window
# ─────────────────────────────────────────────────────────────
class GoodsZoneGuard:
    IDLE           = 0
    GUEST_INTERACT = 1
    COOLDOWN       = 2
    ALERT          = 3

    COOLDOWN_SEC  = 1.5
    ALERT_DISPLAY = 10.0

    def __init__(self, name):
        self.name  = name
        self.state = self.IDLE

        self.baseline_count = 0
        self.clear_counts   = collections.deque(maxlen=8)  # median over last 8 clear readings

        self.exit_ts   = 0.0
        self.alert_ts  = 0.0
        self.alert_msg = ""
        self.missing   = 0

        self.had_guest_contact           = False
        self.staff_touched_during_window = False

    def _update_clear_baseline(self, count: int):
        self.clear_counts.append(int(count))
        self.baseline_count = (
            int(round(float(np.median(self.clear_counts))))
            if self.clear_counts else int(count)
        )

    def _reset_window(self):
        self.had_guest_contact           = False
        self.staff_touched_during_window = False

    def _go_idle(self, count: int):
        self.state = self.IDLE
        self.clear_counts.clear()
        self._update_clear_baseline(count)
        self._reset_window()

    def update(self, guest_interacting: bool, visible_item_count: int,
               now: float, staff_interacting: bool = False):
        """
        guest_interacting  : guest body/wrist touches zone or item in zone
        visible_item_count : RAW item count (no ghost buffer) visible in zone
        staff_interacting  : staff also in zone this frame (suppress false alert)
        """
        visible_item_count = int(visible_item_count)

        if self.state == self.IDLE:
            self._update_clear_baseline(visible_item_count)
            if guest_interacting:
                self.had_guest_contact           = True
                self.staff_touched_during_window = bool(staff_interacting)
                self.state = self.GUEST_INTERACT

        elif self.state == self.GUEST_INTERACT:
            self.staff_touched_during_window = (
                self.staff_touched_during_window or bool(staff_interacting)
            )
            if not guest_interacting:
                self.exit_ts = now
                self.state   = self.COOLDOWN

        elif self.state == self.COOLDOWN:
            self.staff_touched_during_window = (
                self.staff_touched_during_window or bool(staff_interacting)
            )
            if guest_interacting:
                # Guest came back — still the same event
                self.state = self.GUEST_INTERACT
            elif (now - self.exit_ts) >= self.COOLDOWN_SEC:
                diff = self.baseline_count - visible_item_count
                if self.had_guest_contact and diff > 0 and not self.staff_touched_during_window:
                    self.missing   = diff
                    self.alert_msg = (
                        f"GOODS TAKEN [{self.name}]  "
                        f"guest interacted -> "
                        f"{self.baseline_count} -> {visible_item_count} items"
                    )
                    self.alert_ts = now
                    self.state    = self.ALERT
                else:
                    self._go_idle(visible_item_count)

        elif self.state == self.ALERT:
            if guest_interacting:
                self.clear_counts.clear()
                self._update_clear_baseline(visible_item_count)
                self.had_guest_contact           = True
                self.staff_touched_during_window = bool(staff_interacting)
                self.state = self.GUEST_INTERACT
            elif (now - self.alert_ts) > self.ALERT_DISPLAY:
                self._go_idle(visible_item_count)

    @property
    def is_alerting(self):     return self.state == self.ALERT
    @property
    def person_blocking(self): return self.state in (self.GUEST_INTERACT, self.COOLDOWN)

# ─────────────────────────────────────────────────────────────
# Concealment Tracker  (GUEST-ONLY)
# ─────────────────────────────────────────────────────────────
class ConcealmentTracker:
    """
    Detects: item picked up by GUEST (wrist near item, item moves) ->
             item disappears -> ALERT
    Staff picking up items is silently ignored.
    """
    ALERT_DISPLAY = 10.0

    def __init__(self, carry_dist=90, carry_frames=3, move_thresh=18):
        self.carry_dist   = carry_dist
        self.carry_frames = carry_frames
        self.move_thresh  = move_thresh
        self._items: dict = {}
        self.alerts: list = []

    def update(self, items: list, people: list, now: float):
        current_ids = set()

        for it in items:
            tid = it.get("track_id")
            if tid is None: continue
            current_ids.add(tid)
            cx, cy = it["center"]

            near_wrist   = False
            carrier_role = "unknown"

            for p in people:
                for ki in (KP_L_WRIST, KP_R_WRIST):
                    wx, wy = p["kpts"][ki]
                    if (wx > 1 and wy > 1 and
                            dist2((cx, cy), (wx, wy)) < self.carry_dist ** 2):
                        near_wrist   = True
                        carrier_role = p.get("role", "unknown")
                        break
                if near_wrist: break

            if tid not in self._items:
                self._items[tid] = {
                    "name":               it.get("name", "item"),
                    "near_frames":        0,
                    "being_carried":      False,
                    "approach_confirmed": False,
                    "carrier_role":       "unknown",
                    "last_seen_ts":       now,
                    "anchor_pos":         (cx, cy),
                    "has_moved":          False,
                }
            rec = self._items[tid]
            rec["last_seen_ts"] = now
            rec["name"]         = it.get("name", rec["name"])

            if near_wrist:
                if rec["near_frames"] == 0:
                    rec["anchor_pos"]         = (cx, cy)
                    rec["has_moved"]          = False
                    rec["approach_confirmed"] = False
                rec["near_frames"]  += 1
                rec["carrier_role"]  = carrier_role
                ax, ay = rec["anchor_pos"]
                if dist2((cx, cy), (ax, ay)) >= self.move_thresh ** 2:
                    rec["has_moved"] = True
                if rec["has_moved"]:
                    rec["approach_confirmed"] = True
                if rec["near_frames"] >= self.carry_frames and rec["has_moved"]:
                    rec["being_carried"] = True
            else:
                rec["near_frames"] = max(0, rec["near_frames"] - 1)
                if rec["near_frames"] == 0:
                    rec["being_carried"]      = False
                    rec["has_moved"]          = False
                    rec["approach_confirmed"] = False

        for tid in list(self._items):
            rec = self._items[tid]
            age = now - rec["last_seen_ts"]
            if tid not in current_ids:
                is_guest_or_unknown = rec["carrier_role"] in ("guest", "unknown")
                if rec["approach_confirmed"] and age < 2.0 and is_guest_or_unknown:
                    self.alerts.append({
                        "msg": (f"CONCEALMENT: [{rec['name']}] id={tid} "
                                f"concealed by {rec['carrier_role'].upper()}"),
                        "ts": now,
                    })
                    rec["approach_confirmed"] = False
                    rec["being_carried"]      = False
                if age > 4.0:
                    del self._items[tid]

        self.alerts = [a for a in self.alerts
                       if (now - a["ts"]) < self.ALERT_DISPLAY]

    def being_carried_ids(self):
        return {tid for tid, r in self._items.items() if r["being_carried"]}

# ─────────────────────────────────────────────────────────────
# Alert overlay renderer
# ─────────────────────────────────────────────────────────────
def draw_alerts(vis, zone_guards_list, conceal, now):
    H, W = vis.shape[:2]
    msgs = []
    for zg in zone_guards_list:
        if zg is not None and zg.is_alerting:
            msgs.append(("zone", zg.alert_msg, now - zg.alert_ts, zg.ALERT_DISPLAY))
    for a in conceal.alerts:
        msgs.append(("conceal", a["msg"], now - a["ts"], conceal.ALERT_DISPLAY))
    if not msgs: return

    fh    = max(0.38, min(0.52, H / 800))
    lh    = int(fh * 42) + 6
    total = len(msgs) * lh + 12
    sy    = H - total - 8

    cv2.rectangle(vis, (0, sy - 8), (W, H), (0, 0, 0), -1)
    for i, (kind, msg, age, maxage) in enumerate(msgs):
        y        = sy + i * lh + lh
        blink_on = age < 3.0 and int(age * 4) % 2 == 0
        color    = (0, 0, 255) if kind == "zone" else (0, 60, 255)
        if not blink_on: color = tuple(int(c * 0.65) for c in color)
        bar_w = int((W - 20) * max(0.0, 1.0 - age / maxage))
        cv2.rectangle(vis, (10, y + 2), (10 + bar_w, y + 5), color, -1)
        icon = "!" if kind == "zone" else "!!"
        cv2.putText(vis, f"{icon} {msg}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fh, color, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# FFmpeg reader
# ─────────────────────────────────────────────────────────────
def build_ffmpeg_cmd(src, out_w, out_h, out_fps, ffmpeg_path, timeout_us, disabled_opts):
    vf_parts = []
    if out_w > 0 and out_h > 0: vf_parts.append(f"scale={out_w}:{out_h}")
    if out_fps > 0:              vf_parts.append(f"fps={out_fps}")
    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "warning"]
    if is_rtsp(src):
        cmd += ["-rtsp_transport", "tcp"]
        if "timeout"    not in disabled_opts: cmd += ["-timeout",    str(int(timeout_us))]
        if "fflags"     not in disabled_opts: cmd += ["-fflags",     "+nobuffer+discardcorrupt"]
        if "err_detect" not in disabled_opts: cmd += ["-err_detect", "ignore_err"]
        if "max_delay"  not in disabled_opts: cmd += ["-max_delay",  "500000"]
    cmd += ["-i", src, "-an", "-sn", "-dn"]
    if vf_parts: cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd

_UNREC_RE = re.compile(r"Unrecognized option '([^']+)'\.", re.IGNORECASE)

class FFmpegLatestFrameReader:
    def __init__(self, src, width, height, out_fps, timeout_us, ffmpeg_path="ffmpeg"):
        self.src = src; self.w = int(width); self.h = int(height)
        self.out_fps = float(out_fps); self.timeout_us = int(timeout_us)
        self.ffmpeg_path = ffmpeg_path; self.frame_size = self.w * self.h * 3
        self.proc = None; self.running = False; self.thread = None
        self.lock = threading.Lock(); self.latest = None
        self.latest_ts = 0.0; self.frame_id = 0
        self.last_err = ""; self.last_restart_ts = 0.0
        self.disabled_opts = set(); self.disable_count = 0
        self._restart_requested = False

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self): self.running = False; self._kill_proc()

    def get_latest(self):
        with self.lock:
            if self.latest is None: return None, 0.0, 0
            return self.latest.copy(), self.latest_ts, self.frame_id

    def _kill_proc(self):
        p, self.proc = self.proc, None
        if p is None: return
        try: p.kill()
        except: pass
        for attr in ("stdout", "stderr"):
            try:
                s = getattr(p, attr)
                if s: s.close()
            except: pass

    def _map_opt(self, opt):
        opt = opt.strip().lstrip("-")
        return {"timeout": "timeout", "stimeout": "timeout", "fflags": "fflags",
                "err_detect": "err_detect", "max_delay": "max_delay"}.get(opt, opt)

    def _stderr_loop(self, proc):
        try:
            while self.running and proc and proc.stderr:
                line = proc.stderr.readline()
                if not line: break
                s = line.decode(errors="ignore").strip()
                if not s: continue
                self.last_err = s; print("[FFMPEG]", s)
                m = _UNREC_RE.search(s)
                if m:
                    key = self._map_opt(m.group(1))
                    if key not in self.disabled_opts:
                        print(f"[FFMPEG] Auto-disabling: {key}")
                        self.disabled_opts.add(key); self.disable_count += 1
                        self._restart_requested = True
        except: pass

    def _start_proc(self):
        cmd = build_ffmpeg_cmd(self.src, self.w, self.h, self.out_fps,
                               self.ffmpeg_path, self.timeout_us, self.disabled_opts)
        cf = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        print("\n[FFMPEG CMD]\n" + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=10 ** 8, creationflags=cf)
        self.proc = proc; self.last_restart_ts = time.time()
        self._restart_requested = False
        threading.Thread(target=self._stderr_loop, args=(proc,), daemon=True).start()

    def _restart_proc(self):
        if time.time() - self.last_restart_ts < 0.4: time.sleep(0.4)
        self._kill_proc(); self._start_proc()

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n and self.running and self.proc and self.proc.stdout:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk: return None
            buf += chunk
        return buf if len(buf) == n else None

    def _loop(self):
        self._start_proc()
        while self.running:
            if self.proc is None: self._restart_proc(); continue
            if self.proc.poll() is not None or self._restart_requested:
                if self.disable_count > 8:
                    self.last_err = "Too many unsupported FFmpeg options."
                self._restart_proc(); continue
            raw = self._read_exact(self.frame_size)
            if raw is None: self._restart_proc(); continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
            with self.lock:
                self.latest    = frame
                self.latest_ts = time.time()
                self.frame_id += 1

# ─────────────────────────────────────────────────────────────
# Track Buffer  (for visual display only)
# ─────────────────────────────────────────────────────────────
class TrackBuffer:
    """Ghost buffer for smooth drawing only. Never use for loss logic."""
    def __init__(self, ghost_frames=6):
        self.ghost_frames = ghost_frames
        self._buf: dict   = {}

    def update(self, detected_items: list) -> list:
        now_ids = set()
        for it in detected_items:
            tid = it.get("track_id")
            if tid is None: continue
            now_ids.add(tid)
            self._buf[tid] = {"item": it, "ttl": self.ghost_frames}
        for tid in list(self._buf):
            if tid not in now_ids:
                self._buf[tid]["ttl"] -= 1
                if self._buf[tid]["ttl"] <= 0:
                    del self._buf[tid]
        out      = list(detected_items)
        seen_ids = {it.get("track_id") for it in detected_items}
        for tid, rec in self._buf.items():
            if tid not in seen_ids:
                out.append(rec["item"])
        return out

def filter_contained_boxes(items, iou_min_thresh=0.6):
    if len(items) < 2: return items
    def box_area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    def intersection(b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        return max(0, ix2-ix1) * max(0, iy2-iy1)
    n = len(items); suppress = [False]*n
    for i in range(n):
        for j in range(n):
            if i==j or suppress[i] or suppress[j]: continue
            b1=items[i]["bbox"]; b2=items[j]["bbox"]
            a1=box_area(b1);     a2=box_area(b2)
            if a1==0 or a2==0: continue
            inter  = intersection(b1, b2)
            io_min = inter / min(a1, a2)
            if io_min >= iou_min_thresh:
                if a1 > a2: suppress[i] = True
                else:        suppress[j] = True
    return [it for k, it in enumerate(items) if not suppress[k]]

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Anti-theft")
    ap.add_argument("--rtsp",  default="rtsp://admin:KOEYHZ@192.168.2.72:554/cam/realmonitor?channel=1&subtype=1")
    ap.add_argument("--video", default="")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--det_model",  required=True)
    ap.add_argument("--pose_model", required=True)
    ap.add_argument("--imgsz",     type=int,   default=640)
    ap.add_argument("--conf_det",  type=float, default=0.5)
    ap.add_argument("--conf_pose", type=float, default=0.4)
    ap.add_argument("--ffmpeg_path",  default="ffmpeg")
    ap.add_argument("--ffprobe_path", default="ffprobe")
    ap.add_argument("--out_width",  type=int,   default=0)
    ap.add_argument("--out_height", type=int,   default=0)
    ap.add_argument("--ffmpeg_fps", type=float, default=12.0)
    ap.add_argument("--timeout_us", type=int,   default=10_000_000)
    ap.add_argument("--pose_fps",   type=float, default=4.0)
    ap.add_argument("--det_fps",    type=float, default=6.0)
    # Anti-theft tuning
    ap.add_argument("--carry_dist",   type=int,   default=30)
    ap.add_argument("--carry_frames", type=int,   default=3)
    ap.add_argument("--move_thresh",  type=int,   default=18)
    ap.add_argument("--cooldown_sec", type=float, default=1.5)
    # Goods zone config
    ap.add_argument("--goods_zone_names", default="zone1,zone2",
                    help="Comma-separated zone names that use goods-theft logic")
    ap.add_argument("--guest_reach_dist", type=int, default=55,
                    help="Wrist-to-item distance (px) counting as guest reaching into zone")
    # Role tracker tuning
    ap.add_argument("--role_confirm_frames", type=int,   default=3,
                    help="Consecutive same-role frames to confirm/switch label")
    ap.add_argument("--role_iou_thresh",     type=float, default=0.25,
                    help="Min IoU to match det_model role box with pose person")
    # Display
    ap.add_argument("--show_breakdown",  action="store_true")
    ap.add_argument("--draw_items",      action="store_true")
    ap.add_argument("--draw_fps",        action="store_true")
    ap.add_argument("--draw_carry_ring", action="store_true")
    args = ap.parse_args()

    src = args.rtsp if args.rtsp else args.video
    if not src: raise RuntimeError("Provide --rtsp or --video")

    zones_norm = load_zones(args.zones)
    if args.out_width > 0 and args.out_height > 0:
        w, h = args.out_width, args.out_height
    else:
        pw, ph = probe_size_with_ffprobe(src, args.ffprobe_path)
        w, h   = (pw, ph) if (pw and ph) else (640, 360)

    det_model  = YOLO(args.det_model)
    pose_model = YOLO(args.pose_model)

    # Only named zones get goods-theft logic; others are display-only
    goods_zone_names = {norm_name(x)
                        for x in args.goods_zone_names.split(",") if x.strip()}
    zone_guards: dict = {}
    for z in zones_norm:
        zname = z["name"]
        if norm_name(zname) in goods_zone_names:
            zg = GoodsZoneGuard(zname)
            zg.COOLDOWN_SEC    = args.cooldown_sec
            zone_guards[zname] = zg
        else:
            zone_guards[zname] = None  # display-only

    conceal = ConcealmentTracker(
        carry_dist=args.carry_dist,
        carry_frames=args.carry_frames,
        move_thresh=args.move_thresh,
    )
    track_buf    = TrackBuffer(ghost_frames=6)
    role_tracker = PersonRoleTracker(
        confirm_frames=args.role_confirm_frames,
        iou_thresh=args.role_iou_thresh,
    )

    reader = FFmpegLatestFrameReader(
        src=src, width=w, height=h, out_fps=args.ffmpeg_fps,
        timeout_us=args.timeout_us, ffmpeg_path=args.ffmpeg_path)
    reader.start()

    win = "Anti-Theft Monitor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_pose_people   = []
    last_det_items_raw = []  # actually detected this frame  -> used for loss logic
    last_det_items_vis = []  # ghost-buffered                -> used for drawing only
    last_role_dets     = []

    next_pose_ts = next_det_ts = 0.0
    disp_last = pose_last = det_last = time.time()
    disp_frames = pose_runs = det_runs = 0
    disp_fps = pose_fps = det_fps = 0.0
    last_seen_fid = 0; last_frame_ts = time.time()

    ROLE_CLASSES = {"staff", "guest"}

    try:
        while True:
            frame, _, fid = reader.get_latest()
            now = time.time()

            if frame is None:
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank, "WAITING FOR STREAM...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                if reader.last_err:
                    cv2.putText(blank, reader.last_err[:160], (30, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, blank)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")): break
                time.sleep(0.01); continue

            vis = frame.copy()
            H, W = vis.shape[:2]

            if fid == last_seen_fid:
                if now - last_frame_ts > 2.0:
                    cv2.putText(vis, "STREAM STALLED", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                last_seen_fid = fid; last_frame_ts = now

            # ── Build pixel zones ──────────────────────────────────────────────
            zones_px = []
            for z in zones_norm:
                pts = [(int(clamp(p[0], 0, 1) * W), int(clamp(p[1], 0, 1) * H))
                       for p in z["points"]]
                zones_px.append({"name": z["name"], "pts": pts})

            # ── YOLOPose ───────────────────────────────────────────────────────
            pose_iv = 1.0 / args.pose_fps if args.pose_fps > 0 else 0.0
            if pose_iv == 0.0 or now >= next_pose_ts:
                next_pose_ts = now + pose_iv
                pres = pose_model.predict(vis, imgsz=args.imgsz,
                                          conf=args.conf_pose, verbose=False)
                people = []
                if pres:
                    r = pres[0]
                    if r.boxes is not None and r.keypoints is not None and len(r.boxes) > 0:
                        for i in range(len(r.boxes)):
                            people.append({
                                "bbox": r.boxes.xyxy[i].cpu().numpy().astype(float),
                                "kpts": r.keypoints.xy[i].cpu().numpy().astype(float),
                                "role":      "unknown",
                                "role_conf": 0.0,
                            })
                if people:
                    role_tracker.update(people, last_role_dets, now)
                last_pose_people = people
                pose_runs += 1
                if now - pose_last >= 1.0:
                    pose_fps  = pose_runs / (now - pose_last)
                    pose_runs = 0; pose_last = now

            # ── Draw pose + role label ─────────────────────────────────────────
            _fsr = max(0.30, min(0.50, H / 800))
            for p in last_pose_people:
                role  = p.get("role",      "unknown")
                rconf = p.get("role_conf", 0.0)
                color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
                x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                draw_pose_skeleton(vis, p["kpts"], color=color)
                lbl = role.upper()
                if role != "unknown" and rconf > 0:
                    lbl += f" {rconf:.0%}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, _fsr, 1)
                lx = x1; ly = max(th + 4, y1 - 4)
                cv2.rectangle(vis, (lx, ly - th - 3), (lx + tw + 6, ly + 2), (0,0,0), -1)
                cv2.putText(vis, lbl, (lx + 2, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, _fsr, color, 1, cv2.LINE_AA)
                for ki in (KP_L_WRIST, KP_R_WRIST):
                    wx, wy = p["kpts"][ki]
                    if wx > 1 and wy > 1:
                        cv2.circle(vis, (int(wx), int(wy)), 6, (0, 255, 255), 2, cv2.LINE_AA)
                        if args.draw_carry_ring:
                            cv2.circle(vis, (int(wx), int(wy)),
                                       conceal.carry_dist, (0, 220, 180), 1, cv2.LINE_AA)

            # ── YOLO det_model tracking (items + staff/guest) ──────────────────
            det_iv = 1.0 / args.det_fps if args.det_fps > 0 else 0.0
            if det_iv == 0.0 or now >= next_det_ts:
                next_det_ts = now + det_iv
                try:
                    dres = det_model.track(vis, imgsz=args.imgsz, conf=args.conf_det,
                                           persist=True, verbose=False, iou=0.3)
                except Exception:
                    dres = det_model.predict(vis, imgsz=args.imgsz,
                                             conf=args.conf_det, verbose=False)

                items_raw = []
                role_dets = []

                if dres:
                    r = dres[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        clss  = r.boxes.cls.cpu().numpy().astype(int)
                        names = r.names or {}
                        tids  = (r.boxes.id.cpu().numpy().astype(int)
                                 if r.boxes.id is not None else [None] * len(boxes))
                        for i in range(len(boxes)):
                            name  = str(names.get(int(clss[i]), clss[i]))
                            lname = name.lower()
                            bbox  = boxes[i].astype(float)
                            cx, cy = xyxy_center(bbox)

                            if lname in ROLE_CLASSES:
                                role_dets.append({
                                    "bbox": bbox,
                                    "name": lname,
                                    "conf": float(confs[i]),
                                })
                            elif lname == "item":
                                items_raw.append({
                                    "bbox":     bbox,
                                    "center":   (cx, cy),
                                    "conf":     float(confs[i]),
                                    "name":     name,
                                    "track_id": tids[i],
                                })
                            # else: unknown class — skip

                items_raw = filter_contained_boxes(items_raw, iou_min_thresh=0.6)

                # RAW  = what is actually visible right now — used for loss logic
                last_det_items_raw = items_raw
                # VIS  = ghost-buffered — for smooth drawing only
                last_det_items_vis = track_buf.update([dict(it) for it in items_raw])

                last_role_dets = role_dets

                if last_pose_people:
                    role_tracker.update(last_pose_people, role_dets, now)

                det_runs += 1
                if now - det_last >= 1.0:
                    det_fps  = det_runs / (now - det_last)
                    det_runs = 0; det_last = now

            # ── Concealment (use RAW — ghost items are already "gone") ─────────
            conceal.update(last_det_items_raw, last_pose_people, now)
            carried_ids = conceal.being_carried_ids()

            # ── Goods zone guard update ────────────────────────────────────────
            for zd in zones_px:
                zname = zd["name"]
                pts   = zd["pts"]
                guard = zone_guards.get(zname)
                if guard is None:
                    continue  # display-only zone

                zone_items_raw     = items_inside_zone(last_det_items_raw, pts)
                visible_item_count = len(zone_items_raw)

                guest_interacting = any(
                    person_role_is(p, "guest") and
                    person_interacts_with_goods_zone(
                        p, pts, zone_items_raw,
                        reach_dist=args.guest_reach_dist)
                    for p in last_pose_people
                )
                staff_interacting = any(
                    person_role_is(p, "staff") and
                    person_interacts_with_goods_zone(
                        p, pts, zone_items_raw,
                        reach_dist=args.guest_reach_dist)
                    for p in last_pose_people
                )

                guard.update(
                    guest_interacting  = guest_interacting,
                    visible_item_count = visible_item_count,
                    now                = now,
                    staff_interacting  = staff_interacting,
                )

            # ── Draw zones ─────────────────────────────────────────────────────
            _fsl = max(0.35, min(0.50, H / 900))
            _fsc = max(0.50, min(0.75, H / 700))

            for zd in zones_px:
                zname = zd["name"]
                poly  = np.array(zd["pts"], dtype=np.int32)
                guard = zone_guards.get(zname)

                zone_items_raw = items_inside_zone(last_det_items_raw, zd["pts"])
                visible_now    = len(zone_items_raw)

                if guard is not None and guard.is_alerting:
                    zc = (0, 0, 255);   al = 0.18
                elif guard is not None and guard.person_blocking:
                    zc = (0, 140, 255); al = 0.10
                else:
                    zc = (0, 255, 255); al = 0.06

                ov = vis.copy(); cv2.fillPoly(ov, [poly], zc)
                cv2.addWeighted(ov, al, vis, 1 - al, 0, vis)
                cv2.polylines(vis, [poly], True, zc, 1)

                if guard is not None and guard.person_blocking:
                    shown = guard.baseline_count; sfx = " [FROZEN]"
                elif guard is not None and guard.is_alerting:
                    shown = max(0, guard.baseline_count - guard.missing)
                    sfx   = f" [-{guard.missing}!]"
                else:
                    shown = visible_now; sfx = ""

                tx, ty = poly_label_anchor(zd["pts"]); ty = max(14, ty + 16)

                if args.show_breakdown:
                    guest_cnt = sum(
                        1 for p in last_pose_people
                        if person_role_is(p, "guest") and
                        person_interacts_with_goods_zone(
                            p, zd["pts"], zone_items_raw,
                            reach_dist=args.guest_reach_dist)
                    )
                    staff_cnt = sum(
                        1 for p in last_pose_people
                        if person_role_is(p, "staff") and
                        person_interacts_with_goods_zone(
                            p, zd["pts"], zone_items_raw,
                            reach_dist=args.guest_reach_dist)
                    )
                    lbl = f"{zname} G={guest_cnt} S={staff_cnt} I={shown}{sfx}"
                else:
                    lbl = f"{zname} I={shown}{sfx}"

                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, _fsl, 1)
                cv2.rectangle(vis,
                              (max(0, tx-3), max(0, ty-th-3)),
                              (min(W-1, tx+tw+6), min(H-1, ty+3)), (0,0,0), -1)
                cv2.putText(vis, lbl, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, _fsl, zc, 1, cv2.LINE_AA)
                big = str(shown)
                (bw, bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, _fsc, 2)
                cy_ = min(H - 6, ty + bh + 8)
                cv2.rectangle(vis, (tx-2, cy_-bh-3),
                              (min(W-1, tx+bw+6), cy_+3), (0,0,0), -1)
                cv2.putText(vis, big, (tx+2, cy_),
                            cv2.FONT_HERSHEY_SIMPLEX, _fsc, zc, 2, cv2.LINE_AA)

            # ── Draw items (VIS buffer for smooth display) ─────────────────────
            if args.draw_items:
                _fsi = max(0.30, min(0.42, H / 900))
                for it in last_det_items_vis:
                    x1, y1, x2, y2 = [int(v) for v in it["bbox"]]
                    tid = it.get("track_id")
                    if tid in carried_ids:
                        color = (0, 0, 255)
                    elif any(point_in_poly(it["center"], zd["pts"]) for zd in zones_px):
                        color = (0, 220, 0)
                    else:
                        color = (0, 128, 255)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                    lbl = f"#{tid}" if tid is not None else ""
                    if tid in carried_ids: lbl += " CARRIED!"
                    lbl += f" {it['conf']:.2f}"
                    (iw, ih), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, _fsi, 1)
                    lx, ly = x1, max(ih + 2, y1 - 2)
                    cv2.rectangle(vis, (lx, ly-ih-2),
                                  (min(W-1, lx+iw+4), ly+2), (0,0,0), -1)
                    cv2.putText(vis, lbl, (lx+2, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, _fsi, color, 1, cv2.LINE_AA)

                for rd in last_role_dets:
                    x1, y1, x2, y2 = [int(v) for v in rd["bbox"]]
                    rc = ROLE_COLORS.get(rd["name"], (200, 200, 200))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), rc, 1)
                    cv2.putText(vis, f"[det]{rd['name']} {rd['conf']:.2f}",
                                (x1, max(10, y1-2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, rc, 1, cv2.LINE_AA)

            # ── Alerts ─────────────────────────────────────────────────────────
            draw_alerts(
                vis,
                [g for g in zone_guards.values() if g is not None],
                conceal,
                now,
            )

            # ── FPS overlay ────────────────────────────────────────────────────
            disp_frames += 1
            if now - disp_last >= 1.0:
                disp_fps    = disp_frames / (now - disp_last)
                disp_frames = 0; disp_last = now
            if args.draw_fps:
                txt = (f"DISP {disp_fps:.1f} | POSE {pose_fps:.1f} "
                       f"| DET {det_fps:.1f} | IN {args.ffmpeg_fps:.1f}")
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(vis, (6, 4), (tw + 14, th + 12), (0, 0, 0), -1)
                cv2.putText(vis, txt, (10, th + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")): break

    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()