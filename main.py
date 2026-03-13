# Features:
#   1. Zone-based goods theft detection (GoodsZoneGuard)
#   2. Concealment detection — ZONE-ONLY, guest-only, snapshot-at-carry
#   3. Staff/Guest role labeling
#      [NEW] Sliding-window role confirmation: only 2 detections (non-consecutive)
#            within candidate_window_s seconds needed to confirm a role.
#            Label sticks; only switches if the new role also gets enough hits.
#   4. Home Assistant: ONE sensor per goods zone only
#   5. [NEW] Pinned zone items: bbox+ID displayed forever while in zone,
#            even when occluded. Only removed after theft confirmed.
#
# ── CHANGE LOG ─────────────────────────────────────────────────────────────
#   [FIX] PersonRoleTracker: sliding-window confirmation (confirm_count=2,
#         candidate_window_s=8s) → works even with intermittent det_model
#   [FIX] StableItemTracker: items inside goods zones are "pinned" →
#         their bbox+ID is displayed FOREVER (infinite ghost TTL) until:
#           a) re-detected after person leaves (still there → keep)
#           b) explicitly removed by remove_zone_ghosts() after theft confirmed
#         Non-zone items keep the normal ghost_ttl behaviour
#   [KEEP] All other logic unchanged (concealment, GoodsZoneGuard, HA, etc.)
# ────────────────────────────────────────────────────────────────────────────

import os, json, time, argparse, subprocess, threading, re, collections
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    print("[HA] 'requests' not installed — pip install requests — HA integration disabled.")


# ─────────────────────────────────────────────────────────────
# Home Assistant Notifier
# ─────────────────────────────────────────────────────────────
class HomeAssistantNotifier:
    SENSOR_PREFIX = "sensor.antitheft_"
    MAX_SNAPSHOT_FILES = 100

    def __init__(self, ha_url, token, zone_names=None, webhook_id="",
                 snapshot_dir="", clip_duration_s=0.0, notify_cooldown_s=15.0):
        self.ha_url            = ha_url.rstrip("/")
        self.webhook_id        = webhook_id
        self.snapshot_dir      = snapshot_dir
        self.clip_duration_s   = clip_duration_s
        self.notify_cooldown_s = notify_cooldown_s
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        self._last_notify: dict = {}
        self._staged_media: dict = {}
        if snapshot_dir and not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        self._frame_buf: collections.deque = collections.deque()
        self._frame_buf_lock = threading.Lock()
        self._clip_buf_sec   = max(clip_duration_s, 5.0)
        self._enabled = _HAS_REQUESTS and bool(ha_url) and bool(token)
        if not self._enabled:
            print("[HA] Notifier disabled (missing url/token or requests lib)")
            return
        self._init_sensors(zone_names or [])

    def _init_sensors(self, zone_names):
        for z in zone_names:
            entity_id = self.SENSOR_PREFIX + re.sub(r"[^a-zA-Z0-9_]", "_",
                                                     f"zone_{z}".lower())
            attrs = {"friendly_name": f"Anti-Theft Zone: {z}",
                     "message": "", "timestamp": "", "snapshot": "", "clip": ""}
            self._update_sensor(entity_id, "None", attrs)
            print(f"[HA] Initialized sensor -> {entity_id} = None")

    def push_frame(self, frame, ts):
        if self.clip_duration_s <= 0: return
        with self._frame_buf_lock:
            self._frame_buf.append((frame, ts))
            cutoff = ts - self._clip_buf_sec
            while self._frame_buf and self._frame_buf[0][1] < cutoff:
                self._frame_buf.popleft()

    def _prune_snapshot_files(self, keep_paths=None):
        if not self.snapshot_dir:
            return
        keep = {os.path.abspath(p) for p in (keep_paths or []) if p}
        try:
            files = []
            for fn in os.listdir(self.snapshot_dir):
                if not fn.lower().endswith(".jpg"):
                    continue
                fp = os.path.abspath(os.path.join(self.snapshot_dir, fn))
                try:
                    mtime = os.path.getmtime(fp)
                except OSError:
                    continue
                files.append((mtime, fp))
            files.sort(key=lambda x: x[0])
            overflow = max(0, len(files) - self.MAX_SNAPSHOT_FILES)
            removed = 0
            for _mtime, fp in files:
                if removed >= overflow:
                    break
                if fp in keep:
                    continue
                try:
                    os.remove(fp)
                    removed += 1
                except OSError:
                    pass
        except Exception as e:
            print(f"[HA] Snapshot prune error: {e}")

    def _save_snapshot(self, frame, label):
        if not self.snapshot_dir or frame is None: return ""
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        safe   = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        path   = os.path.join(self.snapshot_dir, f"{safe}_{ts_str}.jpg")
        cv2.imwrite(path, frame)
        self._prune_snapshot_files(keep_paths=[path])
        return path

    def stage_snapshot(self, kind, frame):
        if not self.snapshot_dir or frame is None:
            return ""
        staged = self._staged_media.get(kind)
        if staged:
            snap_path = staged.get("snapshot", "")
            if snap_path and os.path.exists(snap_path):
                return snap_path
        snap_path = self._save_snapshot(frame, f"{kind}_pre")
        if snap_path:
            self._staged_media[kind] = {
                "snapshot": snap_path,
                "ts": time.time(),
            }
            print(f"[HA] Staged snapshot -> {kind}  snap={snap_path}")
        return snap_path

    def discard_staged(self, kind):
        staged = self._staged_media.pop(kind, None)
        if not staged:
            return
        snap_path = staged.get("snapshot", "")
        if snap_path and os.path.exists(snap_path):
            try:
                os.remove(snap_path)
                print(f"[HA] Discarded staged snapshot -> {kind}  snap={snap_path}")
            except OSError:
                pass

    def _consume_staged_snapshot(self, kind):
        staged = self._staged_media.pop(kind, None)
        if not staged:
            return ""
        snap_path = staged.get("snapshot", "")
        return snap_path if snap_path and os.path.exists(snap_path) else ""

    def _save_clip(self, label):
        if self.clip_duration_s <= 0 or not self.snapshot_dir: return ""
        with self._frame_buf_lock:
            frames = list(self._frame_buf)
        if not frames: return ""
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        safe   = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        path   = os.path.join(self.snapshot_dir, f"{safe}_{ts_str}.mp4")
        h, w   = frames[0][0].shape[:2]
        fps    = max(1.0, len(frames) / max(0.1, frames[-1][1] - frames[0][1])) if len(frames) > 1 else 10.0
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f, _ in frames: writer.write(f)
        writer.release()
        return path

    def _update_sensor(self, entity_id, state, attributes):
        url = f"{self.ha_url}/api/states/{entity_id}"
        try:
            r = _requests.post(url, headers=self._headers,
                               json={"state": state, "attributes": attributes}, timeout=5)
            if not r.ok:
                print(f"[HA] Sensor update failed {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[HA] Sensor update error: {e}")

    def _fire_webhook(self, data):
        if not self.webhook_id: return
        url = f"{self.ha_url}/api/webhook/{self.webhook_id}"
        try:
            r = _requests.post(url, json=data, timeout=5)
            if not r.ok:
                print(f"[HA] Webhook failed {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[HA] Webhook error: {e}")

    def send_alert(self, kind, message, frame, now, use_staged_snapshot=False,
                   snapshot_path=""):
        if not self._enabled: return
        entity_id = self.SENSOR_PREFIX + re.sub(r"[^a-zA-Z0-9_]", "_", kind.lower())
        last = self._last_notify.get(entity_id, 0.0)
        if (now - last) < self.notify_cooldown_s:
            return
        self._last_notify[entity_id] = now
        snap_path = clip_path = ""

        def _save_media():
            nonlocal snap_path, clip_path
            chosen = ""
            if snapshot_path and os.path.exists(snapshot_path):
                chosen = snapshot_path
            elif use_staged_snapshot:
                chosen = self._consume_staged_snapshot(kind)
            if not chosen:
                chosen = self._save_snapshot(frame, kind)
            snap_path = chosen
            clip_path = self._save_clip(kind)

        t = threading.Thread(target=_save_media, daemon=True)
        t.start(); t.join(timeout=10)
        attrs = {
            "friendly_name": f"Anti-Theft: {kind}", "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "snapshot": snap_path, "clip": clip_path,
        }
        self._update_sensor(entity_id, "ALERT", attrs)
        self._fire_webhook({"type": kind, "message": message,
                            "snapshot_path": snap_path, "clip_path": clip_path,
                            "timestamp": attrs["timestamp"]})
        print(f"[HA] Sent alert -> {entity_id}  snap={snap_path} clip={clip_path}")

    def clear_alert(self, kind):
        self.discard_staged(kind)
        if not self._enabled: return
        entity_id = self.SENSOR_PREFIX + re.sub(r"[^a-zA-Z0-9_]", "_", kind.lower())
        attrs = {"friendly_name": f"Anti-Theft: {kind}", "message": "",
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "snapshot": "", "clip": ""}
        self._update_sensor(entity_id, "OK", attrs)


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

def expand_bbox(bbox, margin):
    x1, y1, x2, y2 = bbox
    m = float(margin)
    return (x1 - m, y1 - m, x2 + m, y2 + m)

def bbox_overlaps_zone(bbox, zone_pts, margin=0.0):
    bb = expand_bbox(bbox, margin) if margin else bbox
    x1, y1, x2, y2 = bb
    sample_pts = [
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
        ((x1 + x2) / 2.0, y1), ((x1 + x2) / 2.0, y2),
        (x1, (y1 + y2) / 2.0), (x2, (y1 + y2) / 2.0),
    ]
    return any(point_in_poly(pt, zone_pts) for pt in sample_pts)

def role_det_present_in_zone(role_detections, zone_pts, role_name=None, margin=0.0):
    want = role_name.lower() if isinstance(role_name, str) else None
    for rd in role_detections or []:
        name = str(rd.get("name", "")).lower()
        if want is not None and name != want:
            continue
        bbox = rd.get("bbox")
        if bbox is not None and bbox_overlaps_zone(bbox, zone_pts, margin=margin):
            return True
    return False

def norm_name(s):
    return str(s).strip().lower().replace(" ", "")

def point_valid(pt):
    return pt[0] > 1 and pt[1] > 1

def items_inside_zone(items, zone_pts):
    return [it for it in items if point_in_poly(it["center"], zone_pts)]

def person_role_is(person, role):
    return person.get("role", "unknown") == role

def item_in_any_zone(item, goods_zones_px):
    for z in goods_zones_px:
        if point_in_poly(item["center"], z["pts"]):
            return z["name"]
    return None


# ─────────────────────────────────────────────────────────────
# Pose skeleton  (COCO-17)
# ─────────────────────────────────────────────────────────────
COCO17_EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
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
# Only WRISTS are allowed to activate a goods zone.
# Body bbox / torso / elbow / shoulder entering the polygon is ignored.
# ─────────────────────────────────────────────────────────────
def wrists_inside_zone(person, zone_pts):
    inside = []
    for ki in (KP_L_WRIST, KP_R_WRIST):
        wx, wy = person["kpts"][ki]
        if point_valid((wx, wy)) and point_in_poly((wx, wy), zone_pts):
            inside.append((float(wx), float(wy)))
    return inside


def person_interacts_with_goods_zone(person, zone_pts, zone_items, reach_dist=55):
    wrists_in_zone = wrists_inside_zone(person, zone_pts)
    if not wrists_in_zone:
        return False

    # A wrist entering the zone is already considered an interaction.
    # If items are visible, keep the near-item check so the logic remains
    # compatible with hand-near-object detection, but never let bbox/body
    # presence trigger the zone by itself.
    if not zone_items:
        return True

    for wx, wy in wrists_in_zone:
        for it in zone_items:
            if dist2((wx, wy), it["center"]) <= reach_dist ** 2:
                return True

    return True


# ─────────────────────────────────────────────────────────────
# Person Role Tracker  — SLIDING WINDOW CONFIRMATION
#
# Problem with the old approach (confirm_frames consecutive):
#   det_model detects staff/guest in short bursts with gaps between
#   them (low conf, motion blur, occlusion). Requiring N CONSECUTIVE
#   frames means the label often never gets set.
#
# New approach — sliding window:
#   Any role that accumulates >= confirm_count detections within the
#   last candidate_window_s seconds gets confirmed, regardless of
#   whether those detections were consecutive.
#
# Stickiness:
#   Once confirmed, the label stays until a DIFFERENT role accumulates
#   confirm_count hits within the window — preventing flicker from
#   single stray detections.
# ─────────────────────────────────────────────────────────────
class PersonRoleTracker:
    def __init__(self,
                 confirm_count: int    = 2,
                 candidate_window_s: float = 8.0,
                 iou_thresh: float     = 0.25,
                 max_age_s: float      = 3.0,
                 person_match_iou: float    = 0.15,
                 person_match_center: float = 120.0,
                 switch_confirm_count=None,
                 switch_min_conf: float = 0.78,
                 switch_margin: int = 2,
                 # legacy kwargs kept so old call-sites don't break
                 confirm_frames: int   = 2,
                 history_len: int      = 10,
                 candidate_ttl_s: float = 1.0):
        self.confirm_count        = confirm_count
        self.candidate_window_s   = candidate_window_s
        self.iou_thresh           = iou_thresh
        self.max_age_s            = max_age_s
        self.person_match_iou     = person_match_iou
        self.person_match_center  = person_match_center
        self.switch_confirm_count = int(switch_confirm_count) if switch_confirm_count is not None else max(int(confirm_count) + 2, 6)
        self.switch_min_conf      = float(switch_min_conf)
        self.switch_margin        = int(switch_margin)
        self._tracks: list        = []
        self._next_track_id: int  = 1

    # ── Spatial matching helpers (unchanged from original) ──────────────
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

    def update(self, persons, role_detections, now):
        # 1. Match persons to tracks
        track_assign = self._match_persons_to_tracks(persons)
        for pi, p in enumerate(persons):
            ti = track_assign[pi]
            if ti is None:
                self._tracks.append({
                    "track_id":       self._next_track_id,
                    "bbox":           p["bbox"],
                    "confirmed_role": "unknown",
                    "confirmed_conf": 0.0,
                    # Sliding window history: deque of (timestamp, role, conf)
                    "det_history":    collections.deque(maxlen=60),
                    "last_seen_ts":   now,
                })
                self._next_track_id += 1
                ti = len(self._tracks) - 1
            else:
                self._tracks[ti]["bbox"]         = p["bbox"]
                self._tracks[ti]["last_seen_ts"] = now
            p["_track_idx"] = ti

        # 2. Match role detections to persons
        role_assign = self._match_role_dets_to_persons(persons, role_detections)

        # 3. Update sliding window + confirm role
        for pi, p in enumerate(persons):
            ti    = p["_track_idx"]
            track = self._tracks[ti]

            # Append new detection hit if matched this frame
            ri = role_assign[pi]
            if ri is not None:
                rd = role_detections[ri]
                track["det_history"].append(
                    (now, rd["name"], float(rd["conf"]))
                )

            # Prune entries outside the window
            cutoff = now - self.candidate_window_s
            track["det_history"] = collections.deque(
                [(t, r, c) for t, r, c in track["det_history"] if t >= cutoff],
                maxlen=60
            )

            # Count hits per role inside the window
            role_counts:    dict = {}
            role_conf_sums: dict = {}
            for _t, r, c in track["det_history"]:
                role_counts[r]    = role_counts.get(r, 0) + 1
                role_conf_sums[r] = role_conf_sums.get(r, 0.0) + c

            # Best role = most hits, must reach confirm_count threshold
            best_role  = None
            best_count = 0
            for r, cnt in role_counts.items():
                if cnt >= self.confirm_count and cnt > best_count:
                    best_count = cnt
                    best_role  = r

            if best_role is not None:
                cur       = track["confirmed_role"]
                cur_count = role_counts.get(cur, 0)
                best_avg_conf = role_conf_sums[best_role] / max(1, best_count)

                if cur == "unknown":
                    track["confirmed_role"] = best_role
                    track["confirmed_conf"] = best_avg_conf
                elif cur == best_role:
                    track["confirmed_conf"] = best_avg_conf
                else:
                    enough_hits   = best_count >= self.switch_confirm_count
                    enough_margin = best_count >= (cur_count + self.switch_margin)
                    enough_conf   = best_avg_conf >= self.switch_min_conf
                    if enough_hits and enough_margin and enough_conf:
                        print(f"[ROLE] track#{track['track_id']} "
                              f"{cur} -> {best_role} "
                              f"(hits={best_count}, avg_conf={best_avg_conf:.2f})")
                        track["confirmed_role"] = best_role
                        track["confirmed_conf"] = best_avg_conf

            p["role"]      = track["confirmed_role"]
            p["role_conf"] = track["confirmed_conf"]
            del p["_track_idx"]

        # 4. Expire stale tracks
        self._tracks = [t for t in self._tracks
                        if (now - t["last_seen_ts"]) < self.max_age_s]
        return persons


# ─────────────────────────────────────────────────────────────
# StableItemTracker — PINNED ZONE ITEMS
#
# Items detected inside goods zones become "pinned":
#   - Pinned items are NEVER auto-expired by ghost_ttl.
#   - When occluded (det_model stops seeing them), they stay in the
#     stable_items list as ghost=True, pinned=True.  Their bbox and ID
#     are drawn on screen with a dashed border.
#   - When re-detected after occlusion ends, they are refreshed
#     (ghost=False) by the spatial proximity match as usual.
#   - They are only removed when remove_zone_ghosts() is called
#     explicitly after theft is confirmed by GoodsZoneGuard.
#
# Non-zone items: unchanged — expire after ghost_ttl seconds.
# ─────────────────────────────────────────────────────────────
class StableItemTracker:
    def __init__(self, pos_thresh: float = 45.0, ghost_ttl: float = 3.0):
        self.pos_thresh = float(pos_thresh)
        self.ghost_ttl  = float(ghost_ttl)
        self._reg: dict = {}
        self._next_sid  = 1

    def _make_rec(self, it: dict, now: float) -> dict:
        return {
            "center":    it["center"],
            "bbox":      list(it["bbox"]),
            "name":      it.get("name", "item"),
            "conf":      float(it.get("conf", 0.5)),
            "last_seen": now,
            "ghost":     False,
            "pinned":    False,   # True = zone item, never auto-expires
        }

    def _best_pos_match(self, cx: float, cy: float, used: set):
        best_sid  = None
        best_dist = float("inf")
        for sid, rec in self._reg.items():
            if sid in used:
                continue
            rx, ry = rec["center"]
            d = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
            if d < self.pos_thresh and d < best_dist:
                best_dist = d
                best_sid  = sid
        return best_sid

    def update(self, detected: list, now: float,
               goods_zones_px: list = None):
        """
        detected       : raw items from det_model this frame
        now            : time.time()
        goods_zones_px : pixel-space goods zones; items inside these zones
                         get pinned (infinite ghost TTL)

        Returns (stable_items, real_items):
          stable_items : real + ghost/pinned  (zone count + display)
          real_items   : only detected         (theft check + concealment)
        """
        used_sids  = set()
        real_items = []

        # ── Phase 1: match detections to registry ─────────────────────
        for it in detected:
            cx, cy = it["center"]
            sid = self._best_pos_match(cx, cy, used_sids)

            if sid is None:
                tid = it.get("track_id")
                if tid is not None and tid in self._reg and tid not in used_sids:
                    sid = tid

            if sid is not None:
                rec = self._reg[sid]
                rec["center"]    = (cx, cy)
                rec["bbox"]      = list(it["bbox"])
                rec["conf"]      = float(it.get("conf", rec["conf"]))
                rec["last_seen"] = now
                rec["ghost"]     = False
            else:
                sid = self._next_sid
                self._next_sid += 1
                self._reg[sid] = self._make_rec(it, now)

            # Pin if inside a goods zone
            if goods_zones_px:
                rec = self._reg[sid]
                if not rec["pinned"]:
                    for z in goods_zones_px:
                        if point_in_poly(rec["center"], z["pts"]):
                            rec["pinned"] = True
                            break

            used_sids.add(sid)
            new_it             = dict(it)
            new_it["track_id"] = sid
            new_it["ghost"]    = False
            new_it["pinned"]   = self._reg[sid]["pinned"]
            real_items.append(new_it)

        # ── Phase 2: ghost or expire unmatched entries ─────────────────
        ghost_items = []
        for sid in list(self._reg.keys()):
            if sid in used_sids:
                continue
            rec = self._reg[sid]
            age = now - rec["last_seen"]

            if rec["pinned"]:
                # Pinned zone item: NEVER expires → always ghost-drawn
                rec["ghost"] = True
                ghost_items.append({
                    "bbox":     rec["bbox"],
                    "center":   rec["center"],
                    "conf":     rec["conf"],
                    "name":     rec["name"],
                    "track_id": sid,
                    "ghost":    True,
                    "pinned":   True,
                })
            elif age > self.ghost_ttl:
                del self._reg[sid]
            else:
                rec["ghost"] = True
                ghost_items.append({
                    "bbox":     rec["bbox"],
                    "center":   rec["center"],
                    "conf":     rec["conf"],
                    "name":     rec["name"],
                    "track_id": sid,
                    "ghost":    True,
                    "pinned":   False,
                })

        return real_items + ghost_items, real_items

    def remove_zone_ghosts(self, zone_pts: list):
        """Remove ALL ghost items (including pinned) from this zone.
        Called after GoodsZoneGuard confirms theft."""
        for sid in list(self._reg.keys()):
            rec = self._reg[sid]
            if rec["ghost"] and point_in_poly(rec["center"], zone_pts):
                del self._reg[sid]


# ─────────────────────────────────────────────────────────────
# Goods Zone Guard  (unchanged logic)
# ─────────────────────────────────────────────────────────────
class GoodsZoneGuard:
    IDLE           = 0
    GUEST_INTERACT = 1
    OCCLUDED_HOLD  = 2
    COOLDOWN       = 3
    ALERT          = 4

    EXIT_SETTLE_SEC = 2.5
    COOLDOWN_SEC    = 1.5
    ALERT_DISPLAY   = 10.0

    def __init__(self, name):
        self.name  = name
        self.state = self.IDLE
        self.baseline_count = 0
        self.clear_counts   = collections.deque(maxlen=8)
        self.exit_ts   = 0.0
        self.alert_ts  = 0.0
        self.alert_msg = ""
        self.missing   = 0
        self.had_guest_contact           = False
        self.staff_touched_during_window = False
        self._staff_in_zone: bool     = False
        self.staff_reset_needed: bool = False
        self._cooldown_max_raw: int       = 0
        self._cooldown_clear_since: float = 0.0
        self._events: list                = []

    def _emit(self, event_name: str):
        self._events.append(event_name)

    def pop_events(self):
        out = list(self._events)
        self._events.clear()
        return out

    def _update_clear_baseline(self, count: int):
        self.clear_counts.append(int(count))
        self.baseline_count = (
            int(round(float(np.median(self.clear_counts))))
            if self.clear_counts else int(count)
        )

    def _reset_window(self):
        self.had_guest_contact           = False
        self.staff_touched_during_window = False
        self._cooldown_max_raw           = 0
        self._cooldown_clear_since       = 0.0
        self.exit_ts                     = 0.0

    def _go_idle(self, count: int, discard_snapshot: bool = True):
        self.state = self.IDLE
        self.clear_counts.clear()
        self._update_clear_baseline(count)
        self._reset_window()
        if discard_snapshot:
            self._emit("discard_snapshot")

    def _start_cooldown(self, now: float):
        self.exit_ts = now
        self._cooldown_max_raw = 0
        self._cooldown_clear_since = 0.0
        self.state = self.COOLDOWN

    def force_rebaseline(self, raw_item_count: int):
        """Called by main loop after staff leaves zone.
        Clears ALL pinned ghosts (already done externally) and re-builds
        the baseline from the actual detected item count."""
        self.clear_counts.clear()
        self._update_clear_baseline(raw_item_count)
        self.staff_reset_needed = False
        self.state = self.IDLE
        self._reset_window()
        self._emit("discard_snapshot")
        print(f"[ZONE {self.name}] Staff left → re-baseline = {raw_item_count} items")

    def update(self, guest_interacting: bool, stable_item_count: int,
               now: float, staff_interacting: bool = False,
               raw_item_count: int = None,
               cooldown_blocked: bool = False,
               guest_present: bool = False,
               staff_present: bool = False,
               unknown_present: bool = False):
        stable_item_count = int(stable_item_count)
        check_count = int(raw_item_count) if raw_item_count is not None else stable_item_count

        zone_occupied = bool(guest_present or staff_present or unknown_present or cooldown_blocked)

        staff_now_present = bool(staff_present or staff_interacting)
        staff_just_left = self._staff_in_zone and not staff_now_present
        if staff_just_left:
            self.staff_reset_needed = True
            self.state = self.IDLE
            self._reset_window()
            self._emit("discard_snapshot")
        self._staff_in_zone = staff_now_present

        if self.state == self.IDLE:
            self._update_clear_baseline(stable_item_count)
            if guest_interacting:
                self.had_guest_contact = True
                self.state = self.GUEST_INTERACT
                self._emit("stage_snapshot")

        elif self.state == self.GUEST_INTERACT:
            if not guest_interacting:
                if zone_occupied:
                    self.state = self.OCCLUDED_HOLD
                else:
                    self._start_cooldown(now)

        elif self.state == self.OCCLUDED_HOLD:
            if guest_interacting:
                self.state = self.GUEST_INTERACT
            elif not zone_occupied:
                self._start_cooldown(now)

        elif self.state == self.COOLDOWN:
            if guest_interacting:
                self._cooldown_max_raw = 0
                self._cooldown_clear_since = 0.0
                self.state = self.GUEST_INTERACT
            elif zone_occupied:
                self._cooldown_max_raw = 0
                self._cooldown_clear_since = 0.0
                self.state = self.OCCLUDED_HOLD
            else:
                if (now - self.exit_ts) < self.EXIT_SETTLE_SEC:
                    pass
                else:
                    if self._cooldown_clear_since <= 0.0:
                        self._cooldown_clear_since = now
                        self._cooldown_max_raw = check_count
                    else:
                        self._cooldown_max_raw = max(self._cooldown_max_raw, check_count)

                    if (now - self._cooldown_clear_since) >= self.COOLDOWN_SEC:
                        best_count = self._cooldown_max_raw
                        diff = self.baseline_count - best_count
                        if self.had_guest_contact and diff > 0:
                            self.missing   = diff
                            self.alert_msg = (
                                f"GOODS TAKEN [{self.name}]  "
                                f"guest left zone -> "
                                f"{self.baseline_count} -> {best_count} items"
                            )
                            self.alert_ts = now
                            self.state    = self.ALERT
                            self._emit("commit_snapshot")
                            print(f"[ZONE {self.name}] {self.alert_msg}")
                        else:
                            self._go_idle(stable_item_count)

        elif self.state == self.ALERT:
            if guest_interacting:
                self.clear_counts.clear()
                self._update_clear_baseline(stable_item_count)
                self.had_guest_contact = True
                self._cooldown_max_raw = 0
                self._cooldown_clear_since = 0.0
                self.state = self.GUEST_INTERACT
                self._emit("stage_snapshot")
            elif (now - self.alert_ts) > self.ALERT_DISPLAY:
                self._go_idle(stable_item_count)

    @property
    def is_alerting(self):     return self.state == self.ALERT
    @property
    def person_blocking(self): return self.state in (self.GUEST_INTERACT, self.OCCLUDED_HOLD, self.COOLDOWN)


# ─────────────────────────────────────────────────────────────
# Concealment Tracker  (ZONE-ONLY, GUEST-ONLY — unchanged logic)
# ─────────────────────────────────────────────────────────────
class ConcealmentTracker:
    ALERT_DISPLAY = 10.0

    def __init__(self, carry_dist=90, carry_frames=3, move_thresh=18,
                 unknown_as_guest=True, grasp_dist=24,
                 bbox_grab_margin=8, conceal_verify_sec=1.2,
                 missing_keepalive_sec=8.0, contact_ttl_sec=1.2):
        self.carry_dist            = carry_dist
        self.carry_frames          = carry_frames
        self.move_thresh           = move_thresh
        self.unknown_as_guest      = unknown_as_guest
        self.grasp_dist            = grasp_dist
        self.bbox_grab_margin      = bbox_grab_margin
        self.conceal_verify_sec    = conceal_verify_sec
        self.missing_keepalive_sec = missing_keepalive_sec
        self.contact_ttl_sec       = contact_ttl_sec
        self._items: dict          = {}
        self.alerts: list          = []

    def _is_guest(self, role: str) -> bool:
        if role == "guest": return True
        if role == "unknown" and self.unknown_as_guest: return True
        return False

    def _is_staff(self, role: str) -> bool:
        return role == "staff"

    def _get_zone_pts(self, zone_name: str, goods_zones_px: list):
        for z in goods_zones_px:
            if z["name"] == zone_name:
                return z["pts"]
        return None

    def _zone_presence(self, zone_pts, people, role_detections):
        guest_present = False
        staff_present = False
        human_present = False

        for p in people or []:
            bbox = p.get("bbox")
            if bbox is None or not bbox_overlaps_zone(bbox, zone_pts):
                continue
            human_present = True
            role = p.get("role", "unknown")
            if self._is_guest(role):
                guest_present = True
            elif self._is_staff(role):
                staff_present = True

        for rd in role_detections or []:
            bbox = rd.get("bbox")
            if bbox is None or not bbox_overlaps_zone(bbox, zone_pts):
                continue
            human_present = True
            name = str(rd.get("name", "")).lower()
            if name == "guest":
                guest_present = True
            elif name == "staff":
                staff_present = True

        return guest_present, staff_present, human_present

    def update(self, items: list, people: list,
               goods_zones_px: list, now: float, frame=None,
               role_detections=None):
        role_detections = role_detections or []
        current_ids = {it["track_id"] for it in items
                       if it.get("track_id") is not None}

        for it in items:
            tid = it.get("track_id")
            if tid is None:
                continue
            cx, cy = it["center"]
            item_zone_name = item_in_any_zone(it, goods_zones_px)
            if item_zone_name is None and tid not in self._items:
                continue

            near_wrist   = False
            strong_near  = False
            carrier_role = "unknown"
            item_zone_pts = None
            if item_zone_name is not None:
                item_zone_pts = self._get_zone_pts(item_zone_name, goods_zones_px)
            expanded_box = expand_bbox(it["bbox"], self.bbox_grab_margin)
            best_d2 = None
            for p in people or []:
                role = p.get("role", "unknown")
                for ki in (KP_L_WRIST, KP_R_WRIST):
                    wx, wy = p["kpts"][ki]
                    if not point_valid((wx, wy)):
                        continue
                    if item_zone_pts is not None and not point_in_poly((wx, wy), item_zone_pts):
                        continue
                    d2 = dist2((cx, cy), (wx, wy))
                    if d2 < self.carry_dist ** 2:
                        near_wrist = True
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2
                            carrier_role = role
                        if d2 <= self.grasp_dist ** 2 or bbox_contains_point(expanded_box, (wx, wy)):
                            strong_near = True

            if tid not in self._items:
                self._items[tid] = {
                    "name":               it.get("name", "item"),
                    "near_frames":        0,
                    "being_carried":      False,
                    "approach_confirmed": False,
                    "grab_hint":          False,
                    "carrier_role":       "unknown",
                    "last_seen_ts":       now,
                    "last_contact_ts":    0.0,
                    "missing_since":      0.0,
                    "anchor_pos":         (cx, cy),
                    "has_moved":          False,
                    "zone_name":          item_zone_name,
                    "carry_snapshot":     None,
                }
            rec = self._items[tid]
            rec["last_seen_ts"] = now
            rec["name"]         = it.get("name", rec["name"])
            rec["missing_since"] = 0.0
            if item_zone_name is not None:
                rec["zone_name"] = item_zone_name

            if near_wrist and self._is_staff(carrier_role):
                rec["near_frames"]        = 0
                rec["being_carried"]      = False
                rec["approach_confirmed"] = False
                rec["grab_hint"]          = False
                rec["has_moved"]          = False
                rec["carry_snapshot"]     = None
                rec["missing_since"]      = 0.0
            elif near_wrist and self._is_guest(carrier_role):
                if rec["near_frames"] == 0:
                    rec["anchor_pos"]         = (cx, cy)
                    rec["has_moved"]          = False
                    rec["approach_confirmed"] = False
                    rec["grab_hint"]          = False
                    if rec.get("carry_snapshot") is None and frame is not None:
                        rec["carry_snapshot"] = frame.copy()
                rec["near_frames"] += 1
                rec["carrier_role"]       = carrier_role
                rec["last_contact_ts"]    = now
                rec["approach_confirmed"] = True
                ax, ay = rec["anchor_pos"]
                if dist2((cx, cy), (ax, ay)) >= self.move_thresh ** 2:
                    rec["has_moved"] = True
                if strong_near or (rec["near_frames"] >= self.carry_frames and rec["has_moved"]):
                    rec["grab_hint"] = True
                if rec["near_frames"] >= self.carry_frames and rec["has_moved"]:
                    rec["being_carried"] = True
            else:
                rec["near_frames"] = max(0, rec["near_frames"] - 1)
                if rec["near_frames"] == 0:
                    rec["being_carried"] = False
                    if (now - rec.get("last_contact_ts", 0.0)) > self.contact_ttl_sec:
                        rec["has_moved"]          = False
                        rec["approach_confirmed"] = False
                        rec["grab_hint"]          = False
                        rec["carry_snapshot"]     = None
                        rec["missing_since"]      = 0.0

        for tid in list(self._items):
            rec = self._items[tid]
            age = now - rec["last_seen_ts"]
            if tid not in current_ids:
                zone_name = rec.get("zone_name")
                armed = (
                    rec.get("approach_confirmed") and
                    rec.get("grab_hint") and
                    zone_name is not None and
                    age < self.missing_keepalive_sec and
                    self._is_guest(rec.get("carrier_role", "unknown"))
                )
                if armed:
                    zone_pts = self._get_zone_pts(zone_name, goods_zones_px)
                    if zone_pts is not None:
                        _guest_present, _staff_present, human_present = self._zone_presence(
                            zone_pts, people, role_detections
                        )
                        if human_present:
                            rec["missing_since"] = 0.0
                        else:
                            if rec.get("missing_since", 0.0) <= 0.0:
                                rec["missing_since"] = now
                            elif (now - rec["missing_since"]) >= self.conceal_verify_sec:
                                self.alerts.append({
                                    "msg": (f"CONCEALMENT [{zone_name}]: "
                                            f"[{rec['name']}] id={tid} "
                                            f"concealed by {rec['carrier_role'].upper()}"),
                                    "ts":        now,
                                    "zone_name": zone_name,
                                    "snapshot":  rec.get("carry_snapshot"),
                                })
                                print(f"[CONCEAL] {self.alerts[-1]['msg']}")
                                rec["approach_confirmed"] = False
                                rec["grab_hint"]          = False
                                rec["being_carried"]      = False
                                rec["missing_since"]      = 0.0
                                rec["carry_snapshot"]     = None
                else:
                    rec["missing_since"] = 0.0

                if age > self.missing_keepalive_sec:
                    del self._items[tid]

        self.alerts = [a for a in self.alerts if (now - a["ts"]) < self.ALERT_DISPLAY]

    def being_carried_ids(self) -> set:
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
# Dashed rectangle for pinned-ghost zone items
# ─────────────────────────────────────────────────────────────
def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8):
    x1, y1 = pt1; x2, y2 = pt2
    edges = [(x1, y1, x2, y1), (x2, y1, x2, y2),
             (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for ax, ay, bx, by in edges:
        dx = bx - ax; dy = by - ay
        length = max(1, int((dx**2 + dy**2) ** 0.5))
        steps  = max(1, length // (dash_len * 2))
        for i in range(steps):
            t0 = i * 2 * dash_len / length
            t1 = min(1.0, (i * 2 + 1) * dash_len / length)
            p0 = (int(ax + dx * t0), int(ay + dy * t0))
            p1 = (int(ax + dx * t1), int(ay + dy * t1))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


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
# TrackBuffer  (visual ghost buffer for smooth display)
# ─────────────────────────────────────────────────────────────
class TrackBuffer:
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

    def remove_zone_items(self, zone_pts: list):
        """Immediately flush visual ghosts for a zone.
        Used when staff refreshes a zone or after theft is confirmed so the
        old pinned bbox/ID disappears right away instead of lingering for a
        few buffer frames."""
        for tid in list(self._buf.keys()):
            item = self._buf[tid].get("item") or {}
            center = item.get("center")
            if center is not None and point_in_poly(center, zone_pts):
                del self._buf[tid]


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
            a1=box_area(b1); a2=box_area(b2)
            if a1==0 or a2==0: continue
            inter  = intersection(b1, b2)
            io_min = inter / min(a1, a2)
            if io_min >= iou_min_thresh:
                if a1 > a2: suppress[i] = True
                else:        suppress[j] = True
    return [it for k, it in enumerate(items) if not suppress[k]]


# ─────────────────────────────────────────────────────────────
# SAHI — Slicing Aided Hyper Inference
#
# Chia frame thành các tile có overlap, chạy YOLO predict() trên
# từng tile, dịch toạ độ về full-frame rồi gộp bằng NMS.
# Cải thiện đáng kể khả năng detect vật thể nhỏ.
#
# Args:
#   model       : YOLO model (det_model)
#   frame       : full BGR frame
#   target_cls  : set of class names cần detect (ví dụ {"item"})
#                 None = detect tất cả class
#   slice_h/w   : kích thước mỗi tile (pixel)
#   overlap     : tỷ lệ overlap giữa các tile (0.0-0.5)
#   conf        : confidence threshold
#   imgsz       : imgsz truyền vào YOLO
#   nms_iou     : IoU threshold cho NMS gộp tile
#   merge_full  : nếu True, cũng chạy predict trên full-frame và gộp lại
# ─────────────────────────────────────────────────────────────
def sahi_infer(model, frame, target_cls=None,
               slice_h=320, slice_w=320, overlap=0.2,
               conf=0.1, imgsz=640, nms_iou=0.5,
               merge_full=False):
    """
    Trả về list of dict:
        {"bbox": np.array([x1,y1,x2,y2]), "conf": float, "cls_id": int, "cls_name": str}
    """
    H, W = frame.shape[:2]
    raw: list = []   # [x1, y1, x2, y2, conf, cls_id]

    # ── Tạo danh sách vị trí các tile ──────────────────────────
    def _tile_starts(dim_size, tile_size, ov):
        stride = max(1, int(tile_size * (1.0 - ov)))
        starts = list(range(0, dim_size - tile_size + 1, stride))
        # Đảm bảo tile cuối luôn cover đến cạnh frame
        last   = max(0, dim_size - tile_size)
        if not starts or starts[-1] < last:
            starts.append(last)
        return starts

    slice_h = min(slice_h, H)
    slice_w = min(slice_w, W)

    ys = _tile_starts(H, slice_h, overlap)
    xs = _tile_starts(W, slice_w, overlap)

    names = model.names or {}
    if target_cls is not None:
        target_ids = {cid for cid, cname in names.items()
                      if str(cname).lower() in target_cls}
    else:
        target_ids = None

    # ── Chạy predict trên từng tile ────────────────────────────
    def _run_on_roi(roi, ox, oy):
        res = model.predict(roi, imgsz=imgsz, conf=conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return
        r = res[0]
        for i in range(len(r.boxes)):
            cid = int(r.boxes.cls[i].cpu())
            if target_ids is not None and cid not in target_ids:
                continue
            bx1, by1, bx2, by2 = r.boxes.xyxy[i].cpu().numpy()
            raw.append([
                float(bx1 + ox), float(by1 + oy),
                float(bx2 + ox), float(by2 + oy),
                float(r.boxes.conf[i].cpu()),
                cid,
            ])

    for y0 in ys:
        y1_tile = min(y0 + slice_h, H)
        for x0 in xs:
            x1_tile = min(x0 + slice_w, W)
            tile    = frame[y0:y1_tile, x0:x1_tile]
            _run_on_roi(tile, x0, y0)

    # ── Tuỳ chọn: gộp thêm kết quả từ full-frame ───────────────
    if merge_full:
        _run_on_roi(frame, 0, 0)

    if not raw:
        return [], names

    # ── NMS để loại bỏ duplicate giữa các tile ─────────────────
    boxes_xywh = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in raw]
    scores     = [d[4] for d in raw]

    keep_idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf, nms_iou)
    if len(keep_idx) == 0:
        return [], names

    keep_idx = keep_idx.flatten()
    result   = []
    for idx in keep_idx:
        d = raw[idx]
        result.append({
            "bbox":     np.array([d[0], d[1], d[2], d[3]], dtype=float),
            "conf":     d[4],
            "cls_id":   d[5],
            "cls_name": str(names.get(d[5], d[5])),
        })
    return result, names


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
    ap.add_argument("--conf_det",  type=float, default=0.1,
                    help="Confidence threshold for det_model items (default 0.4)")
    ap.add_argument("--conf_role", type=float, default=0.72,
                    help="Confidence threshold for staff/guest detections. Raise this when staff sometimes flips to guest (default 0.72).")
    ap.add_argument("--conf_pose", type=float, default=0.4)
    ap.add_argument("--ffmpeg_path",  default="ffmpeg")
    ap.add_argument("--ffprobe_path", default="ffprobe")
    ap.add_argument("--out_width",  type=int,   default=0)
    ap.add_argument("--out_height", type=int,   default=0)
    ap.add_argument("--ffmpeg_fps", type=float, default=12.0)
    ap.add_argument("--timeout_us", type=int,   default=10_000_000)
    ap.add_argument("--pose_fps",   type=float, default=4.0)
    ap.add_argument("--det_fps",    type=float, default=6.0)
    ap.add_argument("--carry_dist",   type=int,   default=40)
    ap.add_argument("--carry_frames", type=int,   default=3)
    ap.add_argument("--move_thresh",  type=int,   default=18)
    ap.add_argument("--grasp_dist",   type=int,   default=24)
    ap.add_argument("--conceal_verify_sec", type=float, default=1.2)
    ap.add_argument("--cooldown_sec", type=float, default=1.5)
    ap.add_argument("--exit_settle_sec", type=float, default=2.5)
    ap.add_argument("--goods_zone_names", default="zone1,zone2")
    ap.add_argument("--guest_reach_dist", type=int, default=55)
    ap.add_argument("--unknown_as_guest", action="store_true", default=False)
    ap.add_argument("--no_unknown_as_guest", dest="unknown_as_guest", action="store_false")
    ap.add_argument("--stable_pos_thresh", type=float, default=45.0)
    ap.add_argument("--stable_ghost_ttl",  type=float, default=3.0,
                    help="Ghost TTL for non-zone items only. Zone items are pinned forever.")
    # NEW role-tracker params
    ap.add_argument("--role_confirm_count", type=int,   default=4,
                    help="Hits needed within --role_window_s to confirm an initial role label.")
    ap.add_argument("--role_window_s",      type=float, default=8.0,
                    help="Sliding window in seconds for role confirmation.")
    ap.add_argument("--role_iou_thresh",    type=float, default=0.25)
    ap.add_argument("--role_switch_count",  type=int,   default=7,
                    help="Hits needed inside the window before switching STAFF↔GUEST.")
    ap.add_argument("--role_switch_min_conf", type=float, default=0.78,
                    help="Average confidence needed before switching STAFF↔GUEST.")
    ap.add_argument("--role_switch_margin", type=int,   default=2,
                    help="New role must beat the current role by at least this many hits before switching.")
    # Legacy compat — kept so old scripts don't break
    ap.add_argument("--role_confirm_frames", type=int, default=2)
    ap.add_argument("--ha_url",       default="http://192.168.2.245:8123/")
    ap.add_argument("--ha_token",     default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4Y2ZlYzc0MWFjMGM0OWFiYmFjOWU4YzM0ZTZiMDZkMyIsImlhdCI6MTc3MzIyMjQyNiwiZXhwIjoyMDg4NTgyNDI2fQ.G0mSeB_myjqozMUN8hXbVxNH6xr8eTzi1SWz-Qj_HRI")
    ap.add_argument("--ha_webhook_id",   default="antitheft_alert")
    ap.add_argument("--ha_snapshot_dir", default="www/snapshots")
    ap.add_argument("--ha_clip_sec",  type=float, default=0.0)
    ap.add_argument("--ha_cooldown",  type=float, default=15.0)
    ap.add_argument("--show_breakdown",  action="store_true")
    ap.add_argument("--draw_items",      action="store_true")
    ap.add_argument("--draw_fps",        action="store_true")
    ap.add_argument("--draw_carry_ring", action="store_true")
    # ── SAHI (Slicing Aided Hyper Inference) ──────────────────
    ap.add_argument("--sahi",            action="store_true", default=False,
                    help="Bật SAHI: chia frame thành tile nhỏ để detect vật thể nhỏ tốt hơn.")
    ap.add_argument("--sahi_slice_h",    type=int,   default=320,
                    help="Chiều cao mỗi tile SAHI (pixel, default=320).")
    ap.add_argument("--sahi_slice_w",    type=int,   default=320,
                    help="Chiều rộng mỗi tile SAHI (pixel, default=320).")
    ap.add_argument("--sahi_overlap",    type=float, default=0.2,
                    help="Tỷ lệ overlap giữa các tile SAHI (0.0-0.5, default=0.2).")
    ap.add_argument("--sahi_nms_iou",    type=float, default=0.5,
                    help="IoU threshold khi gộp kết quả các tile bằng NMS (default=0.5).")
    ap.add_argument("--sahi_merge_full", action="store_true", default=False,
                    help="Nếu bật, cũng chạy thêm predict trên full-frame rồi gộp lại (tốt hơn cho vật thể lớn lẫn nhỏ).")
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

    goods_zone_names = {norm_name(x)
                        for x in args.goods_zone_names.split(",") if x.strip()}
    zone_guards: dict = {}
    for z in zones_norm:
        zname = z["name"]
        if norm_name(zname) in goods_zone_names:
            zg = GoodsZoneGuard(zname)
            zg.COOLDOWN_SEC    = args.cooldown_sec
            zg.EXIT_SETTLE_SEC = args.exit_settle_sec
            zone_guards[zname] = zg
        else:
            zone_guards[zname] = None

    goods_zone_name_list = [z["name"] for z in zones_norm
                            if norm_name(z["name"]) in goods_zone_names]

    conceal = ConcealmentTracker(
        carry_dist         = args.carry_dist,
        carry_frames       = args.carry_frames,
        move_thresh        = args.move_thresh,
        unknown_as_guest   = args.unknown_as_guest,
        grasp_dist         = args.grasp_dist,
        conceal_verify_sec = args.conceal_verify_sec,
    )

    stable_tracker = StableItemTracker(
        pos_thresh = args.stable_pos_thresh,
        ghost_ttl  = args.stable_ghost_ttl,
    )

    track_buf    = TrackBuffer(ghost_frames=6)

    role_tracker = PersonRoleTracker(
        confirm_count        = args.role_confirm_count,
        candidate_window_s   = args.role_window_s,
        iou_thresh           = args.role_iou_thresh,
        switch_confirm_count = args.role_switch_count,
        switch_min_conf      = args.role_switch_min_conf,
        switch_margin        = args.role_switch_margin,
    )

    print(f"[CONFIG] role confirm: {args.role_confirm_count} hits / "
          f"{args.role_window_s}s window  |  switch={args.role_switch_count} hits @ avg_conf>={args.role_switch_min_conf:.2f} "
          f"| unknown_as_guest={args.unknown_as_guest}")
    print(f"[CONFIG] conf_det={args.conf_det} (items)  conf_role={args.conf_role} (staff/guest)")
    print(f"[CONFIG] zone settle={args.exit_settle_sec}s  verify={args.cooldown_sec}s  conceal_verify={args.conceal_verify_sec}s  grasp_dist={args.grasp_dist}px")

    ha = HomeAssistantNotifier(
        ha_url            = args.ha_url,
        token             = args.ha_token,
        zone_names        = goods_zone_name_list,
        webhook_id        = args.ha_webhook_id,
        snapshot_dir      = args.ha_snapshot_dir,
        clip_duration_s   = args.ha_clip_sec,
        notify_cooldown_s = args.ha_cooldown,
    )

    _prev_alerting_zones: set = set()
    _prev_conceal_count:  int = 0

    reader = FFmpegLatestFrameReader(
        src=src, width=w, height=h, out_fps=args.ffmpeg_fps,
        timeout_us=args.timeout_us, ffmpeg_path=args.ffmpeg_path)
    reader.start()

    win = "Anti-Theft Monitor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_pose_people      = []
    last_det_items_raw    = []
    last_det_items_stable = []
    last_det_items_vis    = []
    last_role_dets        = []
    last_goods_zones_px   = []

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

            clean_frame = frame
            vis = frame.copy()
            H, W = vis.shape[:2]

            if fid == last_seen_fid:
                if now - last_frame_ts > 2.0:
                    cv2.putText(vis, "STREAM STALLED", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                last_seen_fid = fid; last_frame_ts = now

            # ── Build pixel zones ──────────────────────────────────────────
            zones_px       = []
            goods_zones_px = []
            for z in zones_norm:
                pts = [(int(clamp(p[0], 0, 1) * W), int(clamp(p[1], 0, 1) * H))
                       for p in z["points"]]
                zd = {"name": z["name"], "pts": pts}
                zones_px.append(zd)
                if norm_name(z["name"]) in goods_zone_names:
                    goods_zones_px.append(zd)
            last_goods_zones_px = goods_zones_px

            # ── YOLOPose ───────────────────────────────────────────────────
            pose_iv = 1.0 / args.pose_fps if args.pose_fps > 0 else 0.0
            if pose_iv == 0.0 or now >= next_pose_ts:
                next_pose_ts = now + pose_iv
                pres = pose_model.predict(clean_frame, imgsz=args.imgsz,
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

            # ── Draw pose + role label ─────────────────────────────────────
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

            # ── YOLO det_model tracking ────────────────────────────────────
            det_iv = 1.0 / args.det_fps if args.det_fps > 0 else 0.0
            if det_iv == 0.0 or now >= next_det_ts:
                next_det_ts = now + det_iv

                items_raw = []
                role_dets = []

                if args.sahi:
                    # ── SAHI path ──────────────────────────────────────────
                    # 1. Chạy SAHI trên các tile nhỏ để detect "item" (vật thể nhỏ)
                    sahi_dets, det_names = sahi_infer(
                        model      = det_model,
                        frame      = clean_frame,
                        target_cls = {"item"},
                        slice_h    = args.sahi_slice_h,
                        slice_w    = args.sahi_slice_w,
                        overlap    = args.sahi_overlap,
                        conf       = args.conf_det,
                        imgsz      = args.imgsz,
                        nms_iou    = args.sahi_nms_iou,
                        merge_full = args.sahi_merge_full,
                    )
                    for sd in sahi_dets:
                        bbox   = sd["bbox"]
                        cx, cy = xyxy_center(bbox)
                        items_raw.append({
                            "bbox":     bbox,
                            "center":   (cx, cy),
                            "conf":     sd["conf"],
                            "name":     sd["cls_name"],
                            "track_id": None,   # SAHI dùng predict(), StableItemTracker tự gán ID
                        })

                    # 2. Chạy full-frame predict() để lấy staff/guest (role_dets)
                    #    Người/role thường to → không cần SAHI, chỉ cần full-frame
                    try:
                        rdres = det_model.track(clean_frame, imgsz=args.imgsz, conf=args.conf_det,
                                                persist=True, verbose=False, iou=0.3)
                    except Exception:
                        rdres = det_model.predict(clean_frame, imgsz=args.imgsz,
                                                  conf=args.conf_det, verbose=False)
                    if rdres:
                        r = rdres[0]
                        if r.boxes is not None and len(r.boxes) > 0:
                            boxes_ff = r.boxes.xyxy.cpu().numpy()
                            confs_ff = r.boxes.conf.cpu().numpy()
                            clss_ff  = r.boxes.cls.cpu().numpy().astype(int)
                            names_ff = r.names or {}
                            for i in range(len(boxes_ff)):
                                lname = str(names_ff.get(int(clss_ff[i]), clss_ff[i])).lower()
                                if lname in ROLE_CLASSES:
                                    if float(confs_ff[i]) >= args.conf_role:
                                        role_dets.append({
                                            "bbox": boxes_ff[i].astype(float),
                                            "name": lname,
                                            "conf": float(confs_ff[i]),
                                        })

                else:
                    # ── Standard path (không dùng SAHI) ───────────────────
                    try:
                        dres = det_model.track(clean_frame, imgsz=args.imgsz, conf=args.conf_det,
                                               persist=True, verbose=False, iou=0.3)
                    except Exception:
                        dres = det_model.predict(clean_frame, imgsz=args.imgsz,
                                                 conf=args.conf_det, verbose=False)

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
                                conf_val = float(confs[i])
                                if lname in ROLE_CLASSES:
                                    # Staff/guest need higher confidence to avoid
                                    # confusion between the two classes
                                    if conf_val >= args.conf_role:
                                        role_dets.append({"bbox": bbox, "name": lname,
                                                          "conf": conf_val})
                                elif lname == "item":
                                    items_raw.append({
                                        "bbox":     bbox,
                                        "center":   (cx, cy),
                                        "conf":     conf_val,
                                        "name":     name,
                                        "track_id": tids[i],
                                    })

                items_raw = filter_contained_boxes(items_raw, iou_min_thresh=0.6)

                # Pass goods_zones_px so zone items get pinned immediately
                last_det_items_stable, last_det_items_raw = stable_tracker.update(
                    items_raw, now,
                    goods_zones_px=goods_zones_px,
                )
                last_det_items_vis = track_buf.update(
                    [dict(it) for it in last_det_items_stable]
                )
                last_role_dets = role_dets
                if last_pose_people:
                    role_tracker.update(last_pose_people, role_dets, now)

                det_runs += 1
                if now - det_last >= 1.0:
                    det_fps  = det_runs / (now - det_last)
                    det_runs = 0; det_last = now

            # ── Concealment update ─────────────────────────────────────────
            conceal.update(
                items           = last_det_items_raw,
                people          = last_pose_people,
                goods_zones_px  = last_goods_zones_px,
                now             = now,
                frame           = vis,
                role_detections = last_role_dets,
            )
            carried_ids = conceal.being_carried_ids()

            # ── Goods zone guard update ────────────────────────────────────
            for zd in zones_px:
                zname = zd["name"]
                pts   = zd["pts"]
                guard = zone_guards.get(zname)
                if guard is None: continue

                zone_items_stable = items_inside_zone(last_det_items_stable, pts)
                stable_count      = len(zone_items_stable)

                zone_items_raw = items_inside_zone(last_det_items_raw, pts)
                raw_count      = len(zone_items_raw)

                interaction_items = zone_items_raw if zone_items_raw else zone_items_stable

                guest_interacting = any(
                    person_role_is(p, "guest") and
                    person_interacts_with_goods_zone(
                        p, pts, interaction_items,
                        reach_dist=args.guest_reach_dist)
                    for p in last_pose_people
                )
                staff_interacting = any(
                    person_role_is(p, "staff") and
                    person_interacts_with_goods_zone(
                        p, pts, interaction_items,
                        reach_dist=args.guest_reach_dist)
                    for p in last_pose_people
                )
                unknown_interacting = any(
                    person_role_is(p, "unknown") and
                    person_interacts_with_goods_zone(
                        p, pts, interaction_items,
                        reach_dist=args.guest_reach_dist)
                    for p in last_pose_people
                )
                if args.unknown_as_guest and unknown_interacting:
                    guest_interacting = True

                guest_present_pose = any(
                    person_role_is(p, "guest") and bbox_overlaps_zone(p["bbox"], pts)
                    for p in last_pose_people
                )
                staff_present_pose = any(
                    person_role_is(p, "staff") and bbox_overlaps_zone(p["bbox"], pts)
                    for p in last_pose_people
                )
                unknown_present_pose = any(
                    person_role_is(p, "unknown") and bbox_overlaps_zone(p["bbox"], pts)
                    for p in last_pose_people
                )
                guest_present_det = role_det_present_in_zone(last_role_dets, pts, role_name="guest")
                staff_present_det = role_det_present_in_zone(last_role_dets, pts, role_name="staff")

                guest_present = guest_present_pose or guest_present_det
                staff_present = staff_present_pose or staff_present_det
                unknown_present = unknown_present_pose and not args.unknown_as_guest
                if args.unknown_as_guest and unknown_present_pose:
                    guest_present = True

                # If pose is lost but det_model still sees a guest/staff head/body
                # in the zone, keep the state on HOLD instead of starting a false alert.
                cooldown_blocked = staff_present or unknown_present

                guard.update(
                    guest_interacting  = guest_interacting,
                    stable_item_count  = stable_count,
                    now                = now,
                    staff_interacting  = staff_interacting,
                    raw_item_count     = raw_count,
                    cooldown_blocked   = cooldown_blocked,
                    guest_present      = guest_present,
                    staff_present      = staff_present,
                    unknown_present    = unknown_present,
                )

                for event_name in guard.pop_events():
                    kind = f"zone_{zname}"
                    if event_name == "stage_snapshot":
                        ha.stage_snapshot(kind=kind, frame=clean_frame.copy())
                    elif event_name == "discard_snapshot":
                        ha.discard_staged(kind)

                # Staff left zone → remove pinned ghosts (staff may have
                # added or removed items) then re-baseline with real count.
                if guard.staff_reset_needed:
                    stable_tracker.remove_zone_ghosts(pts)
                    track_buf.remove_zone_items(pts)
                    guard.force_rebaseline(raw_count)

                if guard.is_alerting and guard.missing > 0:
                    stable_tracker.remove_zone_ghosts(pts)
                    track_buf.remove_zone_items(pts)

            # ── Draw zones ─────────────────────────────────────────────────
            _fsl = max(0.35, min(0.50, H / 900))
            _fsc = max(0.50, min(0.75, H / 700))

            for zd in zones_px:
                zname = zd["name"]
                poly  = np.array(zd["pts"], dtype=np.int32)
                guard = zone_guards.get(zname)

                zone_items_stable = items_inside_zone(last_det_items_stable, zd["pts"])
                visible_now       = len(zone_items_stable)

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
                            p, zd["pts"], zone_items_stable,
                            reach_dist=args.guest_reach_dist)
                    )
                    staff_cnt = sum(
                        1 for p in last_pose_people
                        if person_role_is(p, "staff") and
                        person_interacts_with_goods_zone(
                            p, zd["pts"], zone_items_stable,
                            reach_dist=args.guest_reach_dist)
                    )
                    lbl = f"{zname} G={guest_cnt} S={staff_cnt} I={shown}{sfx}"
                else:
                    lbl = f"{zname} I={shown}{sfx}"

                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, _fsl, 1)
                cv2.rectangle(vis, (max(0, tx-3), max(0, ty-th-3)),
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

            # ── Draw items ─────────────────────────────────────────────────
            if args.draw_items:
                _fsi = max(0.30, min(0.42, H / 900))
                for it in last_det_items_vis:
                    x1, y1, x2, y2 = [int(v) for v in it["bbox"]]
                    tid    = it.get("track_id")
                    ghost  = it.get("ghost",  False)
                    pinned = it.get("pinned", False)
                    in_zone = item_in_any_zone(it, last_goods_zones_px) is not None

                    if tid in carried_ids:
                        color = (0, 0, 255)       # red    = being carried
                    elif pinned and ghost:
                        color = (0, 180, 255)      # orange = pinned zone ghost
                    elif ghost:
                        color = (100, 100, 100)    # grey   = normal ghost
                    elif in_zone:
                        color = (0, 220, 0)        # green  = live in zone
                    else:
                        color = (0, 128, 255)      # blue   = live outside zone

                    # Dashed border for pinned ghosts → visually distinct
                    if pinned and ghost:
                        draw_dashed_rect(vis, (x1, y1), (x2, y2), color, thickness=1)
                    else:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

                    lbl = f"#{tid}" if tid is not None else ""
                    if pinned and ghost:   lbl += " [P]"
                    elif ghost:            lbl += " [G]"
                    if tid in carried_ids: lbl += " CARRIED!"
                    lbl += f" {it['conf']:.2f}"
                    (iw, ih), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, _fsi, 1)
                    lx, ly = x1, max(ih + 2, y1 - 2)
                    # cv2.rectangle(vis, (lx, ly-ih-2),
                    #               (min(W-1, lx+iw+4), ly+2), (0,0,0), -1)
                    # cv2.putText(vis, lbl, (lx+2, ly),
                    #             cv2.FONT_HERSHEY_SIMPLEX, _fsi, color, 1, cv2.LINE_AA)

                for rd in last_role_dets:
                    x1, y1, x2, y2 = [int(v) for v in rd["bbox"]]
                    rc = ROLE_COLORS.get(rd["name"], (200, 200, 200))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), rc, 1)
                    cv2.putText(vis, f"[det]{rd['name']} {rd['conf']:.2f}",
                                (x1, max(10, y1-2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, rc, 1, cv2.LINE_AA)

            # Alert messages stay in console / Home Assistant only.

            # ── Home Assistant ─────────────────────────────────────────────
            curr_alerting_zones = {
                zname for zname, g in zone_guards.items()
                if g is not None and g.is_alerting
            }
            for zname in (curr_alerting_zones - _prev_alerting_zones):
                guard = zone_guards[zname]
                ha.send_alert(kind=f"zone_{zname}", message=guard.alert_msg,
                              frame=clean_frame.copy(), now=now,
                              use_staged_snapshot=True)
            for zname in (_prev_alerting_zones - curr_alerting_zones):
                ha.clear_alert(f"zone_{zname}")
            _prev_alerting_zones = curr_alerting_zones

            curr_conceal_count = len(conceal.alerts)
            if curr_conceal_count > _prev_conceal_count:
                for a in conceal.alerts[_prev_conceal_count:]:
                    zone_name = a.get("zone_name", "")
                    snap      = a.get("snapshot")
                    if zone_name:
                        ha.send_alert(
                            kind    = f"zone_{zone_name}",
                            message = a["msg"],
                            frame   = snap if snap is not None else vis.copy(),
                            now     = now,
                        )
            _prev_conceal_count = curr_conceal_count

            # ── FPS overlay ────────────────────────────────────────────────
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
            ha.push_frame(vis, now)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")): break

    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()