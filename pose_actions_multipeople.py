import sys
import time
from collections import deque

import cv2
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from osc_client import OSCClient

from helpers import (
    clamp_box,
    expand_box,
    energy_to_level,
    compute_energy,
    compute_group_stats,
    is_robot_arms,
    average_to_level,
    max_energy_converter,
)


# ------------------ OSC ------------------

OSC_IP = "127.0.0.1"
OSC_PORT = 8000

# ------------------ Detection ------------------
CONF = 0.4       # min YOLO confidence; higher = fewer false detections
DEVICE = "mps"   # run YOLO on the Mac GPU (Metal) instead of CPU


def main():
    osc = OSCClient(OSC_IP, OSC_PORT)

    detector = YOLO("yolo11n.pt")

    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    pose = vision.PoseLandmarker.create_from_options(options)

    MAX_MISSES = 15   # frames to keep an id's history after it disappears
    HIST_LEN = 20     # keypoint frames kept per person for the energy calc

    track_hist = {}   # tid -> deque of recent keypoint dicts (per person)
    last_seen = {}    # tid -> last frame index this id was returned
    prev_people_count = -1
    # person_energy_state maps tid -> { "level": str, "value": float }
    person_energy_state = {}
    
    # Buffer to debounce people count changes
    STABILITY_FRAMES = 23  # Require 23 consecutive frames with same count to confirm change
    people_count_history = deque(maxlen=STABILITY_FRAMES)

    # last sent values for group stats (floats)
    last_sent_avg = None
    last_sent_max = None
    last_sent_std_time = 0.0
    gesture_active = False  # true while someone holds the robot arms pose

    frame_idx = 0
    seen_ids = set()  # debug: every track id ever created
    max_people = 0    # debug: peak simultaneous people

    # optional video file path as an argument, else webcam
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)

    fps = 30
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        H, W = frame.shape[:2]

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        # detect + track in ONE call: YOLO finds people, ByteTrack keeps their ids
        results = detector.track(
            frame,
            persist=True,
            conf=CONF,
            classes=[0],
            device=DEVICE,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        # rebuild the {tid: {bbox, hist}} shape the rest of the loop expects
        tracks = {}
        boxes = results.boxes
        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy()):
                tid = int(tid)
                if tid not in track_hist:
                    track_hist[tid] = deque(maxlen=HIST_LEN)
                last_seen[tid] = frame_idx
                tracks[tid] = {"bbox": box, "hist": track_hist[tid]}

        # drop history for ids ByteTrack hasn't returned in a while
        for tid in [t for t, f in list(last_seen.items()) if frame_idx - f > MAX_MISSES]:
            track_hist.pop(tid, None)
            last_seen.pop(tid, None)
            person_energy_state.pop(tid, None)

        people_count = len(tracks)

        # debug: multi-person reliability. with a fixed group in frame, total ids
        # should equal peak people; extra ids = churn (re-registrations / swaps)
        seen_ids.update(tracks.keys())
        max_people = max(max_people, people_count)
        if frame_idx % 30 == 0:
            churn = len(seen_ids) - max_people
            active = sorted(tracks.keys())
            print(
                f"fps {fps:.1f}  people {people_count}  peak {max_people}  "
                f"total ids {len(seen_ids)}  churn {churn}  active {active}"
            )

        # Add to history and check if count is stable
        people_count_history.append(people_count)
        
        # Only send change signal if buffer is full and all values match (stable count)
        if len(people_count_history) == STABILITY_FRAMES:
            stable_count = people_count_history[0]
            if all(c == stable_count for c in people_count_history) and stable_count != prev_people_count:
                osc.send_people(4 if stable_count % 4 == 0 else stable_count % 4)
                # Print energy level list (strings) for everyone currently known
                energy_list = [v["level"] for v in person_energy_state.values()]
                print(f"People changed to {stable_count} — energy levels: {energy_list}")
                prev_people_count = stable_count

        # Per-person pose + energy
        gesture_now = False  # set true if anyone is in the robot arms pose
        for tid, t in tracks.items():
            x1, y1, x2, y2 = expand_box(*t["bbox"], W, H)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            )

            result = pose.detect(mp_image)
            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks[0]

            if is_robot_arms(lm):
                gesture_now = True

            def p(i):
                return (lm[i].x, lm[i].y)

            keypoints = {
                "lw": p(15),
                "rw": p(16),
                "la": p(27),
                "ra": p(28),
                "lh": p(23),
                "rh": p(24),
            }

            t["hist"].append(keypoints)

            _, _, _, norm = compute_energy(t["hist"], fps)
            level = energy_to_level(norm)

            if tid not in person_energy_state:
                person_energy_state[tid] = {"level": level, "value": norm}
                # osc.send_energy(tid, level)   don't send individual energy level
            else:
                prev_level = person_energy_state[tid]["level"]
                # if level != prev_level:
                #     osc.send_energy(tid, level) 
                # Always update stored value
                person_energy_state[tid] = {"level": level, "value": norm}

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tid}  {level}",
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # fire the rover arm trigger once when the pose starts
        if gesture_now and not gesture_active:
            osc.send_gesture("roverarm")
            print("robot arms -> rover arm trigger")
        gesture_active = gesture_now

        # Compute group stats (numeric floats 0..1)
        numeric_vals = [v["value"] for v in person_energy_state.values()]
        avg, mn, mx, std = compute_group_stats(numeric_vals)

        # Send OSC when avg/max change by more than 0.1
        if avg is not None:
            if last_sent_avg is None or abs(avg - last_sent_avg) > 0.1:
                osc.send_group_avg(avg * 400 - 200)
                last_sent_avg = avg

        if mx is not None:
            if last_sent_max is None or abs(mx - last_sent_max) > 0.1:
                osc.send_group_max(mx * 26) 
                last_sent_max = mx

        # Send std every 1 second
        now_t = time.time()
        if std is not None and (now_t - last_sent_std_time) >= 1.0:
            osc.send_group_std(std)
            last_sent_std_time = now_t

        # Convert numeric stats to level strings for display
        avg_level = average_to_level(avg)
        max_level = max_energy_converter(mx)

        # Overlay FPS, people, and group energy stats
        if avg is None:
            stats_text = f"FPS: {fps:.1f}   People: {people_count}"
        else:
            stats_text = (
                f"FPS: {fps:.1f}   People: {people_count}   "
                f"Avg: {avg:.2f}({avg_level})  Max: {mx:.2f}({max_level})  Std: {std:.2f}"
            )

        cv2.putText(
            frame,
            stats_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Pose Energy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
