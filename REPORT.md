# Mars Immersive Soundscape - Progress Report

Running log of what was built, the techniques used, results, and the plan.
Not committed to git. Last updated 2026-08-05.

## What this project is

Extends an interactive Mars installation. A webcam feed is analyzed to detect
people and how much they move. Those signals stream over OSC to Max/MSP, which
plays NASA JPL Mars rover audio that responds to visitor movement. My focus is
improving the detection and tracking, then building movement-driven sound states.

## Techniques and implementations (report-ready)

Detection and tracking
- YOLO11n (Ultralytics) for person detection. Nano size for speed, upgraded from
  the older YOLOv8n for better accuracy at the same speed.
- Runs on the Apple GPU via MPS (Metal) instead of the CPU.
- Explicit confidence threshold (0.4) to filter out false detections.
- Person-only class filter (COCO class 0).
- ByteTrack multi-object tracker (Ultralytics model.track, persist=True), which
  replaced a hand-written greedy IoU tracker. ByteTrack adds:
  - Kalman filter motion model that predicts each person's next position
  - Optimal assignment (linear assignment, lap library) instead of greedy matching
  - Two-stage association that recovers low-confidence detections
  - Track buffer that holds an ID through brief occlusion
- Per-person energy history keyed by track ID.

Pose and energy
- MediaPipe Pose Landmarker (lite) finds body joints for each person.
- Energy is the median speed of wrists, ankles, and hips, normalized to a level
  (idle, light, moderate, high).

Signal and sound
- OSC over UDP (python-osc) to Max/MSP on port 8000.
- Max side: a patch (in the assets folder, not committed) plays real NASA sounds
  with sfplay~. It maps /group/avg to the wind bed volume and /gesture/roverarm to
  a 5 second laser zap (delay-stopped).
- Gestures: the robot-arms pose (read from shoulders, elbows, wrists) fires a
  /gesture/roverarm OSC flag, edge-triggered so it sends once per strike.
- Planned: state triggers (dust storm first).

Measurement
- Debug metric: unique IDs, peak people, and churn (extra IDs beyond the peak =
  re-registrations or swaps) to measure tracking reliability before and after.

Workflow
- Own GitHub fork, small iterative single-line commits.

## Results so far

- Single-person baseline: 3 unique IDs for one person, one phantom detection, ~10 fps.
- After changes: phantom detections gone, unique IDs down toward 1, ~10 fps.
  The fps limit is the pose step, not detection.
- Multi-person (2 to 4), tested on a front-facing corridor clip: IDs stay stable
  while people are in frame, churn is mostly people entering and leaving plus minor
  edge flicker as they exit. fps 12 to 28 at this resolution. Good enough for the
  1 to 4 person case.
- On a hard 9-person overhead plaza clip: IDs hold unless occlusion lasts a long
  time, people close together sometimes merge into one. Crowd-density stressors,
  not the target case.
- First end-to-end sound slice works: movement drives the wind bed volume and the
  robot-arms gesture fires a 5 second laser zap, through a Max patch. Proves the
  full chain from camera to detection to OSC to Max to NASA sound.

## Scope decisions

- Dropped median and mode room stats. With only 1 to 4 people they are basically
  redundant with the average energy that is already sent, so they would not change
  what the audience hears. Focus stays on detection robustness and the sound states.

## Changelog

- 7d2d82a  upgraded detection to yolo11 and bytetrack with confidence filtering on gpu
- 310029b  added multi person debug for people count peak and id churn
- 7fc5a18  added optional video file argument for testing
- 7451139  added robot arms gesture that fires a rover arm osc trigger

## Timeline (solo student estimates)

- Milestone 1 (harden 1 to 4 people): ~4 to 8 hrs
- Milestone 2 (dust storm state): ~6 to 10 hrs
- To a working dust-storm demo: ~10 to 18 hrs (about 2 to 3 focused days)
- Later: each extra state ~2 to 4 hrs, gestures ~4 to 8 hrs

## Roadmap

Milestone 1 - harden 1 to 4 people
- 1.1 multi-person debug output (done)
- 1.2 test and tune (done, tested on a front-facing 1 to 4 person clip, tracking solid)
- 1.3 survive brief occlusion

Milestone 2 - dust storm state (first sound state)
- define the trigger (high collective energy sustained)
- detect the state and send OSC
- wire it in Max

Later, one iteration each: earthquake, rover appears / rover arm, everybody
jumping. Then gestures (waving, arm-out).

## Session log

### 2026-08-06

Milestone 1 (harden 1 to 4 person detection)
- Added a multi-person debug readout to the main loop: current people count, peak
  simultaneous people, total unique ids, and churn (extra ids beyond the peak,
  which flags re-registrations or id swaps), plus the active id set.
- Added an optional video-file argument so a clip can be passed instead of the
  webcam, for repeatable testing.
- Tested on a front-facing corridor clip with 1 to 4 people: ids stay stable while
  people are in frame, churn is mostly people entering and leaving, fps 12 to 28.
  Good for the target case. Also tested a hard 9-person overhead plaza clip where
  ids hold except through long occlusion and close people sometimes merge. Those
  are crowd stressors, not the 1 to 4 person case, so no code change was needed.

Scope decision
- Evaluated adding median and mode room stats and dropped them. With only 1 to 4
  people they are redundant with the average that is already sent.

Gesture
- Added robot-arms gesture detection. Reads shoulders, elbows, and wrists from the
  pose (mediapipe already provides them) and detects the pose where both elbows are
  out at about shoulder height with one forearm up and one down. When someone
  strikes it, it fires a /gesture/roverarm OSC flag, edge-triggered so it sends once
  per strike. Gated on landmark visibility.

Sound side (lives in the assets folder, not committed)
- Sourced NASA Mars audio (public domain) from the Sounds of Mars page: wind
  (ambient bed), SuperCam laser zaps (gesture trigger), and rover driving.
- Built a Max patch that receives the OSC, plays the sounds with sfplay~, maps
  /group/avg to the wind bed volume, and fires the laser for 5 seconds on
  /gesture/roverarm (a delay then a stop). The sounds and patch live in
  ~/Documents/Mars-Immersive-Soundscape-assets/, outside the git repo.

Result
- First end-to-end sound slice works: moving changes the wind, the robot-arms pose
  fires the laser. Full chain from camera to detection to OSC to Max to NASA sound.

Commits today: 310029b, 7fc5a18, 7451139.
