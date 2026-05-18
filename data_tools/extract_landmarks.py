import argparse
import json
from pathlib import Path

import copy

import cv2
import mediapipe as mp

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


def lm_to_dict(lm):
    out = {
        "x": float(lm.x),
        "y": float(lm.y),
        "z": float(lm.z),
    }
    if hasattr(lm, "visibility"):
        out["visibility"] = float(lm.visibility)
    if hasattr(lm, "presence"):
        out["presence"] = float(lm.presence)
    return out


def landmarks_to_list(landmarks):
    if not landmarks:
        return None
    return [lm_to_dict(lm) for lm in landmarks]


def create_pose_landmarker(model_path: str):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return PoseLandmarker.create_from_options(options)


def create_hand_landmarker(model_path: str):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


def create_face_landmarker(model_path: str):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def extract_hand_sides(hand_res):
    """
    Returns (left_hand, right_hand, left_world, right_world).
    Uses handedness when available and falls back safely when it is missing.
    """
    left_hand = None
    right_hand = None
    left_world = None
    right_world = None

    if not hand_res.hand_landmarks:
        return left_hand, right_hand, left_world, right_world

    for idx, hand_landmarks in enumerate(hand_res.hand_landmarks):
        print("Hand")
        data = landmarks_to_list(hand_landmarks)
        world_data = None
        if hand_res.hand_world_landmarks and idx < len(hand_res.hand_world_landmarks):
            world_data = landmarks_to_list(hand_res.hand_world_landmarks[idx])

        label = None
        if hand_res.handedness and idx < len(hand_res.handedness):
            if len(hand_res.handedness[idx]) > 0:
                label = hand_res.handedness[idx][0].category_name.lower()

        if label == "left":
            left_hand = data
            left_world = world_data
        elif label == "right":
            right_hand = data
            right_world = world_data
        else:
            # Fallback if handedness is missing
            if left_hand is None:
                left_hand = data
                left_world = world_data
            elif right_hand is None:
                right_hand = data
                right_world = world_data

    return left_hand, right_hand, left_world, right_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--pose_model", required=True, help="pose_landmarker.task path")
    parser.add_argument("--hand_model", required=True, help="hand_landmarker.task path")
    parser.add_argument("--face_model", required=True, help="face_landmarker.task path")
    parser.add_argument(
        "--mirror_input",
        action="store_true",
        help="Flip each frame horizontally before detection if the saved video is mirrored.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frames = []
    frame_idx = 0

    with (
        create_pose_landmarker(args.pose_model) as pose_landmarker,
        create_hand_landmarker(args.hand_model) as hand_landmarker,
        create_face_landmarker(args.face_model) as face_landmarker,
    ):
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if args.mirror_input:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)

            pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_res = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            pose2d = None
            pose3d = None

            if pose_res.pose_landmarks:
                pose2d = landmarks_to_list(pose_res.pose_landmarks[0])

            if pose_res.pose_world_landmarks:
                pose3d = landmarks_to_list(pose_res.pose_world_landmarks[0])

            left_hand, right_hand, left_world, right_world = extract_hand_sides(hand_res)
            
            print(left_hand, right_hand, left_world, right_world)
    

            face_landmarks = None
            if face_res.face_landmarks:
                face_landmarks = landmarks_to_list(face_res.face_landmarks[0])

            frame_data = {
                "frameIndex": frame_idx,
                "pose3D": pose3d,
                "pose2D": pose2d,
                "faceLandmarks": face_landmarks,
                "leftHand": copy.deepcopy(left_hand),
                "rightHand": copy.deepcopy(right_hand),
                "leftHandWorld": copy.deepcopy(left_world),
                "rightHandWorld": copy.deepcopy(right_world),
            }

            print("FRAME", frame_idx)
            print("WRITE LEFT:", left_hand is not None)
            print("WRITE RIGHT:", right_hand is not None)

            frames.append(frame_data)

            frame_idx += 1

    cap.release()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False)

    print(f"Saved {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    main()