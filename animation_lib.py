import json
import cv2
import numpy as np
from scipy.interpolate import interp1d
import os


class PoseLoader:
    """
    Handles the loading, parsing, and mathematical interpolation of JSON data.
    """

    # source_fps indicates the frame count the sequence is loaded from
    # target_fps indicates the frame count the video is produced with
    def __init__(self, source_fps=10, target_fps=20):
        # changing the frame ratio could result in speeding up or slowing down; it probably makes sense to make these dynamic based on the input video
        self.source_fps = source_fps
        self.target_fps = target_fps

    def load_sequence(self, file_path):
        """Loads a JSON file and returns (matrix, joint_names, num_frames)."""
        # Check that the file exists
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return None, None

        # open the file
        with open(file_path, "r") as f:
            raw = json.load(f)
        # Handle different formats (flat list or 'frames' dictionary)
        data = raw["frames"] if "frames" in raw else raw

        return self._process_data(data)

    def _process_data(self, frames_data):
        # 1. Extract structure
        frames_dict = {}
        if isinstance(frames_data, list):
            for item in frames_data:
                f_idx = int(item["frame"])
                if f_idx not in frames_dict:
                    frames_dict[f_idx] = []
                frames_dict[f_idx].append(item)
        else:
            frames_dict = frames_data

        # STANDARD JOINT SET DEFINITION
        # Names must include the 'left_' or 'right_' prefix to be read by the Renderer
        base_joints = [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # Generate the names for the fingers of both hands
        hand_suffixes = [
            "thumb_cmc",
            "thumb_mcp",
            "thumb_ip",
            "thumb_tip",
            "index_mcp",
            "index_pip",
            "index_dip",
            "index_tip",
            "middle_mcp",
            "middle_pip",
            "middle_dip",
            "middle_tip",
            "ring_mcp",
            "ring_pip",
            "ring_dip",
            "ring_tip",
            "pinky_mcp",
            "pinky_pip",
            "pinky_dip",
            "pinky_tip",
        ]

        left_hand = [f"left_{s}" for s in hand_suffixes]
        right_hand = [f"right_{s}" for s in hand_suffixes]

        self.target_joints = base_joints + left_hand + right_hand

        joint_map = {name: i for i, name in enumerate(self.target_joints)}
        frame_indices = sorted([int(k) for k in frames_dict.keys()])

        if len(frame_indices) < 2:
            return None, None

        # Matrix creation
        num_frames = len(frame_indices)
        num_joints = len(self.target_joints)
        pose_matrix = np.full((num_frames, num_joints, 2), np.nan)

        for t, f_idx in enumerate(frame_indices):
            key = str(f_idx) if isinstance(frames_data, dict) else f_idx
            items = frames_dict[key]
            for item in items:
                if item["joint"] in joint_map:
                    idx = joint_map[item["joint"]]
                    pose_matrix[t, idx, 0] = item["x"]
                    pose_matrix[t, idx, 1] = item["y"]

        return self._interpolate(pose_matrix, num_frames), self.target_joints

    def _interpolate(self, matrix, original_frames):
        duration = original_frames / self.source_fps
        target_frames = int(duration * self.target_fps)

        orig_time = np.arange(original_frames)
        new_time = np.linspace(0, original_frames - 1, target_frames)

        num_joints = matrix.shape[1]

        # Empty final matrix
        smooth_matrix = np.zeros((target_frames, num_joints, 2))


        # Identify which joints are "children" of the wrists to make them relative
        # Map: child_index -> parent_index
        hierarchy = {}

        # Find the wrist indices
        try:
            idx_lw = self.target_joints.index("left_wrist")
            idx_rw = self.target_joints.index("right_wrist")
        except ValueError:
            # If the wrists are missing from the model, relative logic can't be used
            idx_lw = idx_rw = -1

        for i, name in enumerate(self.target_joints):
            # If it's a left-hand finger
            if "left_" in name and "wrist" not in name and "elbow" not in name and "shoulder" not in name and "hip" not in name and "knee" not in name and "ankle" not in name:
                if idx_lw != -1: hierarchy[i] = idx_lw
            # If it's a right-hand finger
            elif "right_" in name and "wrist" not in name and "elbow" not in name and "shoulder" not in name and "hip" not in name and "knee" not in name and "ankle" not in name:
                if idx_rw != -1: hierarchy[i] = idx_rw


        # Work on a copy to avoid corrupting the original data during the loop
        working_matrix = matrix.copy()

        # For each frame, subtract the parent's position from the child.
        # working_matrix now holds ABSOLUTE coordinates for the body
        # and RELATIVE coordinates for the fingers.
        for child_idx, parent_idx in hierarchy.items():
            # Vectorized operation across all frames (very fast)
            # NaN - Number = NaN (correct, preserves missing data)
            working_matrix[:, child_idx, :] -= matrix[:, parent_idx, :]

        # --- INTERPOLATION AND SMOOTHING ---
        for j in range(num_joints):
            vx, vy = working_matrix[:, j, 0], working_matrix[:, j, 1]
            mask = ~np.isnan(vx)

            if mask.sum() < 2:
                smooth_matrix[:, j, :] = np.nan
                continue

            # Always use linear for stability
            kind = "linear"

            try:
                # Fill_value here repeats the LAST KNOWN RELATIVE POSITION.
                # Example: "The finger is 5cm to the right of the wrist".
                fx = interp1d(orig_time[mask], vx[mask], kind=kind, bounds_error=False, fill_value=(vx[mask][0], vx[mask][-1]))
                fy = interp1d(orig_time[mask], vy[mask], kind=kind, bounds_error=False, fill_value=(vy[mask][0], vy[mask][-1]))

                raw_x = fx(new_time)
                raw_y = fy(new_time)

                # Smoothing (Moving Average)
                smooth_matrix[:, j, 0] = self._smooth_signal(raw_x, window_size=5)
                smooth_matrix[:, j, 1] = self._smooth_signal(raw_y, window_size=5)
            except:
                smooth_matrix[:, j, :] = np.nan

        # --- CONVERSION BACK TO GLOBAL SPACE ---
        # Now we need to add the (interpolated) wrist position back to the (interpolated) fingers
        for child_idx, parent_idx in hierarchy.items():
            # Absolute_Child = Interpolated_Relative_Child + Interpolated_Absolute_Parent
            smooth_matrix[:, child_idx, :] += smooth_matrix[:, parent_idx, :]

        return smooth_matrix


    def _smooth_signal(self, data, window_size=5):
            """Applies a moving average to reduce jitter."""
            # If we have too little data, we can't smooth anything
            if len(data) < window_size:
                return data

            # Create a kernel (a window) of equal weights
            kernel = np.ones(window_size) / window_size

            # 'same' keeps the same length as the original array
            smoothed = np.convolve(data, kernel, mode='same')

            # Convolution at the edges can be imprecise, so we restore the original edges
            # to prevent the hands from "flying off" at the start and end
            smoothed[:window_size] = data[:window_size]
            smoothed[-window_size:] = data[-window_size:]

            return smoothed


class HumanoidRenderer:
    """
    Handles rendering exclusively (graphic style, colors, proportions).
    Version: Enhanced Aesthetics (Face, Trapezius, Full Torso) + Custom Configuration
    """

    def __init__(self, width=800, height=600):
        self.w = width
        self.h = height

        # Base Color Palette
        self.C_BG = (40, 40, 45)
        self.C_SKIN = (220, 230, 255)       # Skin
        self.C_TORSO = (60, 60, 70)         # Outfit
        self.C_HAND_JOINT = (0, 0, 0)       # Black joints

        # New Colors for the Face
        self.C_HEAD = (220, 230, 255)       # Head (same as skin or different)
        self.C_FACE_FEATURES = (60, 60, 70) # Dark gray for eyes/nose/mouth

        # Finger Topology (Unchanged)
        self.fingers = {
            "thumb": ["cmc", "mcp", "ip", "tip"],
            "index": ["mcp", "pip", "dip", "tip"],
            "middle": ["mcp", "pip", "dip", "tip"],
            "ring": ["mcp", "pip", "dip", "tip"],
            "pinky": ["mcp", "pip", "dip", "tip"],
        }

        # PALETTE: 10 distinct colors (BGR format) - custom configuration
        self.finger_colors = {
            "left": {
                "thumb":  (255, 200, 0),
                "index":  (255, 0, 0),
                "middle": (200, 0, 100),
                "ring":   (255, 100, 255),
                "pinky":  (128, 0, 128),
            },
            "right": {
                "thumb":  (0, 200, 255),
                "index":  (0, 0, 255),
                "middle": (0, 100, 255),
                "ring":   (0, 255, 255),
                "pinky":  (50, 205, 50),
            }
        }

        # THICKNESS AND LIMIT SETTINGS
        self.th_finger_bone = 6
        self.r_finger_joint = 6
        self.r_finger_tip = 5
        self.max_bone_len = 150

        # ANATOMICAL DIMENSIONS
        self.head_radius = 60      # Head size (slightly reduced from 70 for proportion)
        self.neck_width_top = 22   # Neck width at head attachment
        self.shoulder_width_off = 15 # How much to "inflate" the shoulders for the torso

    def create_canvas(self):
        return np.full((self.h, self.w, 3), self.C_BG, dtype=np.uint8)

    def _gp(self, joints, name):
        """Get Point: Retrieves safe pixel coordinates."""
        if name in joints and not np.isnan(joints[name]).any():
            x, y = joints[name]
            px = int(x * self.w * 1.3) + (self.w // 2)
            py = int(y * self.h * 1.3) + (self.h // 2)

            margin = 300
            if (px < -margin or px > self.w + margin or py < -margin or py > self.h + margin):
                return None
            return (px, py)
        return None

    def draw_frame(self, img, joints_dict, pose_name):
            # Retrieve main key points
            p_nose = self._gp(joints_dict, "nose")
            p_ls = self._gp(joints_dict, "left_shoulder")
            p_rs = self._gp(joints_dict, "right_shoulder")
            p_lh = self._gp(joints_dict, "left_hip")
            p_rh = self._gp(joints_dict, "right_hip")

            # NECK AND TRAPEZIUS
            if p_nose and p_ls and p_rs:
                neck_base_x = (p_ls[0] + p_rs[0]) // 2
                chin_y = p_nose[1] + int(self.head_radius * 0.8)

                trap_l = (p_ls[0] + int((neck_base_x - p_ls[0])*0.25), p_ls[1] - 5)
                trap_r = (p_rs[0] + int((neck_base_x - p_rs[0])*0.25), p_rs[1] - 5)

                pts_neck = np.array([
                    [p_nose[0] - self.neck_width_top, chin_y],
                    [p_nose[0] + self.neck_width_top, chin_y],
                    trap_r, trap_l
                ], np.int32)

                cv2.fillConvexPoly(img, pts_neck, self.C_SKIN, cv2.LINE_AA)

            # TORSO (Corrected)
            if p_ls and p_rs and p_rh and p_lh:
                shoulder_l = (p_ls[0] - self.shoulder_width_off, p_ls[1])
                shoulder_r = (p_rs[0] + self.shoulder_width_off, p_rs[1])
                hip_r = (p_rh[0] + 5, p_rh[1])
                hip_l = (p_lh[0] - 5, p_lh[1])

                pts_torso = np.array([shoulder_l, shoulder_r, hip_r, hip_l], np.int32)

                cv2.fillConvexPoly(img, pts_torso, self.C_TORSO, cv2.LINE_AA)

                # --- FIX BELOW ---
                # Replaced cv2.LINE_JOIN_ROUND with cv2.LINE_AA
                cv2.polylines(img, [pts_torso], True, self.C_TORSO, 15, cv2.LINE_AA)

            # LIMBS
            self._draw_limb(img, joints_dict, "left_shoulder", "left_elbow", 20)
            self._draw_limb(img, joints_dict, "right_shoulder", "right_elbow", 20)
            self._draw_limb(img, joints_dict, "left_elbow", "left_wrist", 16)
            self._draw_limb(img, joints_dict, "right_elbow", "right_wrist", 16)

            # HEAD AND FACE
            if p_nose:
                cv2.circle(img, p_nose, self.head_radius, self.C_HEAD, -1, cv2.LINE_AA)

                eye_y = p_nose[1] - 15
                eye_off = 18
                cv2.circle(img, (p_nose[0] - eye_off, eye_y), 4, self.C_FACE_FEATURES, -1, cv2.LINE_AA)
                cv2.circle(img, (p_nose[0] + eye_off, eye_y), 4, self.C_FACE_FEATURES, -1, cv2.LINE_AA)

                nx, ny = p_nose[0], p_nose[1] + 5
                pts_nose = np.array([
                    [nx - 6, ny + 8], [nx, ny], [nx + 6, ny + 8]
                ], np.int32)
                cv2.polylines(img, [pts_nose], False, self.C_FACE_FEATURES, 2, cv2.LINE_AA)

                mouth_y = p_nose[1] + 28
                mw = 12
                cv2.line(img, (p_nose[0] - mw, mouth_y), (p_nose[0] + mw, mouth_y), self.C_FACE_FEATURES, 2, cv2.LINE_AA)

            # HANDS
            self._draw_hand(img, joints_dict, "left")
            self._draw_hand(img, joints_dict, "right")

            # TEXT OVERLAY
            if pose_name:
                cv2.putText(
                    img, f"Pose: {pose_name}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
                )

    def _draw_limb(self, img, joints, u, v, th):
        p1, p2 = self._gp(joints, u), self._gp(joints, v)
        if p1 and p2:
            cv2.line(img, p1, p2, self.C_SKIN, th, cv2.LINE_AA)
            cv2.circle(img, p1, th // 2, self.C_SKIN, -1, cv2.LINE_AA)
            cv2.circle(img, p2, th // 2, self.C_SKIN, -1, cv2.LINE_AA)

    def _draw_hand(self, img, joints, side):
        wrist = self._gp(joints, f"{side}_wrist")
        if not wrist:
            return

        cv2.circle(img, wrist, 7, self.C_HAND_JOINT, -1, cv2.LINE_AA)

        for f_name, suffixes in self.fingers.items():
            color = self.finger_colors[side].get(f_name, (160, 160, 160))

            prev = wrist
            for s in suffixes:
                curr = self._gp(joints, f"{side}_{f_name}_{s}")

                if curr:
                    # EUCLIDEAN DISTANCE CALCULATION (original logic)
                    dist = np.linalg.norm(np.array(curr) - np.array(prev))

                    # SANITY CHECK (original logic)
                    if dist < self.max_bone_len:
                        cv2.line(img, prev, curr, color, self.th_finger_bone, cv2.LINE_AA)

                        r = self.r_finger_tip if "tip" in s else self.r_finger_joint
                        cv2.circle(img, curr, r, self.C_HAND_JOINT, -1, cv2.LINE_AA)

                        prev = curr
                    else:
                        break


class AnimationDirector:
    """
    Director: orchestrates the Loader and Renderer to produce the final video.
    """

    def __init__(self, poses_dir, output_file="output_sequence.mp4"):
        self.poses_dir = poses_dir
        self.output_file = output_file

        # Keep the frame rate high as required (30 fps source)
        self.loader = PoseLoader(source_fps=30, target_fps=30)
        self.renderer = HumanoidRenderer()

        # TRANSITION CONFIGURATION
        # How many frames should the smooth transition last?
        # 10 frames at 30fps = 0.33 seconds (fast but visible)
        self.transition_frames = 10

    def _create_transition(self, start_pose, end_pose, steps):
        """
        Generates synthetic frames that linearly interpolate between start_pose and end_pose.
        """
        transition_data = []
        # np.linspace generates numbers from 0 to 1 (e.g., 0.0, 0.1, 0.2 ... 1.0)
        # Exclude 0 (equal to start_pose) to avoid duplicate frames
        alphas = np.linspace(0, 1, steps + 2)[1:-1]

        for alpha in alphas:
            # Vector formula: Result = Start * (1-a) + End * a
            # numpy handles this operation across all joints simultaneously
            mixed_frame = (start_pose * (1 - alpha)) + (end_pose * alpha)
            transition_data.append(mixed_frame)

        return transition_data

    def render_sequence(self, pose_names):
        print(f"--- Starting Sequence Rendering: {pose_names} ---")

        # Data Loading
        segments = []
        common_joints = None

        for name in pose_names:
            path = os.path.join(self.poses_dir, f"{name}.json")
            print(f"Loading: {name}...")

            matrix, joints = self.loader.load_sequence(path)

            if matrix is None:
                print(f" -> SKIPPED: {name}")
                continue

            if common_joints is None:
                common_joints = joints
            elif joints != common_joints:
                print(f" -> WARNING: Different joints in {name}.")

            segments.append((matrix, name))

        if not segments:
            print("No valid data to render.")
            return

        # Video Generation
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_file,
            fourcc,
            self.loader.target_fps,
            (self.renderer.w, self.renderer.h),
        )

        # Approximate frame count (including transition frames)
        total_frames_est = sum(len(s[0]) for s in segments) + (len(segments)-1) * self.transition_frames
        print(f"Generating video (~{total_frames_est} total frames)...")

        frame_counter = 0
        last_pose_frame = None # Memory of the previous cut

        for matrix, current_name in segments:

            # --- TRANSITION PHASE ---
            # If a previous pose exists, generate the bridge
            if last_pose_frame is not None:
                # Get the first frame of the NEW pose
                target_pose_frame = matrix[0]

                # Generate the intermediate frames
                trans_frames = self._create_transition(last_pose_frame, target_pose_frame, self.transition_frames)

                for t_frame in trans_frames:
                    img = self.renderer.create_canvas()

                    # Build the joints dictionary
                    frame_joints = {
                        name: t_frame[idx] for idx, name in enumerate(common_joints)
                    }

                    # Draw the transition.
                    # Use "..." as the name to indicate the passage.
                    self.renderer.draw_frame(img, frame_joints, pose_name="...")

                    out.write(img)
                    cv2.imshow("Preview", img)
                    # Maximum speed for the preview
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        out.release()
                        cv2.destroyAllWindows()
                        return
                    frame_counter += 1

            # --- ACTUAL POSE RENDERING PHASE ---
            for frame_data in matrix:
                img = self.renderer.create_canvas()

                frame_joints = {
                    name: frame_data[idx] for idx, name in enumerate(common_joints)
                }

                self.renderer.draw_frame(img, frame_joints, pose_name=current_name)

                if frame_counter % 30 == 0:
                    print(f"Render: {frame_counter}", end="\r")

                out.write(img)
                cv2.imshow("Preview", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    out.release()
                    cv2.destroyAllWindows()
                    return

                frame_counter += 1

            # Save the last frame of THIS pose to use as the start of the next transition
            last_pose_frame = matrix[-1]

        out.release()
        cv2.destroyAllWindows()
        print(f"\nDone! Video saved to: {self.output_file}")
