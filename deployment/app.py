import os
import json
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.secret_key = 'supersecretkey'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

class VideoProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.id2label = {0: "standing", 1: "lying", 2: "eating"}
        self.class_list = list(self.id2label.values())

    def is_point_in_rectangle(self, point, rect):
        px, py = point
        x1, y1, x2, y2 = rect
        return x1 <= px <= x2 and y1 <= py <= y2

    def process_video(self, video_path, roi_rectangle, output_path, 
                     unusual_threshold_seconds=300, unusual_display_seconds=300, 
                     eating_grace_period_seconds=60):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Error: Could not open video."

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        eating_log = {}
        transition_log = {}
        completed_eating_sessions = {}
        current_frame = 0
        last_percentage = -1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            if total_frames > 0:
                percentage = (current_frame / total_frames) * 100
                int_percentage = int(percentage)
                if int_percentage > last_percentage:
                    print(f"\rProcessing: {int_percentage}%", end='')
                    last_percentage = int_percentage

            results = self.model.track(source=frame, conf=0.8, persist=True, stream=False, save=False)
            if not results or results[0].boxes is None or results[0].boxes.cls is None or results[0].boxes.id is None:
                out.write(frame)
                continue

            boxes = results[0].boxes
            class_ids = boxes.cls.tolist()
            track_ids = list(map(int, boxes.id.tolist()))
            xyxy = boxes.xyxy.cpu().numpy()

            if not class_ids or not track_ids:
                continue

            count_by_class = {label: 0 for label in self.class_list}
            valid_indices = []

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if self.is_point_in_rectangle((center_x, center_y), roi_rectangle):
                    valid_indices.append(i)
                    class_id = int(class_ids[i])
                    label = self.id2label[class_id]
                    count_by_class[label] += 1

            total_pigs = sum(count_by_class.values())

            grace_period_frames = eating_grace_period_seconds * fps
            for i in valid_indices:
                class_id = int(class_ids[i])
                label = self.id2label[class_id]
                track_id = int(track_ids[i])

                if track_id not in transition_log:
                    transition_log[track_id] = {
                        'current_class': label,
                        'last_class': None,
                        'last_change_frame': current_frame,
                        'eating_frames': 0,
                        'is_unusual': False
                    }
                    completed_eating_sessions[track_id] = []

                current_info = transition_log[track_id]
                if current_info['current_class'] != label:
                    if (current_info['current_class'] == "eating" and 
                        label != 'eating' and
                        (current_frame - current_info['last_change_frame']) > grace_period_frames and 
                        current_info['eating_frames'] > 0):
                        completed_eating_sessions[track_id].append(current_info['eating_frames'])
                        eating_log[track_id] = eating_log.get(track_id, 0) + current_info['eating_frames']
                        current_info['eating_frames'] = 0
                    current_info['last_class'] = current_info['current_class']
                    current_info['current_class'] = label
                    current_info['last_change_frame'] = current_frame

                if label == "eating":
                    current_info['eating_frames'] += 1
                elif (current_info['last_class'] == "eating" and 
                      (current_frame - current_info['last_change_frame']) <= grace_period_frames):
                    current_info['eating_frames'] += 1

                if current_info['eating_frames'] / fps >= unusual_threshold_seconds:
                    current_info['is_unusual'] = True
                elif (current_info['current_class'] != "eating" and 
                      current_info['last_class'] == "eating" and 
                      (current_frame - current_info['last_change_frame']) > unusual_display_seconds * fps):
                    current_info['is_unusual'] = False

                eating_log[track_id] = eating_log.get(track_id, 0) + current_info['eating_frames']

                x1, y1, x2, y2 = map(int, xyxy[i])
                text = f"{label} ID {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if track_id in transition_log and transition_log[track_id]['eating_frames'] > 0:
                    eating_frames = transition_log[track_id]['eating_frames']
                    eating_seconds = eating_frames / fps
                    cv2.putText(frame, f"Time: {eating_seconds:.2f}s", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if track_id in transition_log and transition_log[track_id]['is_unusual']:
                    cv2.putText(frame, "Unusual", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            x1, y1, x2, y2 = roi_rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            y_offset = 30
            cv2.putText(frame, "Pig Count in ROI:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            for i, label in enumerate(self.class_list):
                count = count_by_class[label]
                text = f"{label}: {count}"
                cv2.putText(frame, text, (10, y_offset + 30 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Total pigs in ROI: {total_pigs}", 
                        (10, y_offset + 30 + len(self.class_list) * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            out.write(frame)

        for track_id, info in transition_log.items():
            if info['eating_frames'] > 0:
                completed_eating_sessions[track_id].append(info['eating_frames'])
                eating_log[track_id] = eating_log.get(track_id, 0) + info['eating_frames']

        cap.release()
        out.release()
        print("\nProcessing complete.")
        return True, "Video processed successfully."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            try:
                roi_x1 = int(request.form.get('roi_x1', 88))
                roi_y1 = int(request.form.get('roi_y1', 20))
                roi_x2 = int(request.form.get('roi_x2', 2671))
                roi_y2 = int(request.form.get('roi_y2', 1424))
                unusual_threshold = int(request.form.get('unusual_threshold', 30))
                unusual_display = int(request.form.get('unusual_display', 20))
                eating_grace = int(request.form.get('eating_grace', 30))
            except ValueError:
                flash('Invalid input parameters. Using defaults.')
                roi_x1, roi_y1, roi_x2, roi_y2 = 88, 20, 2671, 1424
                unusual_threshold, unusual_display, eating_grace = 30, 20, 30

            roi_rectangle = [roi_x1, roi_y1, roi_x2, roi_y2]
            output_filename = f"output_{uuid.uuid4().hex}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            # Initialize processor with model path
            model_path = "/home/jacktran/project_2/output/deployment/models/best.pt"
            processor = VideoProcessor(model_path)
            success, message = processor.process_video(
                video_path, roi_rectangle, output_path,
                unusual_threshold_seconds=unusual_threshold,
                unusual_display_seconds=unusual_display,
                eating_grace_period_seconds=eating_grace
            )

            if success:
                return render_template('result.html', video_url=output_filename)
            else:
                flash(message)
                return redirect(request.url)

    return render_template('index.html')

@app.route('/outputs/<filename>')
def serve_video(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)