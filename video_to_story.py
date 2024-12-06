import cv2
import torch
import openai
import time

openai.api_key = ''

VIDEO_FILENAME = "recorded_video.mp4"
RECORD_TIME = 10
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_INTERVAL = 5

MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def record_video(filename, duration, frame_width, frame_height):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_objects(video_filename, detection_interval=DETECTION_INTERVAL):
    cap = cv2.VideoCapture(video_filename)
    detected_objects = set()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % detection_interval == 0:
            results = MODEL(frame)
            objects_in_frame = [MODEL.names[int(obj[5])] for obj in results.xyxy[0]]
            detected_objects.update(objects_in_frame)

    cap.release()
    return detected_objects

def generate_story(objects):
    messages = [
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": f"Create a short story that involves the following objects: {', '.join(objects)}."}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def main():
    record_video(VIDEO_FILENAME, RECORD_TIME, FRAME_WIDTH, FRAME_HEIGHT)
    detected_objects = detect_objects(VIDEO_FILENAME)
    if detected_objects:
        story = generate_story(detected_objects)
        print(detected_objects)
        print(story)

if __name__ == "__main__":
    main()
