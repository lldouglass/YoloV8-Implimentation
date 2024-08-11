from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import io
import torch
from yolov5 import YOLOv5  # Import YOLOv5

app = FastAPI()

# Load the YOLOv5 model
model = YOLOv5("C:\Users\logan\Field Session Pt2\Field_Session_Project\exported\train.py", device='cuda' if torch.cuda.is_available() else 'cpu')

@app.post("/process-video/") 
async def process_video(file: UploadFile = File(...)):
    # Load the video from the uploaded file
    video_stream = io.BytesIO(await file.read())
    cap = cv2.VideoCapture(video_stream)
    
    # Prepare to process video and write output
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (YOLOv5 requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = model.predict(rgb_frame)
        
        # Draw boxes and labels on the frame
        for detection in results.xyxy[0]:
            # Extract detection data
            xmin, ymin, xmax, ymax, conf, cls = detection
            # Draw rectangle for detected object
            start_point = (int(xmin), int(ymin))
            end_point = (int(xmax), int(ymax))
            color = (255, 0, 0)  # Red color in BGR
            thickness = 2
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        
        frames.append(frame)
    
    # Save processed frames to a new video
    out_stream = io.BytesIO()
    out = cv2.VideoWriter(out_stream, cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()
    
    # Reset stream position to the beginning
    out_stream.seek(0)
    return StreamingResponse(out_stream, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
