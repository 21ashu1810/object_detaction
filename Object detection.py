import cv2
import supervision as sv
from inference import get_model


def process_camera_feed():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    
    model = get_model(model_id="yolov8n-640")

    
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()  

        if not ret:
            print("Error: Could not read frame.")
            break

        
        results = model.infer(frame)[0]

        
        detections = sv.Detections.from_inference(results)

        
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        
        cv2.imshow('YOLOv8 Real-time Object Detection', annotated_frame)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  
    cv2.destroyAllWindows() 

process_camera_feed()
