import io, pygame, rpc, serial, serial.tools.list_ports, sys
import cv2
import numpy as np
from ultralytics import YOLO
import struct

# Fix Python 2.x.
try: input = raw_input
except NameError: pass

print("\nAvailable Ports:\n")
for port, desc, hwid in serial.tools.list_ports.comports():
    print("{} : {} [{}]".format(port, desc, hwid))
sys.stdout.write("\nPlease enter a port name: ")
sys.stdout.flush()
interface = rpc.rpc_usb_vcp_master(port=input())
print("")
sys.stdout.flush()

# Initialize the YOLO model
model = YOLO('yolov8n-pose.pt')  # load an official detection model

# Initialize Pygame
pygame.init()
screen_w = 640
screen_h = 480
try:
    screen = pygame.display.set_mode((screen_w, screen_h), flags=pygame.RESIZABLE)
except TypeError:
    screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Frame Buffer")
clock = pygame.time.Clock()

def jpg_frame_buffer_cb(data):
    sys.stdout.flush()

    # Unpack blob data
    num_blobs = struct.unpack("<I", data[:4])[0]
    blob_stats = []
    offset = 4
    for _ in range(num_blobs):
        blob_stats.append(struct.unpack("<4I", data[offset:offset+16]))
        offset += 16
    
    
    # Convert the remaining data to an image
    image_data = data[offset:]
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Resize the image 160x120 to 640x480
    
    image = cv2.resize(image, (640, 480))
    
    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)
    
    # Run YOLOv8 tracking on the flipped frame
    results = model.track(image, persist=True)
    keypoint_indices = [0, 9, 10]
        # Print blob coordinates
    print("Blob Coordinates:")
    for blob in blob_stats:
        print("x: {}, y: {}, w: {}, h: {}".format(blob[0], blob[1], blob[2], blob[3]))

    keypoint_coordinates = []
    blob_coordinates = []

    for blob in blob_stats:
        x, y, w, h = blob
        
        # 원본 이미지 크기 (160x120)
        original_width, original_height = 160, 120
        
        # 새로운 이미지 크기 (640x480)
        new_width, new_height = 640, 480
        
        # 리사이징 비율 계산
        width_ratio = new_width / original_width
        height_ratio = new_height / original_height
        
        # 좌표 및 크기 리사이징
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)
    
        blob_coordinates.append((x, y, w, h))
    

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Convert to numpy array
            for person_keypoints in keypoints:
                if len(person_keypoints) > 0:  # Check if keypoints are not empty
                    for i in keypoint_indices:
                        if i < len(person_keypoints):  # Check if the index is within the range
                            point = person_keypoints[i]
                            x, y = int(point[0]), int(point[1])  # Extract x, y coordinates
                            keypoint_coordinates.append([x,y])
        else:
            print("No keypoints detected")

    

    # Visualize the results on the flipped frame
    annotated_frame = results[0].plot()
    


    for blob in blob_coordinates:
        x, y, w, h = blob
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle around blob

    for point in keypoint_coordinates:
        x, y = point
        cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)  # Draw red circle at keypoint position

    # Check if the keypoint is inside any of the blob bounding boxes
    max_deviation = 20

    for blob in blob_coordinates:
        bx, by, bw, bh = blob
        for point in keypoint_coordinates:
            x, y = point
            # Check if the point is within the blob region with a maximum deviation of 10 pixels
            if (bx - max_deviation) <= x <= (bx + bw + max_deviation) and (by - max_deviation) <= y <= (by + bh + max_deviation):
                # If the keypoint is inside the blob bounding box (with deviation), draw "Smoking Detected" text
                cv2.putText(annotated_frame, "Smoking Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # Resize the annotated frame to 640x480 (although it should already be this size)
    annotated_frame = cv2.resize(annotated_frame, (640, 480))

    # Display the annotated frame using OpenCV
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    
    # Convert the flipped image (without YOLO annotations) to a format suitable for Pygame
    flipped_image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
    flipped_image_rgb = np.rot90(flipped_image_rgb)
    flipped_image_rgb = pygame.surfarray.make_surface(flipped_image_rgb)
    
    try:
        screen.blit(pygame.transform.scale(flipped_image_rgb, (screen_w, screen_h)), (0, 0))
        pygame.display.update()
        clock.tick()
    except pygame.error: pass

    print(clock.get_fps())

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        pygame.quit()
        quit()

while(True):
    sys.stdout.flush()
    result = interface.call("jpeg_image_stream", "sensor.GRAYSCALE,sensor.QQVGA")
    if result is not None:
        interface.stream_reader(jpg_frame_buffer_cb, queue_depth=8)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
