import cv2
import cvlib as cv
import numpy as np
import urllib.request
import time

def main():
    print("--- AI Animal Finder ---")
    

    esp32_ip = input("Enter the ESP32-CAM's IP address (e.g., 192.168.4.1): ")
    stream_url = f'http://{esp32_ip}:81/stream'
    
    
    target_animal = input("Which animal do you want to find? (e.g., 'dog', 'cat', 'bird'): ").lower()
    
    print(f"\n[INFO] Starting stream from: {stream_url}")
    print(f"[INFO] Looking for: {target_animal}")
    print("[INFO] Press 'q' to quit.")

    stream = None
    try:

        stream = urllib.request.urlopen(stream_url)
    except Exception as e:
        print(f"\n[ERROR] Could not open stream: {e}")
        print("Please check the IP address and make sure the robot is on.")
        return

    bytes_buffer = b''
    

    while True:
        try:
            
            bytes_buffer += stream.read(1024)
            
            
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9') 
            
          
            if a != -1 and b != -1:
                
                jpg = bytes_buffer[a:b+2]
                
                bytes_buffer = bytes_buffer[b+2:]
                
                
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    print("[WARN] Received empty frame. Skipping.")
                    continue

               
                bbox, labels, conf = cv.detect_common_objects(frame, model='yolov4-tiny')
                
                found_target = False
                
                
                for label, c in zip(labels, conf):
                    if label == target_animal:
                        print(f"!!! TARGET FOUND: {label} (Confidence: {c*100:.2f}%) !!!")
                        found_target = True

                
                output_frame = cv.object_detection.draw_bbox(frame, bbox, labels, conf, write_conf=True)
                
                if found_target:
                    cv2.putText(output_frame, "!!! TARGET ACQUIRED !!!", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                
                cv2.imshow("AI Animal Finder", output_frame)

                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quitting...")
                    break

        except Exception as e:
            print(f"[ERROR] Stream error: {e}")
            time.sleep(2) 
            
            
            try:
                stream.close()
                stream = urllib.request.urlopen(stream_url)
                bytes_buffer = b''
            except Exception as e:
                print(f"[ERROR] Reconnect failed: {e}")
                time.sleep(5)

   
    cv2.destroyAllWindows()
    if stream:
        stream.close()

if __name__ == '__main__':
    main()
