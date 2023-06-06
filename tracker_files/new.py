import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import time
from PIL import Image
import os
from datetime import datetime
import csv

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("video1.mp4")

# Lane
Line1 = [(762, 480), (177, 1071)]
Line2 = [(900, 474), (951, 1071)]
Line3 = [(1043, 462), (1826, 1070)]

# Area
area1 = [(762, 480), (177, 1071), (951, 1071), (900, 474)]
area2 = [(900, 474), (951, 1071), (1826, 1070), (1043, 462)]

# capture area
# cap_Line1 = [(285, 954), (1634, 954)]
cap_Line2 = [(483, 774), (1410, 774)]

# counting vehicle
vehicle_count = 0

count = 0
center_points_prev_frame = []
prev_y = []

tracking_objects = {}
track_id = 0
tracking_objects_prev = {}

# directory for save images 
output_directory = "/home/jony/Desktop/work/Deepsort/custom/outputImg"
os.makedirs(output_directory, exist_ok=True)
# txt file directory 
output_file_directory = "/home/jony/Desktop/work/Deepsort/custom/recordfiles"
# csv file directory 
output_csv_file_directory = "/home/jony/Desktop/work/Deepsort/custom/output_csv"

# Time 
time = datetime.now()

# saving in a txt file 
# with open(f"{output_file_directory}/{time}.txt","w") as file:
# saving in a csv file 
with open(f"{output_csv_file_directory}/savedData.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(["Vehicle class","Vehicle id","Box[x,y,w,h]","Center Point[cx,cy]","Detected vehicle images Path"])
    while True:
        ret, frame = cap.read()


        count += 1
        if not ret:
            break
        # 10 frame (skiping frame)
        if count % 3 != 0:
            continue

        # Center points current frame
        center_points_crnt_frame = []
        crnt_y = []
        for line in [Line1, Line2, Line3]:
            cv2.line(frame, line[0], line[1], (255, 0, 0), 3)
        # for area in [area1,area2]:
        #      cv2.polylines(frame,[np.array(area, np.int32)], True,(15,220,10),6)

        # line for captaure area
        # cv2.line(frame, cap_Line1[0], cap_Line1[1], (0, 0, 255), 2)
        cv2.line(frame, cap_Line2[0], cap_Line2[1], (0, 0, 255), 2)


        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)
        class_id = 0

        # vehicle_boxes = []
        # vehicle_centers = []
        crp_vehicle_dtls = []
        vehicle_centers_boxes = []
    
        for box in boxes:
            (x, y, w, h) = box
            class_name = od.classes[class_ids[class_id]]
            class_id += 1
            cx = int((x + x + w)/2)
            cy = int((y+y+h)/2)
            # print(f"{cx,cy}")

            if class_name in ["bicycle", "car", "motorbike", "bus", "truck"]:
                # if cv2.line(np.zeros(frame.shape[:2], dtype=np.uint8), (x, y), (x + w, y + h), 255, 1).any():
                #     # vehicle_boxes.append((x, y, w, h))
                #     # vehicle_centers.append((cx, cy))
                #     vehicle_centers_boxes.append((x,y,w,h,cx,cy,class_name,))
                

                result1 = cv2.pointPolygonTest(
                    np.array(area1, np.int32), (int(cx), int(cy)), False)
                if result1 >= 0:
                    # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + 100, y-30), (0, 255, 0), -1)
                    cv2.putText(frame, str(class_name)+" Lane 1",
                                (x, y-10), 0, .75, (0, 0, 255), 2)
                    center_points_crnt_frame.append((cx, cy,x,y,w,h,class_name))

                result2 = cv2.pointPolygonTest(
                    np.array(area2, np.int32), (int(cx), int(cy)), False)
                if result2 >= 0:
                    # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + 100, y-30), (0, 255, 0), -1)
                    cv2.putText(frame, str(class_name)+" Lane 2",
                                (x, y-10), 0, .75, (0, 0, 255), 2)
                    center_points_crnt_frame.append((cx, cy,x,y,w,h,class_name))

        if count <= 2:
            for pt in center_points_crnt_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1]-pt[1])

                    if distance < 20:
                        tracking_objects[track_id] = pt
                        track_id += 1
        else:

            tracking_objects_copy = tracking_objects.copy()
            center_points_crnt_frame_copy = center_points_crnt_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_crnt_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1]-pt[1])

                    # update Ids position
                    if distance < 30:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_crnt_frame:
                            center_points_crnt_frame.remove(pt)
                        continue
                # Remove Tds lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in center_points_crnt_frame:
                tracking_objects[track_id] = pt
                track_id += 1
        for object_id, pt in tracking_objects.items():
                if cv2.line(np.zeros(frame.shape[:2], dtype=np.uint8), (x, y), (x + w, y + h), 255, 1).any():
                    vehicle_centers_boxes.append([pt[0], pt[1],pt[2],pt[3],pt[4],pt[5],object_id,class_name])
                cv2.circle(frame, (pt[0],pt[1]), 5, (0, 0, 255), -1)
                cv2.putText(frame, str(object_id),(pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
                                
                # vehicle_boxes.append((x, y, w, h))
                # vehicle_centers.append((cx, cy))
                
                    
                    

        # for center_x, center_y,x,y,w,h,obj_id,cls_name in vehicle_centers_boxes:
        #     if center_y > 760 and center_y < 780:
        #         vehicle_count += 1
        #         crp_vehicle_dtls.append([obj_id,cls_name,x,y,w,h,center_x,center_y])

        # if not crp_vehicle_dtls:
        #     continue
        # else:
        #     for i in range(len(crp_vehicle_dtls)):
        #         # file.write(f"Time:{time}, Vehicle class:{crp_vehicle_dtls[i][1]}, Vehicle id:{crp_vehicle_dtls[i][0]}, Box:[x:{crp_vehicle_dtls[i][2]}, y:{crp_vehicle_dtls[i][3]}, h:{crp_vehicle_dtls[i][4]}, w:{crp_vehicle_dtls[i][5]}], Centerpoint:[cx:{crp_vehicle_dtls[i][6]}, cy:{crp_vehicle_dtls[i][7]}]\n")
        #         object_image = frame[crp_vehicle_dtls[i][3]:crp_vehicle_dtls[i][3]+crp_vehicle_dtls[i][4], crp_vehicle_dtls[i][2]:crp_vehicle_dtls[i][2]+crp_vehicle_dtls[i][5]]
        #         object_image_path = os.path.join(output_directory, f"object_{crp_vehicle_dtls[i][0]}_{time}.jpg")
        #         detected_vehicle_image_path = f"{output_directory}/object_{crp_vehicle_dtls[i][0]}_{time}.jpg"
        #         cv2.imwrite(object_image_path, object_image)
        #         data = [[crp_vehicle_dtls[i][1],crp_vehicle_dtls[i][0],[crp_vehicle_dtls[i][2],crp_vehicle_dtls[i][3],crp_vehicle_dtls[i][5],crp_vehicle_dtls[i][4]],[crp_vehicle_dtls[i][6],crp_vehicle_dtls[i][7]],detected_vehicle_image_path]]
        #         writer.writerows(data)
        #         print("Data saved successfully in the CSV file.")

        print(f"croped vehicle: {crp_vehicle_dtls}")

        cv2.putText(frame, "Vehicle Count: {}".format(vehicle_count), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        # # Make a copy of the points
        center_points_prev_frame = center_points_crnt_frame.copy()
        # tracking_objects_prev = tracking_objects.copy()
        key = cv2.waitKey(1)
        if key == 27:
            break


cap.release()
cv2.destroyAllWindows()
