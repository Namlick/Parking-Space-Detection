import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_RED
from tabulate import tabulate
from time import sleep

# number of parking spaces
size = 19
##########################

row1 = [0]*size
for z in range(size):
    row1[z] = z+1
row2 = [0]*size

bool1 = [0]*size
for z in range(size):
    bool1[z] = z+1
bool2 = [0]*size

class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []

    def detect_motion(self):
        capture = open_cv.VideoCapture(self.video)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        while capture.isOpened():
            result, frame = capture.read()
            if frame is None:
                break

            if not result:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))

            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.__apply(grayed, index, c)

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if statuses[index] else COLOR_RED
                draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)
                
                row2Prev = row2
                if color == COLOR_RED:
                    row2[index] = "full"
                    bool2[index] = 1
                    #if row2[index] == row2Prev[index]:
                    #    print(row2)
                elif color == COLOR_GREEN:
                    row2[index] = "empty"
                    bool2[index] = 0
                array = [bool1,bool2]
                table = [['Parking Space #','Status']]
                print(tabulate(table))
                for i in range(size):
                    print(f"      {row1[i]}          {row2[i]}")
                #print(row2)
                with open('Output_Data/Output_formatted.txt','w') as f:
                    f.write(tabulate(table))
                    f.write('\n')
                    for i in range(size):
                        f.write(f"      {row1[i]}          {row2[i]}\n")
                with open('Output_Data/Output_raw.txt','w') as f:
                    for i in range(size):
                        f.write(f"{row1[i]} {row2[i]}\n")
                with open('Output_Data/Output_raw.txt','r') as r:
                    out = r.read()
                    empty = out.count("empty")
                    full = size-empty
                print(f"Number of empty spots:  {empty}")
                print(f"Number of full spots:   {size-empty}")
                with open('Output_Data/Output_formatted.txt','a') as f:
                    f.write(f"\nNumber of empty spots:  {empty}\n")
                    f.write(f"Number of full spots:   {size-empty}\n")
                with open('Output_Data/Output_raw.txt','a') as f:
                    f.write(f"\n{empty}\n")
                    f.write(f"{full}\n")
                with open('Output_Data/Array.txt','w') as f:
                    f.write(f"{array[0]}\n")
                    f.write(f"{array[1]}\n")
                    f.write(f"{empty}\n")
                    f.write(f"{full}\n")

                
                #row1 = np.array([index])
                #row2 = np.array([0*index])
                #row2[index] = color
                #if color == COLOR_GREEN:
                #    print(f"{index} is empty")
                #    row2[index] = "0"
                #elif color == COLOR_RED:
                #    print(f"{index} is full")
                #    row2[index] = "1"
                #print(f"{row1}  {row2}")
                #print("---------------")


            open_cv.imshow(str(self.video), new_frame)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        logging.debug("status: %s", status)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass
