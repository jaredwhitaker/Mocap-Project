import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

cap = cv2.VideoCapture('Clips/clip_0.mov')
detector = PoseDetector()

posList = []
center_x_list = np.array([])
left_hip_data = np.array([])
right_hip_data = np.array([])
left_shoulder_data = np.array([])
right_shoulder_data = np.array([])
left_elbow_data = np.array([])
right_elbow_data = np.array([])
left_wrist_data = np.array([])
right_wrist_data = np.array([])


# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_animation.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_width, frame_height))

while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if lmList != 0:
        try:
            center_x = bboxInfo['center'][0]
            center_x_list = np.append(center_x_list, center_x)
            max_load_point = np.argmax(center_x_list)

            left_shoulder = lmList[11][:2]
            left_shoulder_data = np.append(left_shoulder_data, left_shoulder[0])

            right_shoulder = lmList[12][:2]
            right_shoulder_data = np.append(right_shoulder_data, right_shoulder[0])

            left_elbow = lmList[13][:2]
            left_elbow_data = np.append(left_elbow_data, left_elbow[0])

            right_elbow = lmList[14][:2]
            right_elbow_data = np.append(right_elbow_data, right_elbow[0])

            left_wrist = lmList[15][:2]
            left_wrist_data = np.append(left_wrist_data, left_wrist[0])

            right_wrist = lmList[16][:2]
            right_wrist_data = np.append(right_wrist_data, right_wrist[0])

            left_hip = lmList[23][:2]
            left_hip_data = np.append(left_hip_data, left_hip[0])

            right_hip = lmList[24][:2]
            right_hip_data = np.append(right_hip_data, right_hip[0])


            # TODO last hand at lmlist[22][:2]

            # region Significant Points
            # left shoulder
            cv2.circle(img, left_shoulder, 10, (255, 255, 255), cv2.FILLED)

            # right shoulder
            cv2.circle(img, right_shoulder, 10, (0, 255, 255), cv2.FILLED)

            # left elbow
            cv2.circle(img, left_elbow, 10, (0, 0, 255), cv2.FILLED)

            # right elbow
            cv2.circle(img, right_elbow, 10, (0, 0, 255), cv2.FILLED)

            # left wrist
            cv2.circle(img, left_wrist, 10, (0, 0, 255), cv2.FILLED)

            # right wrist
            cv2.circle(img, right_wrist, 10, (0, 0, 255), cv2.FILLED)

            # left hip
            cv2.circle(img, left_hip, 10, (0, 0, 255), cv2.FILLED)

            # right hip
            cv2.circle(img, right_hip, 10, (0, 0, 255), cv2.FILLED)



            # endregion

        except:
            pass

    out.write(img)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # with open('test_animation.txt', 'w') as f:
    #     f.writelines(['%s\n' % item for item in posList])

def calc_rate_of_change(arr):

    # rolling average rate of change for smoothness in graph
    rolling_avg = np.convolve(arr, np.ones(3) / 3, mode='valid')
    diff = rolling_avg[1:] - rolling_avg[:-1]

    return np.abs(diff / rolling_avg[:-1])

    # diff = np.cumsum(arr, dtype=float)
    # diff[n:] = np.abs(diff[n:] - diff[:-n])
    # return diff[n - 1:] / n

def average_speed(torso=True):
    if torso == True:
        return (right_hip_data + left_hip_data) / 2.
    else:
        return (right_shoulder_data + left_shoulder_data) / 2.

data_full = np.array([right_hip_data, left_hip_data, right_shoulder_data, left_shoulder_data,
                      right_elbow_data, left_elbow_data, right_wrist_data, left_wrist_data])

roc_torso = calc_rate_of_change(average_speed(torso=True))
roc_pelvis = calc_rate_of_change(average_speed(torso=False))
roc_lead_elbow = calc_rate_of_change(right_elbow_data)
roc_lead_hand = calc_rate_of_change(right_wrist_data)

contact_point = np.argmax(roc_lead_hand)
full_load = np.argmax(center_x_list)

cap.release()
out.release()
cv2.destroyAllWindows()

# def define_swing_window():
#     # find where the local maxima are the most similar for the start
#     # find where the lead hand is at it's minimum in the swing for the end


def plot_data():

    # sns.lineplot(data=calc_rate_of_change(center_x_list), label='center')
    sns.lineplot(roc_pelvis, label='pelvis')
    sns.lineplot(roc_torso, label='torso')
    sns.lineplot(roc_lead_elbow, label='lead elbow')
    sns.lineplot(roc_lead_hand, label='lead hand')

    plt.axvline(x=full_load, label='Max Load Point', color='black', linestyle=':')
    plt.axvline(x=contact_point, label='Contact Point', color='black', linestyle='--')
    plt.legend()
    plt.show()

def find_peak(arr):
    # peak_index =
    # while !(peak_index < contact_point and peak_index > full_load):
    peaks, _ = find_peaks(arr, distance=10)
    # return peak that is the closest number to the full load
    diff = np.abs(peaks - full_load)
    index = np.argmin(diff)

    return peaks[index]


def print_kinematic_sequence():
    data_dict = {
        'Pelvis': find_peak(roc_pelvis),
        'Torso': find_peak(roc_torso),
        'Lead Elbow': find_peak(roc_lead_elbow),
        'Lead Hand': find_peak(roc_lead_hand)
    }
    sorted_vals = {k: v for k, v in sorted(data_dict.items(), key=lambda item: item[1])}
    keys = list(sorted_vals.keys())

    print('Pro Sequence: 1. Pelvis, 2. Torso, 3. Lead Elbow, 4. Lead Hand')
    print('Hitter Sequence: 1. {0}, 2. {1}, 3. {2}, 4. {3}'.format(keys[0], keys[1], keys[2], keys[3]))

plot_data()
print_kinematic_sequence()
# print(return_velos())
