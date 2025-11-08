import cv2
import os

# Enter the sign you want to capture
sign = input("Enter the sign/letter to capture: ").upper()
num_samples = 200  # number of images per sign
save_path = f"../two_hand_dataset/{sign}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"Starting data collection for sign: {sign}")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collect Data - Press 'q' to quit", frame)

    # Save frames
    if count < num_samples:
        file_name = os.path.join(save_path, f"{sign}_{count}.jpg")
        cv2.imwrite(file_name, frame)
        count += 1
        print(f"Captured {count}/{num_samples}")

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Data collection for {sign} completed!")
