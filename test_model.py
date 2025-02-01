import os
import cv2
from ultralytics import YOLO

# Path to your trained YOLOv8 model
model_path = "Model/best.pt"  # XXXXXXXXXXXXX - Adjust the path if needed

# Load the YOLOv8 model
model = YOLO(model_path)

# Function to process an image
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Perform inference on the image
    results = model(img)

    # The results object is a list, so we need to access the first result
    result = results[0]  # Access the first result

    # Draw the results (bounding boxes, labels, etc.) on the image2
    image_with_boxes = result.plot()  # This adds bounding boxes to the image

    # Show the result
    cv2.imshow("Processed Image", image_with_boxes)
    cv2.waitKey(0)  # Wait for any key to close the window
    cv2.destroyAllWindows()

    # Optionally, save the result if needed
    output_image_path = "RESULTS/processed_image.jpg"
    if not os.path.exists("RESULTS"):
        os.makedirs("RESULTS")
    cv2.imwrite(output_image_path, image_with_boxes)
    print(f"Processed image saved to: {output_image_path}")

# Function to process a video
def process_video(video_path):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video details (frame width, height, and frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize the video writer to save the output video
    output_video_path = "RESULTS/output_video.avi"  # You can change the output file name or format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if necessary
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame.")
            break  # End of video if no frame is returned

        # Perform inference on the current frame
        results = model(frame)

        # The results object is a list, so we need to access the first result
        result = results[0]  # Access the first result

        # Draw the results (bounding boxes, labels, etc.) on the frame
        frame_with_boxes = result.plot()  # This adds bounding boxes to the frame

        # Write the frame with bounding boxes to the output video
        out.write(frame_with_boxes)

        # Optionally, display the frame in a window (for debugging)
        cv2.imshow("Processed Video", frame_with_boxes)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break  # Exit the loop if 'q' is pressed

    # Release resources after processing the video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to: {output_video_path}")

# Main function to choose input type
def main():
    # Prompt user to choose input type
    input_type = input("Enter '1' to process an image or '2' to process a video: ").strip()

    if input_type == '1':
        image_path = input("Enter the path to the image file (e.g., 'image.jpg'): ").strip()
        process_image(image_path)

    elif input_type == '2':
        video_path = input("Enter the path to the video file (e.g., 'video.mp4'): ").strip()
        process_video(video_path)

    else:
        print("Invalid input. Please enter '1' for image or '2' for video.")

if __name__ == "__main__":
    main()
