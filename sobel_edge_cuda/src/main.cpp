#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Initialize video capture from the default camera
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera stream." << std::endl;
        return -1;
    }

    // Set desired frame resolution (optional, use dimensions from Plan.txt)
    int frame_width = 640;
    int frame_height = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);

    // Check if the resolution was set correctly (some cameras might not support it)
    frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Actual frame resolution: " << frame_width << "x" << frame_height << std::endl;


    cv::Mat frame, gray;

    std::cout << "Press ESC to exit." << std::endl;

    while (true) {
        cap >> frame; // Capture a frame
        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame." << std::endl;
            break;
        }

        // Convert to grayscale; necessary for Sobel later
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Display the grayscale image
        cv::imshow("Grayscale Feed", gray);

        // Break the loop if the user presses the ESC key (ASCII 27)
        if (cv::waitKey(1) == 27) {
            std::cout << "ESC key pressed. Exiting..." << std::endl;
            break;
        }
    }

    // Clean up: Release the camera and destroy windows
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Camera released and windows closed." << std::endl;

    return 0;
} 