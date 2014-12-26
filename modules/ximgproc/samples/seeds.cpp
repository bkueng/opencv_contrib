#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

void trackbarChanged(int pos, void* data);

static void help()
{
    cout << "\nThis program demonstrates SEEDS superpixels using OpenCV class SuperpixelSEEDS\n"
            "Use [space] to toggle output mode\n"
            "\n"
            "It captures either from the camera of your choice: 0, 1, ... default 0\n"
            "Or from an input image\n"
            "Call:\n"
            "./seeds [camera #, default 0]\n"
            "./seeds [input image file]\n" << endl;
}

static const char* window_name = "SEEDS Superpixels";

static bool init = false;

void trackbarChanged(int, void*)
{
    init = false;
}


int main(int argc, char** argv)
{
    VideoCapture cap;
    Mat input_image;
    bool use_video_capture = false;
    help();

    if( argc == 2 )
    {
        if( strlen(argv[1]) == 1 && isdigit(argv[1][0]) )
            cap.open(argv[1][0] - '0'); //video capture
        else
            cap.open(argv[1]); //video file
        use_video_capture = true;
    }
    else if( argc == 1 )
    {
        cap.open(0);
        use_video_capture = true;
    }
    else if( argc >= 2 )
    {
        input_image = imread(argv[1]);
    }

    if( use_video_capture )
    {
        if( !cap.isOpened() )
        {
            cout << "Could not initialize capturing...\n";
            return -1;
        }
    }
    else if( input_image.empty() )
    {
        cout << "Could not open image...\n";
        return -1;
    }

    namedWindow(window_name, 0);
    int num_iterations = 0;
    int prior = 2;
    bool double_step = true;
    int num_superpixels = 400;
    int num_levels = 4;
    int num_histogram_bins = 5;
    int restore_level = 0;
    createTrackbar("Number of Superpixels", window_name, &num_superpixels, 1000, trackbarChanged);
    createTrackbar("Smoothing Prior", window_name, &prior, 5, trackbarChanged);
    createTrackbar("Number of Levels", window_name, &num_levels, 10, trackbarChanged);
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);
    createTrackbar("Video Restore Level", window_name, &restore_level, 8, 0);

    Mat result, mask;
    Ptr<SuperpixelSEEDS> seeds;
    int width, height;
    int display_mode = 0;
    Mat rand_colors;

    for (;;)
    {
        Mat frame;
        if( use_video_capture )
            cap >> frame;
        else
            input_image.copyTo(frame);

        if( frame.empty() )
            break;

        if( !init )
        {
            width = frame.size().width;
            height = frame.size().height;
            seeds = createSuperpixelSEEDS(width, height, frame.channels(), num_superpixels,
                    num_levels, prior, num_histogram_bins, double_step);
            init = true;

            //random colored output
            RNG rng(12345);
            int num_colors = seeds->getNumberOfSuperpixels();
            rand_colors = Mat(num_colors, 1, CV_8UC3);
            for (int i = 0; i < num_colors; ++i)
            {
                rand_colors.at<Vec3b>(0, i) = Vec3b(rng.uniform(0, 255),
                        rng.uniform(150, 230), rng.uniform(180, 255));
            }
            cvtColor(rand_colors, rand_colors, COLOR_HSV2BGR);
        }
        Mat converted;
        cvtColor(frame, converted, COLOR_BGR2HSV);

        double t = (double) getTickCount();
        vector<int> ret;
        //seeds->iterate(converted, num_iterations);
        //*
        ret = seeds->iterateVideo(converted, num_iterations, restore_level, 1);
        for (int i=0; i<ret.size(); ++i) {
            printf("split/merged label: %i\n", ret[i]);
        }
        //*/
        result = frame;

        t = ((double) getTickCount() - t) / getTickFrequency();
        printf("SEEDS segmentation took %i ms with %3i superpixels\n",
                (int) (t * 1000), seeds->getNumberOfSuperpixels());

        /* retrieve the segmentation result */
        Mat labels;
        seeds->getLabels(labels);

        /* get the contours for displaying */
        seeds->getLabelContourMask(mask, false);
        result.setTo(Scalar(0, 0, 255), mask);

        /* display output */
        switch (display_mode)
        {
        case 0: //superpixel contours
            imshow(window_name, result);
            break;
        case 1: //mask
            imshow(window_name, mask);
            break;
        case 2: //labels array
        {
            for (int y=0; y<height; ++y) {
                for(int x=0; x<width; ++x) {
                    result.at<Vec3b>(y, x) = rand_colors.at<Vec3b>(0, labels.at<int>(y, x));
                }
            }
            imshow(window_name, result);
        }
            break;
        }


        int c = waitKey(1);
        if( (c & 255) == 'q' || c == 'Q' || (c & 255) == 27 )
            break;
        else if( (c & 255) == ' ' )
            display_mode = (display_mode + 1) % 3;
    }

    return 0;
}
