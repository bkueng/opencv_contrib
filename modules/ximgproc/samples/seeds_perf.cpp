#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>

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

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])) )
    {
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
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
    int num_iterations = 4;
    int prior = 1;
    bool double_step = false;
    int num_superpixels = 336;
    int num_levels = 4;
    int sparse_quantization = 1;
    createTrackbar("Number of Superpixels", window_name, &num_superpixels, 1000, trackbarChanged);
    createTrackbar("Smoothing Prior", window_name, &prior, 5, trackbarChanged);
    createTrackbar("Number of Levels", window_name, &num_levels, 10, trackbarChanged);
    createTrackbar("Iterations", window_name, &num_iterations, 12, 0);
    createTrackbar("Sparse Quantization", window_name, &sparse_quantization, 26, trackbarChanged);

    Mat result, mask;
    Ptr<SuperpixelSEEDS> seeds;
    int width, height;
    int display_mode = 0;

    for (;;)
    {
        Mat frame;
        if( use_video_capture )
            cap >> frame;
        else
            input_image.copyTo(frame);

        if( frame.empty() )
            break;


        width = frame.size().width;
        height = frame.size().height;

        printf("SQ, hist_bins, bins_init, iter [ms]\n");

        const int num_rep = 25;
        const int num_avg = 20;
        int SQ_count = 6;
        int SQ[] = {0, 1, 2, 3, 4, 5};
        for(int SQ_idx=0; SQ_idx < SQ_count; ++SQ_idx)
        {
            sparse_quantization = SQ[SQ_idx];
            for(int bins = 2; bins <= 10; ++bins)
            {
                int num_histogram_bins = bins;

                vector<double> times_init, times_iter;
                for(int i=0; i<num_rep; ++i) {

                    /*
                    seeds = createSuperpixelSEEDS(width, height, frame.channels(), num_superpixels,
                        num_levels, prior, num_histogram_bins, double_step);
                    //*/

                    //*
                    seeds = createSuperpixelSpectralSEEDS(3, 3, num_histogram_bins*
                            num_histogram_bins*num_histogram_bins
                            , sparse_quantization, width, height, num_superpixels,
                            num_levels, prior, double_step);
                    //seeds->generateCodebook("codebook_color_train_set_desnormalized.txt");
                    seeds->generateCodebook("codebook_color_train_set_shiling.txt");
                    srand(0);
                    //seeds->generateCodebook(frame);
                    //*/

                    vector<double> ret = seeds->iterateSpectral(frame, num_iterations);
                    //vector<double> ret = seeds->iterate(frame, num_iterations);

                    times_init.push_back(ret[0]);
                    times_iter.push_back(ret[1]);
                }
                double avg_init=0., avg_iter=0;
                for(int i=0; i<num_avg; ++i) {
                    avg_init += times_init[i];
                    avg_iter += times_iter[i];
                }
                avg_init /= (double)num_avg;
                avg_iter /= (double)num_avg;
                std::sort(times_init.begin(), times_init.end());
                std::sort(times_iter.begin(), times_iter.end());

                printf("%i, %i, %.2lf, %.2lf\n",
                        sparse_quantization, num_histogram_bins, avg_init*1000., avg_iter*1000.);
            }
        }

        result = frame;


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
            // use the last x bit to determine the color. Note that this does not
            // guarantee that 2 neighboring superpixels have different colors.
            const int num_label_bits = 2;
            labels &= (1 << num_label_bits) - 1;
            labels *= 1 << (16 - num_label_bits);
            imshow(window_name, labels);
        }
            break;
        }


        while(true) {
            int c = waitKey(1);
            if( (c & 255) == 'q' || c == 'Q' || (c & 255) == 27 )
                break;
            else if( (c & 255) == ' ' )
                display_mode = (display_mode + 1) % 3;
        }
    }

    return 0;
}
