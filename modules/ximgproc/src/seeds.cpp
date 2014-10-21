/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, Beat Kueng (beat-kueng@gmx.net), Lukas Vogel, Morten Lysgaard
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

/******************************************************************************\
*                            SEEDS Superpixels                                *
*  This code implements the superpixel method described in:                   *
*  M. Van den Bergh, X. Boix, G. Roig, B. de Capitani and L. Van Gool,        *
*  "SEEDS: Superpixels Extracted via Energy-Driven Sampling", ECCV 2012       *
\******************************************************************************/

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <limits>
using namespace std;



//required confidence when double_step is used
#define REQ_CONF 0.1f

#define MINIMUM_NR_SUBLABELS 1


// the type of the histogram and the T array
typedef float HISTN;


namespace cv {
namespace ximgproc {

class SuperpixelSEEDSImpl : public SuperpixelSEEDS
{
public:

    SuperpixelSEEDSImpl(int image_width, int image_height, int image_channels,
            int num_superpixels, int num_levels,  int prior = 2,
           int histogram_bins = 5,  bool double_step = false);

    /* spectral SEEDS */
    SuperpixelSEEDSImpl(const String& filters_file, int total_channels,
            int histogram_bins, int image_width, int image_height,
            int num_superpixels, int num_levels, int prior = 2,
            bool double_step = false);
    void initSpectralSEEDS(int filter_size, int feature_count, int sparse_quantization);
    virtual void generateCodebook(InputArray input);
    virtual void generateCodebook(const String& file);
    virtual void iterateSpectral(InputArray input, int num_iterations = 4);
    void sort_idx(const float* feature, int* idx);
    typedef unsigned int BinaryCodebook;
    inline void quantizeFeature(const float* feature, BinaryCodebook* feature_binary);
    inline void normalizeFeature(float* feature);
    inline void applyFilters(InputArray input, vector<Mat>& filter_output);
    inline void addFilterOutputToCodebook(const vector<Mat>& filter_output,
            int x, int y, int bin_index);
    inline int binaryDistance(const BinaryCodebook* feature1, const BinaryCodebook* feature2) const;
    static inline int popcount(unsigned int v);

    virtual ~SuperpixelSEEDSImpl();

    virtual int getNumberOfSuperpixels() { return nrLabels(seeds_top_level); }

    virtual void iterate(InputArray img, int num_iterations = 4);


    virtual void getLabels(OutputArray labels_out);
    virtual void getLabelContourMask(OutputArray image, bool thick_line = false);

private:
    /* initialization */
    void initialize(int num_superpixels, int num_levels);
    void initImage(InputArray img);
    void assignLabels();
    void computeHistograms(int until_level = -1);
    template<typename _Tp>
    inline void initImageBins(const Mat& img, int max_value);


    /* pixel operations */
    inline void update(int label_new, int image_idx, int label_old);
    //image_idx = y*width+x
    inline void addPixel(int level, int label, int image_idx);
    inline void deletePixel(int level, int label, int image_idx);
    inline bool probability(int image_idx, int label1, int label2, int prior1, int prior2);
    inline int threebyfour(int x, int y, int label);
    inline int fourbythree(int x, int y, int label);

    inline void updateLabels();
    // main loop for pixel updating
    void updatePixels();


    /* block operations */
    void addBlock(int level, int label, int sublevel, int sublabel);
    inline void addBlockToplevel(int label, int sublevel, int sublabel);
    void deleteBlockToplevel(int label, int sublevel, int sublabel);

    // intersection on label1A and intersection_delete on label1B
    // returns intA - intB
    float intersectConf(int level1, int label1A, int label1B, int level2, int label2);

    //main loop for block updates
    void updateBlocks(int level, float req_confidence = 0.0f);

    /* go to next block level */
    int goDownOneLevel();

    //make sure a superpixel stays connected (h=horizontal,v=vertical, f=forward,b=backward)
    inline bool checkSplit_hf(int a11, int a12, int a21, int a22, int a31, int a32);
    inline bool checkSplit_hb(int a12, int a13, int a22, int a23, int a32, int a33);
    inline bool checkSplit_vf(int a11, int a12, int a13, int a21, int a22, int a23);
    inline bool checkSplit_vb(int a21, int a22, int a23, int a31, int a32, int a33);


    //compute initial label for sublevels: level <= seeds_top_level
    //this is an equally sized grid with size nr_h[level]*nr_w[level]
    int computeLabel(int level, int x, int y) {
        return std::min(y / (height / nr_wh[2 * level + 1]), nr_wh[2 * level + 1] - 1) * nr_wh[2 * level]
                + std::min((x / (width / nr_wh[2 * level])), nr_wh[2 * level] - 1);
    }
    inline int nrLabels(int level) const {
        return nr_wh[2 * level + 1] * nr_wh[2 * level];
    }

    int width, height; //image size
    int nr_bins; //number of histogram bins per channel
    int nr_channels; //number of image channels
    bool forwardbackward;

    int seeds_nr_levels;
    int seeds_top_level; // == seeds_nr_levels-1 (const)
    int seeds_current_level; //start with level seeds_top_level-1, then go down
    bool seeds_double_step;
    int seeds_prior;

    // keep one labeling for each level
    vector<int> nr_wh; // [2*level]/[2*level+1] number of labels in x-direction/y-direction

    /* pre-initialized arrays. they are not modified afterwards */
    int* labels_bottom; //labels of level==0
    vector<int*> parent_pre_init;

    unsigned int* image_bins; //[y*width + x] bin index (histogram) of each image pixel

    vector<int*> parent; //[level][label] = corresponding label of block with level+1
    int* labels; //output labels: labels of level==seeds_top_level
    unsigned int* nr_partitions; //[label] how many partitions label has on toplevel

    int histogram_size; //== pow(nr_bins, nr_channels)
    int histogram_size_aligned;
    vector<HISTN*> histogram; //[level][label * histogram_size_aligned + j]
    vector<HISTN*> T; //[level][label] how many pixels with this label

    /* spectral SEEDS */
    String spec_filters_file;
    int spec_filter_size;
    int spec_feature_count;
    int spec_sparse_quantization;
    vector<Mat> spec_filters;
    float* spec_codebook; //TODO: no explicit dynamic memory
    BinaryCodebook* spec_binary_codebook;
    int spec_binary_codebook_entry_size; //length of one entry -> from feature count
    int* spec_tmp_idx; //TODO: avoid this
    bool spec_codebook_exists;


    /* OpenCV containers for our memory arrays. This makes sure memory is
     * allocated & released properly */
    Mat labels_mat;
    Mat labels_bottom_mat;
    Mat nr_partitions_mat;
    Mat image_bins_mat;
    vector<Mat> histogram_mat;
    vector<Mat> T_mat;
    vector<Mat> parent_mat;
    vector<Mat> parent_pre_init_mat;
};

CV_EXPORTS Ptr<SuperpixelSEEDS> createSuperpixelSEEDS(int image_width, int image_height,
        int image_channels, int num_superpixels, int num_levels, int prior, int histogram_bins,
        bool double_step)
{
    return makePtr<SuperpixelSEEDSImpl>(image_width, image_height, image_channels,
            num_superpixels, num_levels, prior, histogram_bins, double_step);
}

CV_EXPORTS_W Ptr<SuperpixelSEEDS> createSuperpixelSpectralSEEDS(
	const String& filters_file, int filter_size, int feature_count,
	int total_channels, int histogram_bins, int sparse_quantization,
	int image_width, int image_height, int num_superpixels, int num_levels,
	int prior, bool double_step)
{
    Ptr<SuperpixelSEEDSImpl> ret = makePtr<SuperpixelSEEDSImpl>(filters_file,
            total_channels, histogram_bins, image_width, image_height,
            num_superpixels, num_levels, prior, double_step);
    if( ret )
        ret->initSpectralSEEDS(filter_size, feature_count, sparse_quantization);
    return ret;
}

SuperpixelSEEDSImpl::SuperpixelSEEDSImpl(int image_width, int image_height, int image_channels,
            int num_superpixels, int num_levels, int prior, int histogram_bins, bool double_step)
{
    width = image_width;
    height = image_height;
    nr_bins = histogram_bins;
    nr_channels = image_channels;
    seeds_double_step = double_step;
    seeds_prior = std::min(prior, 5);

    spec_codebook = NULL;
    spec_binary_codebook = NULL;
    spec_tmp_idx = NULL;
    spec_codebook_exists = false;

    histogram_size = nr_bins;
    for (int i = 1; i < nr_channels; ++i)
        histogram_size *= nr_bins;
    histogram_size_aligned = (histogram_size
        + ((CV_MALLOC_ALIGN / sizeof(HISTN)) - 1)) & -static_cast<int>(CV_MALLOC_ALIGN / sizeof(HISTN));

    initialize(num_superpixels, num_levels);
}

SuperpixelSEEDSImpl::SuperpixelSEEDSImpl(const String& filters_file,
		int total_channels, int histogram_bins, int image_width,
		int image_height, int num_superpixels, int num_levels, int prior,
		bool double_step) {
    width = image_width;
    height = image_height;
    nr_bins = histogram_bins;
    nr_channels = total_channels;
    seeds_double_step = double_step;
    seeds_prior = std::min(prior, 5);

    spec_codebook = NULL;
    spec_binary_codebook = NULL;
    spec_tmp_idx = NULL;
    spec_codebook_exists = false;

    histogram_size = nr_bins;
    histogram_size_aligned = (histogram_size
        + ((CV_MALLOC_ALIGN / sizeof(HISTN)) - 1)) & -static_cast<int>(CV_MALLOC_ALIGN / sizeof(HISTN));

    initialize(num_superpixels, num_levels);

    spec_filters_file = filters_file;
}
void SuperpixelSEEDSImpl::initSpectralSEEDS(int filter_size, int feature_count,
    int sparse_quantization)
{
	CV_Assert(sparse_quantization >= 0 && sparse_quantization < feature_count);
	CV_Assert(filter_size > 0);

	spec_filter_size = filter_size;
	spec_feature_count = feature_count;
    const int bits_per_binary_entry = sizeof(BinaryCodebook) * CHAR_BIT;
    spec_binary_codebook_entry_size = ((spec_feature_count
            + ((bits_per_binary_entry) - 1)) & -bits_per_binary_entry) / bits_per_binary_entry;
	spec_sparse_quantization = sparse_quantization;
    spec_codebook = new float[nr_bins * spec_feature_count];
    if( spec_sparse_quantization )
    {
        //allocate one more than number of bins: the last one is used as temporary variable
        const int num_entries = spec_binary_codebook_entry_size*(nr_bins+1);
        spec_binary_codebook = new BinaryCodebook[num_entries];
        memset(spec_binary_codebook, 0, sizeof(BinaryCodebook)*num_entries);
    }
    spec_tmp_idx = new int[spec_feature_count];

	spec_filters.resize(nr_channels);
	const int filter_size2 = spec_filter_size * spec_filter_size;
    for (int i = 0; i < nr_channels; ++i)
    {
        spec_filters[i] = Mat(spec_feature_count, filter_size2, CV_32FC1);
    }

    //load the filters file
    FILE* file_handle = fopen(spec_filters_file.c_str(), "r");
    CV_Assert(file_handle);

    for (int i = 0; i < spec_feature_count; i++)
    {
        for (int channel = 0; channel < nr_channels; ++channel)
        {
            float* chan_data = (float*) spec_filters[channel].data;
            for (int p = 0; p < filter_size2; ++p)
            {
                fscanf(file_handle, "%f", chan_data + i * filter_size2 + p);
            }
        }
    }
    fclose(file_handle);
}

SuperpixelSEEDSImpl::~SuperpixelSEEDSImpl()
{
    if(spec_codebook) delete[](spec_codebook);
    if(spec_binary_codebook) delete[](spec_binary_codebook);
    if(spec_tmp_idx) delete[](spec_tmp_idx);
}


void SuperpixelSEEDSImpl::iterate(InputArray img, int num_iterations)
{
    initImage(img);

    // block updates
    while (seeds_current_level >= 0)
    {
        if( seeds_double_step )
            updateBlocks(seeds_current_level, REQ_CONF);

        updateBlocks(seeds_current_level);
        seeds_current_level = goDownOneLevel();
    }
    updateLabels();

    for (int i = 0; i < num_iterations; ++i)
        updatePixels();
}
void SuperpixelSEEDSImpl::getLabels(OutputArray labels_out)
{
    labels_out.assign(labels_mat);
}

void SuperpixelSEEDSImpl::initialize(int num_superpixels, int num_levels)
{
    /* enforce parameter restrictions */
    if( num_superpixels < 10 )
        num_superpixels = 10;
    if( num_levels < 2 )
        num_levels = 2;
    int num_superpixels_h = (int)sqrtf((float)num_superpixels * height / width);
    int num_superpixels_w = num_superpixels_h * width / height;
    seeds_nr_levels = num_levels + 1;
    float seeds_wf, seeds_hf;
    do
    {
        --seeds_nr_levels;
        seeds_wf = (float)width / num_superpixels_w / (1<<(seeds_nr_levels-1));
        seeds_hf = (float)height / num_superpixels_h / (1<<(seeds_nr_levels-1));
    } while( seeds_wf < 1.f || seeds_hf < 1.f );
    int seeds_w = (int)ceil(seeds_wf);
    int seeds_h = (int)ceil(seeds_hf);
    CV_Assert(seeds_nr_levels > 0);

    seeds_top_level = seeds_nr_levels - 1;
    image_bins_mat = Mat(height, width, CV_32SC1);
    image_bins = (unsigned int*)image_bins_mat.data;

    // init labels
    labels_mat = Mat(height, width, CV_32SC1);
    labels = (int*)labels_mat.data;
    labels_bottom_mat = Mat(height, width, CV_32SC1);
    labels_bottom = (int*)labels_bottom_mat.data;
    parent.resize(seeds_nr_levels);
    parent_pre_init.resize(seeds_nr_levels);
    nr_wh.resize(2 * seeds_nr_levels);
    int level = 0;
    int nr_seeds_w = (int)floor(width / seeds_w);
    int nr_seeds_h = (int)floor(height / seeds_h);
    nr_wh[2 * level] = nr_seeds_w;
    nr_wh[2 * level + 1] = nr_seeds_h;
    parent_mat.push_back(Mat(nr_seeds_h, nr_seeds_w, CV_32SC1));
    parent[level] = (int*)parent_mat.back().data;
    parent_pre_init_mat.push_back(Mat(nr_seeds_h, nr_seeds_w, CV_32SC1));
    parent_pre_init[level] = (int*)parent_pre_init_mat.back().data;
    for (level = 1; level < seeds_nr_levels; level++)
    {
        nr_seeds_w /= 2; // always partitioned in 2x2 sub-blocks
        nr_seeds_h /= 2;
        parent_mat.push_back(Mat(nr_seeds_h, nr_seeds_w, CV_32SC1));
        parent[level] = (int*)parent_mat.back().data;
        parent_pre_init_mat.push_back(Mat(nr_seeds_h, nr_seeds_w, CV_32SC1));
        parent_pre_init[level] = (int*)parent_pre_init_mat.back().data;
        nr_wh[2 * level] = nr_seeds_w;
        nr_wh[2 * level + 1] = nr_seeds_h;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                parent_pre_init[level - 1][computeLabel(level - 1, x, y)] =
                        computeLabel(level, x, y); // set parent
            }
        }
    }
    nr_partitions_mat = Mat(nr_wh[2 * seeds_top_level + 1],
            nr_wh[2 * seeds_top_level], CV_32SC1);
    nr_partitions = (unsigned int*)nr_partitions_mat.data;

    //preinit the labels (these are not changed anymore later)
    int i = 0;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            labels_bottom[i] = computeLabel(0, x, y);
            ++i;
        }
    }

    // create histogram buffers
    histogram.resize(seeds_nr_levels);
    T.resize(seeds_nr_levels);
    histogram_mat.resize(seeds_nr_levels);
    T_mat.resize(seeds_nr_levels);
    for (level = 0; level < seeds_nr_levels; level++)
    {
        histogram_mat[level] = Mat(nr_wh[2 * level + 1],
                nr_wh[2 * level]*histogram_size_aligned, CV_32FC1);
        histogram[level] = (HISTN*)histogram_mat[level].data;
        T_mat[level] = Mat(nr_wh[2 * level + 1], nr_wh[2 * level], CV_32FC1);
        T[level] = (HISTN*)T_mat[level].data;
    }
}


template<typename _Tp>
void SuperpixelSEEDSImpl::initImageBins(const Mat& img, int max_value)
{
    int img_width = img.size().width;
    int img_height = img.size().height;
    int channels = img.channels();

    for (int y = 0; y < img_height; ++y)
    {
        for (int x = 0; x < img_width; ++x)
        {
            const _Tp* ptr = img.ptr<_Tp>(y, x);
            int bin = 0;
            for (int i = 0; i < channels; ++i)
                bin = bin * nr_bins + (int) ptr[i] * nr_bins / max_value;
            image_bins[y * img_width + x] = bin;
        }
    }
}

/* specialization for float: max_value is assumed to be 1.0f */
template<>
void SuperpixelSEEDSImpl::initImageBins<float>(const Mat& img, int)
{
    int img_width = img.size().width;
    int img_height = img.size().height;
    int channels = img.channels();

    for (int y = 0; y < img_height; ++y)
    {
        for (int x = 0; x < img_width; ++x)
        {
            const float* ptr = img.ptr<float>(y, x);
            int bin = 0;
            for(int i=0; i<channels; ++i)
                bin = bin * nr_bins + std::min((int)(ptr[i] * (float)nr_bins), nr_bins-1);
            image_bins[y*img_width + x] = bin;
        }
    }
}

void SuperpixelSEEDSImpl::initImage(InputArray img)
{
    Mat src = img.getMat();
    int depth = src.depth();
    seeds_current_level = seeds_nr_levels - 2;
    forwardbackward = true;

    assignLabels();

    CV_Assert(src.size().width == width && src.size().height == height);
    CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);
    CV_Assert(src.channels() == nr_channels);

    // initialize the histogram bins from the image
    switch (depth)
    {
    case CV_8U:
        initImageBins<uchar>(src, 1 << 8);
        break;
    case CV_16U:
        initImageBins<ushort>(src, 1 << 16);
        break;
    case CV_32F:
        initImageBins<float>(src, 1);
        break;
    }

    computeHistograms();
}

// adds labeling to all the blocks at all levels and sets the correct parents
void SuperpixelSEEDSImpl::assignLabels()
{
    /* each top level label is partitioned into 4 elements */
    int nr_labels_toplevel = nrLabels(seeds_top_level);
    for (int i = 0; i < nr_labels_toplevel; ++i)
        nr_partitions[i] = 4;

    for (int level = 1; level < seeds_nr_levels; level++)
    {
        memcpy(parent[level - 1], parent_pre_init[level - 1],
                sizeof(int) * nrLabels(level - 1));
    }
}

void SuperpixelSEEDSImpl::computeHistograms(int until_level)
{
    if( until_level == -1 )
        until_level = seeds_nr_levels - 1;
    until_level++;

    // clear histograms
    for (int level = 0; level < seeds_nr_levels; level++)
    {
        int nr_labels = nrLabels(level);
        memset(histogram[level], 0,
                sizeof(HISTN) * histogram_size_aligned * nr_labels);
        memset(T[level], 0, sizeof(HISTN) * nr_labels);
    }

    // build histograms on the first level by adding the pixels to the blocks
    for (int i = 0; i < width * height; ++i)
        addPixel(0, labels_bottom[i], i);

    // build histograms on the upper levels by adding the histogram from the level below
    for (int level = 1; level < until_level; level++)
    {
        for (int label = 0; label < nrLabels(level - 1); label++)
        {
            addBlock(level, parent[level - 1][label], level - 1, label);
        }
    }
}

void SuperpixelSEEDSImpl::updateBlocks(int level, float req_confidence)
{
    int labelA;
    int labelB;
    int sublabel;
    bool done;
    int step = nr_wh[2 * level];

    // horizontal bidirectional block updating
    for (int y = 1; y < nr_wh[2 * level + 1] - 1; y++)
    {
        for (int x = 1; x < nr_wh[2 * level] - 2; x++)
        {
            // choose a label at the current level
            sublabel = y * step + x;
            // get the label at the top level (= superpixel label)
            labelA = parent[level][y * step + x];
            // get the neighboring label at the top level (= superpixel label)
            labelB = parent[level][y * step + x + 1];

            if( labelA == labelB )
                continue;

            // get the surrounding labels at the top level, to check for splitting
            int a11 = parent[level][(y - 1) * step + (x - 1)];
            int a12 = parent[level][(y - 1) * step + (x)];
            int a21 = parent[level][(y) * step + (x - 1)];
            int a22 = parent[level][(y) * step + (x)];
            int a31 = parent[level][(y + 1) * step + (x - 1)];
            int a32 = parent[level][(y + 1) * step + (x)];
            done = false;

            if( nr_partitions[labelA] == 2 || (nr_partitions[labelA] > 2 // 3 or more partitions
                    && checkSplit_hf(a11, a12, a21, a22, a31, a32)) )
            {
                // run algorithm as usual
                float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                if( conf > req_confidence )
                {
                    deleteBlockToplevel(labelA, level, sublabel);
                    addBlockToplevel(labelB, level, sublabel);
                    done = true;
                }
            }

            if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
            {
                // try opposite direction
                sublabel = y * step + x + 1;
                int a13 = parent[level][(y - 1) * step + (x + 1)];
                int a14 = parent[level][(y - 1) * step + (x + 2)];
                int a23 = parent[level][(y) * step + (x + 1)];
                int a24 = parent[level][(y) * step + (x + 2)];
                int a33 = parent[level][(y + 1) * step + (x + 1)];
                int a34 = parent[level][(y + 1) * step + (x + 2)];
                if( nr_partitions[labelB] <= 2 // == 2
                        || (nr_partitions[labelB] > 2 && checkSplit_hb(a13, a14, a23, a24, a33, a34)) )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelB, level, sublabel);
                        addBlockToplevel(labelA, level, sublabel);
                        x++;
                    }
                }
            }
        }
    }

    // vertical bidirectional
    for (int x = 1; x < nr_wh[2 * level] - 1; x++)
    {
        for (int y = 1; y < nr_wh[2 * level + 1] - 2; y++)
        {
            // choose a label at the current level
            sublabel = y * step + x;
            // get the label at the top level (= superpixel label)
            labelA = parent[level][y * step + x];
            // get the neighboring label at the top level (= superpixel label)
            labelB = parent[level][(y + 1) * step + x];

            if( labelA == labelB )
                continue;

            int a11 = parent[level][(y - 1) * step + (x - 1)];
            int a12 = parent[level][(y - 1) * step + (x)];
            int a13 = parent[level][(y - 1) * step + (x + 1)];
            int a21 = parent[level][(y) * step + (x - 1)];
            int a22 = parent[level][(y) * step + (x)];
            int a23 = parent[level][(y) * step + (x + 1)];

            done = false;
            if( nr_partitions[labelA] == 2 || (nr_partitions[labelA] > 2 // 3 or more partitions
                    && checkSplit_vf(a11, a12, a13, a21, a22, a23)) )
            {
                // run algorithm as usual
                float conf = intersectConf(seeds_top_level, labelB, labelA, level, sublabel);
                if( conf > req_confidence )
                {
                    deleteBlockToplevel(labelA, level, sublabel);
                    addBlockToplevel(labelB, level, sublabel);
                    done = true;
                }
            }

            if( !done && (nr_partitions[labelB] > MINIMUM_NR_SUBLABELS) )
            {
                // try opposite direction
                sublabel = (y + 1) * step + x;
                int a31 = parent[level][(y + 1) * step + (x - 1)];
                int a32 = parent[level][(y + 1) * step + (x)];
                int a33 = parent[level][(y + 1) * step + (x + 1)];
                int a41 = parent[level][(y + 2) * step + (x - 1)];
                int a42 = parent[level][(y + 2) * step + (x)];
                int a43 = parent[level][(y + 2) * step + (x + 1)];
                if( nr_partitions[labelB] <= 2 // == 2
                        || (nr_partitions[labelB] > 2 && checkSplit_vb(a31, a32, a33, a41, a42, a43)) )
                {
                    // run algorithm as usual
                    float conf = intersectConf(seeds_top_level, labelA, labelB, level, sublabel);
                    if( conf > req_confidence )
                    {
                        deleteBlockToplevel(labelB, level, sublabel);
                        addBlockToplevel(labelA, level, sublabel);
                        y++;
                    }
                }
            }
        }
    }
}

int SuperpixelSEEDSImpl::goDownOneLevel()
{
    int old_level = seeds_current_level;
    int new_level = seeds_current_level - 1;

    if( new_level < 0 )
        return -1;

    // reset nr_partitions
    memset(nr_partitions, 0, sizeof(int) * nrLabels(seeds_top_level));

    // go through labels of new_level
    int labels_new_level = nrLabels(new_level);
    //the lowest level (0) has 1 partition, all higher levels are
    //initially partitioned into 4
    int partitions = new_level ? 4 : 1;

    for (int label = 0; label < labels_new_level; ++label)
    {
        // assign parent = parent of old_label
        int& cur_parent = parent[new_level][label];
        int p = parent[old_level][cur_parent];
        cur_parent = p;

        nr_partitions[p] += partitions;
    }

    return new_level;
}

void SuperpixelSEEDSImpl::updatePixels()
{
    int labelA;
    int labelB;
    int priorA = 0;
    int priorB = 0;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 2; x++)
        {

            labelA = labels[(y) * width + (x)];
            labelB = labels[(y) * width + (x + 1)];

            if( labelA != labelB )
            {
                int a22 = labelA;
                int a23 = labelB;
                if( forwardbackward )
                {
                    // horizontal bidirectional
                    int a11 = labels[(y - 1) * width + (x - 1)];
                    int a12 = labels[(y - 1) * width + (x)];
                    int a21 = labels[(y) * width + (x - 1)];
                    int a31 = labels[(y + 1) * width + (x - 1)];
                    int a32 = labels[(y + 1) * width + (x)];
                    if( checkSplit_hf(a11, a12, a21, a22, a31, a32) )
                    {
                        if( seeds_prior )
                        {
                            priorA = threebyfour(x, y, labelA);
                            priorB = threebyfour(x, y, labelB);
                        }

                        if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                        {
                            update(labelB, y * width + x, labelA);
                        }
                        else
                        {
                            int a13 = labels[(y - 1) * width + (x + 1)];
                            int a14 = labels[(y - 1) * width + (x + 2)];
                            int a24 = labels[(y) * width + (x + 2)];
                            int a33 = labels[(y + 1) * width + (x + 1)];
                            int a34 = labels[(y + 1) * width + (x + 2)];
                            if( checkSplit_hb(a13, a14, a23, a24, a33, a34) )
                            {
                                if( probability(y * width + x + 1, labelB, labelA, priorB, priorA) )
                                {
                                    update(labelA, y * width + x + 1, labelB);
                                    x++;
                                }
                            }
                        }
                    }
                }
                else
                { // forward backward
                    // horizontal bidirectional
                    int a13 = labels[(y - 1) * width + (x + 1)];
                    int a14 = labels[(y - 1) * width + (x + 2)];
                    int a24 = labels[(y) * width + (x + 2)];
                    int a33 = labels[(y + 1) * width + (x + 1)];
                    int a34 = labels[(y + 1) * width + (x + 2)];
                    if( checkSplit_hb(a13, a14, a23, a24, a33, a34) )
                    {
                        if( seeds_prior )
                        {
                            priorA = threebyfour(x, y, labelA);
                            priorB = threebyfour(x, y, labelB);
                        }

                        if( probability(y * width + x + 1, labelB, labelA, priorB, priorA) )
                        {
                            update(labelA, y * width + x + 1, labelB);
                            x++;
                        }
                        else
                        {
                            int a11 = labels[(y - 1) * width + (x - 1)];
                            int a12 = labels[(y - 1) * width + (x)];
                            int a21 = labels[(y) * width + (x - 1)];
                            int a31 = labels[(y + 1) * width + (x - 1)];
                            int a32 = labels[(y + 1) * width + (x)];
                            if( checkSplit_hf(a11, a12, a21, a22, a31, a32) )
                            {
                                if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                                {
                                    update(labelB, y * width + x, labelA);
                                }
                            }
                        }
                    }
                }
            } // labelA != labelB
        } // for x
    } // for y

    for (int x = 1; x < width - 1; x++)
    {
        for (int y = 1; y < height - 2; y++)
        {

            labelA = labels[(y) * width + (x)];
            labelB = labels[(y + 1) * width + (x)];
            if( labelA != labelB )
            {
                int a22 = labelA;
                int a32 = labelB;

                if( forwardbackward )
                {
                    // vertical bidirectional
                    int a11 = labels[(y - 1) * width + (x - 1)];
                    int a12 = labels[(y - 1) * width + (x)];
                    int a13 = labels[(y - 1) * width + (x + 1)];
                    int a21 = labels[(y) * width + (x - 1)];
                    int a23 = labels[(y) * width + (x + 1)];
                    if( checkSplit_vf(a11, a12, a13, a21, a22, a23) )
                    {
                        if( seeds_prior )
                        {
                            priorA = fourbythree(x, y, labelA);
                            priorB = fourbythree(x, y, labelB);
                        }

                        if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                        {
                            update(labelB, y * width + x, labelA);
                        }
                        else
                        {
                            int a31 = labels[(y + 1) * width + (x - 1)];
                            int a33 = labels[(y + 1) * width + (x + 1)];
                            int a41 = labels[(y + 2) * width + (x - 1)];
                            int a42 = labels[(y + 2) * width + (x)];
                            int a43 = labels[(y + 2) * width + (x + 1)];
                            if( checkSplit_vb(a31, a32, a33, a41, a42, a43) )
                            {
                                if( probability((y + 1) * width + x, labelB, labelA, priorB, priorA) )
                                {
                                    update(labelA, (y + 1) * width + x, labelB);
                                    y++;
                                }
                            }
                        }
                    }
                }
                else
                { // forwardbackward
                    // vertical bidirectional
                    int a31 = labels[(y + 1) * width + (x - 1)];
                    int a33 = labels[(y + 1) * width + (x + 1)];
                    int a41 = labels[(y + 2) * width + (x - 1)];
                    int a42 = labels[(y + 2) * width + (x)];
                    int a43 = labels[(y + 2) * width + (x + 1)];
                    if( checkSplit_vb(a31, a32, a33, a41, a42, a43) )
                    {
                        if( seeds_prior )
                        {
                            priorA = fourbythree(x, y, labelA);
                            priorB = fourbythree(x, y, labelB);
                        }

                        if( probability((y + 1) * width + x, labelB, labelA, priorB, priorA) )
                        {
                            update(labelA, (y + 1) * width + x, labelB);
                            y++;
                        }
                        else
                        {
                            int a11 = labels[(y - 1) * width + (x - 1)];
                            int a12 = labels[(y - 1) * width + (x)];
                            int a13 = labels[(y - 1) * width + (x + 1)];
                            int a21 = labels[(y) * width + (x - 1)];
                            int a23 = labels[(y) * width + (x + 1)];
                            if( checkSplit_vf(a11, a12, a13, a21, a22, a23) )
                            {
                                if( probability(y * width + x, labelA, labelB, priorA, priorB) )
                                {
                                    update(labelB, y * width + x, labelA);
                                }
                            }
                        }
                    }
                }
            } // labelA != labelB
        } // for y
    } // for x
    forwardbackward = !forwardbackward;

    // update border pixels
    for (int x = 0; x < width; x++)
    {
        labelA = labels[x];
        labelB = labels[width + x];
        if( labelA != labelB )
            update(labelB, x, labelA);
        labelA = labels[(height - 1) * width + x];
        labelB = labels[(height - 2) * width + x];
        if( labelA != labelB )
            update(labelB, (height - 1) * width + x, labelA);
    }
    for (int y = 0; y < height; y++)
    {
        labelA = labels[y * width];
        labelB = labels[y * width + 1];
        if( labelA != labelB )
            update(labelB, y * width, labelA);
        labelA = labels[y * width + width - 1];
        labelB = labels[y * width + width - 2];
        if( labelA != labelB )
            update(labelB, y * width + width - 1, labelA);
    }
}

void SuperpixelSEEDSImpl::update(int label_new, int image_idx, int label_old)
{
    //change the label of a single pixel
    deletePixel(seeds_top_level, label_old, image_idx);
    addPixel(seeds_top_level, label_new, image_idx);
    labels[image_idx] = label_new;
}

void SuperpixelSEEDSImpl::addPixel(int level, int label, int image_idx)
{
    histogram[level][label * histogram_size_aligned + image_bins[image_idx]]++;
    T[level][label]++;
}

void SuperpixelSEEDSImpl::deletePixel(int level, int label, int image_idx)
{
    histogram[level][label * histogram_size_aligned + image_bins[image_idx]]--;
    T[level][label]--;
}

void SuperpixelSEEDSImpl::addBlock(int level, int label, int sublevel,
        int sublabel)
{
    parent[sublevel][sublabel] = label;

    HISTN* h_label = &histogram[level][label * histogram_size_aligned];
    HISTN* h_sublabel = &histogram[sublevel][sublabel * histogram_size_aligned];

    //add the (sublevel, sublabel) block to the block (level, label)
    int n = 0;
#if CV_SSSE3
    const int loop_end = histogram_size - 3;
    for (; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time
        __m128 h_labelp = _mm_load_ps(h_label + n);
        __m128 h_sublabelp = _mm_load_ps(h_sublabel + n);
        h_labelp = _mm_add_ps(h_labelp, h_sublabelp);
        _mm_store_ps(h_label + n, h_labelp);
    }
#endif

    //loop peeling
    for (; n < histogram_size; n++)
        h_label[n] += h_sublabel[n];

    T[level][label] += T[sublevel][sublabel];
}

void SuperpixelSEEDSImpl::addBlockToplevel(int label, int sublevel, int sublabel)
{
    addBlock(seeds_top_level, label, sublevel, sublabel);
    nr_partitions[label]++;
}

void SuperpixelSEEDSImpl::deleteBlockToplevel(int label, int sublevel, int sublabel)
{
    HISTN* h_label = &histogram[seeds_top_level][label * histogram_size_aligned];
    HISTN* h_sublabel = &histogram[sublevel][sublabel * histogram_size_aligned];

    //do the reverse operation of add_block_toplevel
    int n = 0;
#if CV_SSSE3
    const int loop_end = histogram_size - 3;
    for (; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time
        __m128 h_labelp = _mm_load_ps(h_label + n);
        __m128 h_sublabelp = _mm_load_ps(h_sublabel + n);
        h_labelp = _mm_sub_ps(h_labelp, h_sublabelp);
        _mm_store_ps(h_label + n, h_labelp);
    }
#endif

    //loop peeling
    for (; n < histogram_size; ++n)
        h_label[n] -= h_sublabel[n];

    T[seeds_top_level][label] -= T[sublevel][sublabel];

    nr_partitions[label]--;
}

void SuperpixelSEEDSImpl::updateLabels()
{
    for (int i = 0; i < width * height; ++i)
        labels[i] = parent[0][labels_bottom[i]];
}

bool SuperpixelSEEDSImpl::probability(int image_idx, int label1, int label2,
        int prior1, int prior2)
{
    unsigned int color = image_bins[image_idx];
    float P_label1 = histogram[seeds_top_level][label1 * histogram_size_aligned + color]
                                                * T[seeds_top_level][label2];
    float P_label2 = histogram[seeds_top_level][label2 * histogram_size_aligned + color]
                                                * T[seeds_top_level][label1];

    if( seeds_prior )
    {
        float p;
        if( prior2 != 0 )
            p = (float) prior1 / prior2;
        else //pathological case
            p = 1.f;
        switch( seeds_prior )
        {
        case 5: p *= p;
            //no break
        case 4: p *= p;
            //no break
        case 3: p *= p;
            //no break
        case 2:
            p *= p;
            P_label1 *= T[seeds_top_level][label2];
            P_label2 *= T[seeds_top_level][label1];
            //no break
        case 1:
            P_label1 *= p;
            break;
        }
    }

    return (P_label2 > P_label1);
}

int SuperpixelSEEDSImpl::threebyfour(int x, int y, int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * x x x x
     * x O o x
     * x x x x
     */

#if CV_SSSE3
    __m128i addp = _mm_set1_epi32(1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp);
    countp = _mm_add_epi32(count3, countp);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;
    count += (labels[(y - 1) * width + x - 1] == label);
    count += (labels[(y - 1) * width + x] == label);
    count += (labels[(y - 1) * width + x + 1] == label);
    count += (labels[(y - 1) * width + x + 2] == label);

    count += (labels[y * width + x - 1] == label);
    count += (labels[y * width + x + 2] == label);

    count += (labels[(y + 1) * width + x - 1] == label);
    count += (labels[(y + 1) * width + x] == label);
    count += (labels[(y + 1) * width + x + 1] == label);
    count += (labels[(y + 1) * width + x + 2] == label);

    return count;
#endif
}

int SuperpixelSEEDSImpl::fourbythree(int x, int y, int label)
{
    /* count how many pixels in a neighborhood of (x,y) have the label 'label'.
     * neighborhood (x=counted, o,O=ignored, O=(x,y)):
     * x x x o
     * x O o x
     * x o o x
     * x x x o
     */

#if CV_SSSE3
    __m128i addp_border = _mm_set_epi32(0, 1, 1, 1);
    __m128i addp_middle = _mm_set_epi32(1, 0, 0, 1);
    __m128i labelp = _mm_set1_epi32(label);
    /* 1. row */
    __m128i data1 = _mm_loadu_si128((__m128i*) (labels + (y-1)*width + x -1));
    __m128i mask1 = _mm_cmpeq_epi32(data1, labelp);
    __m128i countp = _mm_and_si128(mask1, addp_border);
    /* 2. row */
    __m128i data2 = _mm_loadu_si128((__m128i*) (labels + y*width + x -1));
    __m128i mask2 = _mm_cmpeq_epi32(data2, labelp);
    __m128i count1 = _mm_and_si128(mask2, addp_middle);
    countp = _mm_add_epi32(countp, count1);
    /* 3. row */
    __m128i data3 = _mm_loadu_si128((__m128i*) (labels + (y+1)*width + x -1));
    __m128i mask3 = _mm_cmpeq_epi32(data3, labelp);
    __m128i count3 = _mm_and_si128(mask3, addp_middle);
    countp = _mm_add_epi32(count3, countp);
    /* 4. row */
    __m128i data4 = _mm_loadu_si128((__m128i*) (labels + (y+2)*width + x -1));
    __m128i mask4 = _mm_cmpeq_epi32(data4, labelp);
    __m128i count4 = _mm_and_si128(mask4, addp_border);
    countp = _mm_add_epi32(countp, count4);

    countp = _mm_hadd_epi32(countp, countp);
    countp = _mm_hadd_epi32(countp, countp);
    return _mm_cvtsi128_si32(countp);
#else
    int count = 0;
    count += (labels[(y - 1) * width + x - 1] == label);
    count += (labels[(y - 1) * width + x] == label);
    count += (labels[(y - 1) * width + x + 1] == label);

    count += (labels[y * width + x - 1] == label);
    count += (labels[y * width + x + 2] == label);

    count += (labels[(y + 1) * width + x - 1] == label);
    count += (labels[(y + 1) * width + x + 2] == label);

    count += (labels[(y + 2) * width + x - 1] == label);
    count += (labels[(y + 2) * width + x] == label);
    count += (labels[(y + 2) * width + x + 1] == label);

    return count;
#endif
}

float SuperpixelSEEDSImpl::intersectConf(int level1, int label1A, int label1B,
        int level2, int label2)
{
    float sumA = 0, sumB = 0;
    float* h1A = &histogram[level1][label1A * histogram_size_aligned];
    float* h1B = &histogram[level1][label1B * histogram_size_aligned];
    float* h2 = &histogram[level2][label2 * histogram_size_aligned];
    const float count1A = T[level1][label1A];
    const float count2 = T[level2][label2];
    const float count1B = T[level1][label1B] - count2;

    /* this calculates several things:
     * - normalized intersection of a histogram. which is equal to:
     *   sum i over bins ( min(histogram1_i / T1_i, histogram2_i / T2_i) )
     * - intersection A = intersection of (level1, label1A) and (level2, label2)
     * - intersection B =
     *     intersection of (level1, label1B) - (level2, label2) and (level2, label2)
     *   where (level1, label1B) - (level2, label2)
     *     is the substraction of 2 histograms (-> delete_block method)
     * - returns the difference between the 2 intersections: intA - intB
     */

    int n = 0;
#if CV_SSSE3
    __m128 count1Ap = _mm_set1_ps(count1A);
    __m128 count2p = _mm_set1_ps(count2);
    __m128 count1Bp = _mm_set1_ps(count1B);
    __m128 sumAp = _mm_set1_ps(0.0f);
    __m128 sumBp = _mm_set1_ps(0.0f);

    const int loop_end = histogram_size - 3;
    for(; n < loop_end; n += 4)
    {
        //this does exactly the same as the loop peeling below, but 4 elements at a time

        // normal
        __m128 h1Ap = _mm_load_ps(h1A + n);
        __m128 h1Bp = _mm_load_ps(h1B + n);
        __m128 h2p = _mm_load_ps(h2 + n);

        __m128 h1ApC2 = _mm_mul_ps(h1Ap, count2p);
        __m128 h2pC1A = _mm_mul_ps(h2p, count1Ap);
        __m128 maskA = _mm_cmple_ps(h1ApC2, h2pC1A);
        __m128 sum1AddA = _mm_and_ps(maskA, h1ApC2);
        __m128 sum2AddA = _mm_andnot_ps(maskA, h2pC1A);
        sumAp = _mm_add_ps(sumAp, sum1AddA);
        sumAp = _mm_add_ps(sumAp, sum2AddA);

        // del
        __m128 diffp = _mm_sub_ps(h1Bp, h2p);
        __m128 h1BpC2 = _mm_mul_ps(diffp, count2p);
        __m128 h2pC1B = _mm_mul_ps(h2p, count1Bp);
        __m128 maskB = _mm_cmple_ps(h1BpC2, h2pC1B);
        __m128 sum1AddB = _mm_and_ps(maskB, h1BpC2);
        __m128 sum2AddB = _mm_andnot_ps(maskB, h2pC1B);
        sumBp = _mm_add_ps(sumBp, sum1AddB);
        sumBp = _mm_add_ps(sumBp, sum2AddB);
    }
    // merge results (quite expensive)
    float sum1Asse;
    sumAp = _mm_hadd_ps(sumAp, sumAp);
    sumAp = _mm_hadd_ps(sumAp, sumAp);
    _mm_store_ss(&sum1Asse, sumAp);

    float sum1Bsse;
    sumBp = _mm_hadd_ps(sumBp, sumBp);
    sumBp = _mm_hadd_ps(sumBp, sumBp);
    _mm_store_ss(&sum1Bsse, sumBp);

    sumA += sum1Asse;
    sumB += sum1Bsse;
#endif

    //loop peeling
    for (; n < histogram_size; ++n)
    {
        // normal intersect
        if( h1A[n] * count2 < h2[n] * count1A ) sumA += h1A[n] * count2;
        else sumA += h2[n] * count1A;

        // intersect_del
        float diff = h1B[n] - h2[n];
        if( diff * count2 < h2[n] * count1B ) sumB += diff * count2;
        else sumB += h2[n] * count1B;
    }

    float intA = sumA / (count1A * count2);
    float intB = sumB / (count1B * count2);
    return intA - intB;
}

bool SuperpixelSEEDSImpl::checkSplit_hf(int a11, int a12, int a21, int a22, int a31, int a32)
{
    if( (a22 != a21) && (a22 == a12) && (a22 == a32) ) return false;
    if( (a22 != a11) && (a22 == a12) && (a22 == a21) ) return false;
    if( (a22 != a31) && (a22 == a32) && (a22 == a21) ) return false;
    return true;
}
bool SuperpixelSEEDSImpl::checkSplit_hb(int a12, int a13, int a22, int a23, int a32, int a33)
{
    if( (a22 != a23) && (a22 == a12) && (a22 == a32) ) return false;
    if( (a22 != a13) && (a22 == a12) && (a22 == a23) ) return false;
    if( (a22 != a33) && (a22 == a32) && (a22 == a23) ) return false;
    return true;

}
bool SuperpixelSEEDSImpl::checkSplit_vf(int a11, int a12, int a13, int a21, int a22, int a23)
{
    if( (a22 != a12) && (a22 == a21) && (a22 == a23) ) return false;
    if( (a22 != a11) && (a22 == a21) && (a22 == a12) ) return false;
    if( (a22 != a13) && (a22 == a23) && (a22 == a12) ) return false;
    return true;
}
bool SuperpixelSEEDSImpl::checkSplit_vb(int a21, int a22, int a23, int a31, int a32, int a33)
{
    if( (a22 != a32) && (a22 == a21) && (a22 == a23) ) return false;
    if( (a22 != a31) && (a22 == a21) && (a22 == a32) ) return false;
    if( (a22 != a33) && (a22 == a23) && (a22 == a32) ) return false;
    return true;
}

void SuperpixelSEEDSImpl::getLabelContourMask(OutputArray image, bool thick_line)
{
    image.create(height, width, CV_8UC1);
    Mat dst = image.getMat();
    dst.setTo(Scalar(0));

    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int neighbors = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                {
                    int index = y * width + x;
                    int mainindex = j * width + k;
                    if( labels[mainindex] != labels[index] )
                    {
                        if( thick_line || !*dst.ptr<uchar>(y, x) )
                            neighbors++;
                    }
                }
            }
            if( neighbors > 1 )
                *dst.ptr<uchar>(j, k) = (uchar)-1;
        }
    }
}



void SuperpixelSEEDSImpl::addFilterOutputToCodebook(
        const vector<Mat>& filter_output, int x, int y, int bin_index)
{
    float* codebook_bin = spec_codebook + bin_index * spec_feature_count;
    for (int i = 0; i < spec_feature_count; ++i)
        codebook_bin[i] = (float) filter_output[i].at<float>(y, x);

    if( spec_sparse_quantization )
    {
        quantizeFeature(codebook_bin, spec_binary_codebook + spec_binary_codebook_entry_size*bin_index);
    }
    else
    {
        normalizeFeature(codebook_bin);
    }
}

void SuperpixelSEEDSImpl::generateCodebook(InputArray input)
{
    //apply filters
    vector<Mat> filter_output(spec_feature_count);
    applyFilters(input, filter_output);

    int bin = 0;
    if( spec_sparse_quantization )
        bin = 1; //we leave the first codebook entry 0

    //randomly select features from the image
    for (int y = 0; y < height && bin < nr_bins; ++y)
    {
        for (int x = 0; x < width && bin < nr_bins; ++x)
        {
            int rn = (height - y) * width;
            if( rand() % rn < nr_bins - bin )
            {
                addFilterOutputToCodebook(filter_output, x, y, bin++);
            }
        }
    }

    //make sure enough were selected: in most cases this is not needed
    for (; bin < nr_bins; ++bin)
    {
        int x = rand() % width;
        int y = rand() % height;
        addFilterOutputToCodebook(filter_output, x, y, bin++);
    }
    spec_codebook_exists = true;
}

void SuperpixelSEEDSImpl::generateCodebook(const String& file)
{
    FILE* file_handle = fopen(file.c_str(), "r");
    CV_Assert(file_handle);

    for (int bin = 0; bin < nr_bins; ++bin)
    {
        float* codebook_bin = spec_codebook + bin * spec_feature_count;
        for (int m = 0; m < spec_feature_count; ++m)
            fscanf(file_handle, "%f", codebook_bin + m);

        if( spec_sparse_quantization )
        {
            quantizeFeature(codebook_bin, spec_binary_codebook + spec_binary_codebook_entry_size*bin);
        }
        else
        {
            normalizeFeature(codebook_bin);
        }
    }
    fclose(file_handle);

    spec_codebook_exists = true;
}

void SuperpixelSEEDSImpl::quantizeFeature(const float* feature, BinaryCodebook* feature_binary)
{
    sort_idx(feature, spec_tmp_idx);

    for (int i = 0; i < spec_sparse_quantization; ++i) {
        int idx = spec_tmp_idx[i] / (sizeof(BinaryCodebook)*CHAR_BIT);
        int bit = spec_tmp_idx[i] % (sizeof(BinaryCodebook)*CHAR_BIT);
        feature_binary[idx] |= 1 << bit;
    }
}

void SuperpixelSEEDSImpl::normalizeFeature(float* feature)
{
    float norm = 0.f;
    for (int i = 0; i < spec_feature_count; i++)
        norm += feature[i] * feature[i];

    if( norm != 0.f )
        norm = 1.f / sqrtf(norm);

    for (int i = 0; i < spec_feature_count; i++)
        feature[i] *= norm;
}

void SuperpixelSEEDSImpl::applyFilters(InputArray input, vector<Mat>& filter_output)
{
    //extract input channels
    vector<Mat> input_channels(nr_channels);
    if( input.isMatVector() )
    {
        vector<Mat> src_vec;
        input.getMatVector(src_vec);
        int num_channels = 0;
        for (size_t i = 0; i < src_vec.size(); ++i)
        {
            Mat& src = src_vec[i];

            CV_Assert(src.size().width == width && src.size().height == height);

            split(src, &input_channels[num_channels]);
            num_channels += src.channels();
            CV_Assert(num_channels <= num_channels);
        }
        CV_Assert(nr_channels == num_channels);
    }
    else //assume MAT
    {
        Mat src = input.getMat();

        CV_Assert(src.channels() == nr_channels);
        CV_Assert(src.size().width == width && src.size().height == height);

        split(src, &input_channels[0]);
    }
    for (int i = 0; i < nr_channels; ++i)
        input_channels[i].convertTo(input_channels[i], CV_32F);

    //we assume filter_output is either empty or the matrices do not contain
    //any data yet
    filter_output.resize(spec_feature_count);
    const int filter_size2 = spec_filter_size * spec_filter_size;

    //apply filters
    for (int i = 0; i < spec_feature_count; i++)
    {
        /* filter configuration */
        Point anchor = Point(-1, -1);
        double delta = 0;
        int ddepth = CV_32F;
        Mat temp;

        int channel = 0;
        float* filter_data = (float*) spec_filters[channel].data;
        Mat kernel = Mat(spec_filter_size, spec_filter_size, CV_32F, filter_data + i * filter_size2);
        filter2D(input_channels[channel], filter_output[i], ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
        ++channel;
        for (; channel < nr_channels; ++channel)
        {
            filter_data = (float*) spec_filters[channel].data;
            kernel = Mat(spec_filter_size, spec_filter_size, CV_32F, filter_data + i * filter_size2);
            filter2D(input_channels[channel], temp, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);
            filter_output[i] += temp;
        }
    }
}

void SuperpixelSEEDSImpl::iterateSpectral(InputArray input, int num_iterations)
{
    CV_Assert(spec_codebook_exists);

    seeds_current_level = seeds_nr_levels - 2;
    forwardbackward = true;

    assignLabels();

    //apply filters
    vector<Mat> filter_output(spec_feature_count);
    applyFilters(input, filter_output);

    //assign image_bins using filter_output
    BinaryCodebook* tmp_binary_feature = spec_binary_codebook + nr_bins*spec_binary_codebook_entry_size;
    float* feature = new float[spec_feature_count]; //TODO: not dynamic...

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int bin_index = 0;

            for (int i = 0; i < spec_feature_count; ++i)
                feature[i] = (float) filter_output[i].at<float>(y, x);

            //TODO: this is similar code as in codebook generation...
            if( spec_sparse_quantization )
            {
                for (int i = 0; i < spec_binary_codebook_entry_size; ++i)
                    tmp_binary_feature[i] = 0;
                quantizeFeature(feature, tmp_binary_feature);

                //lookup
                int min_distance = spec_feature_count;

                for (int bin = 0; bin < nr_bins; ++bin)
                {
                    int current_distance = binaryDistance(tmp_binary_feature,
                        spec_binary_codebook + spec_binary_codebook_entry_size*bin);
                    //TODO: AND distance function with max..?

                    if( current_distance < min_distance )
                    {
                        min_distance = current_distance;
                        bin_index = bin;
                        if( current_distance == 0 )
                            break;
                    }
                }
            }
            else
            {
                normalizeFeature(feature);

                float max_distance = FLT_MIN;
                for (int bin = 0; bin < nr_bins; ++bin)
                {
                    float current_distance = 0;
                    for (int j = 0; j < spec_feature_count; ++j)
                        current_distance += feature[j] * spec_codebook[bin * spec_feature_count + j];

                    if( current_distance > max_distance )
                    {
                        max_distance = current_distance;
                        bin_index = bin;
                    }
                }
            }
            image_bins[y * width + x] = bin_index;
        }
    }
    delete[] (feature);


    //TODO: duplicate code from SEEDS::iterate...
    computeHistograms();

    // block updates
    while (seeds_current_level >= 0)
    {
        if( seeds_double_step )
            updateBlocks(seeds_current_level, REQ_CONF);

        updateBlocks(seeds_current_level);
        seeds_current_level = goDownOneLevel();
    }
    updateLabels();

    for (int i = 0; i < num_iterations; ++i)
        updatePixels();
}

int SuperpixelSEEDSImpl::popcount(unsigned int v)
{
    //TODO: use popcnt instruction (?) -> but in a portable way!
    //32 bit bitcounting
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}

int SuperpixelSEEDSImpl::binaryDistance(const BinaryCodebook* feature1,
        const BinaryCodebook* feature2) const
{
    int distance = 0;
    /* hamming distance */
    for (int i = 0; i < spec_binary_codebook_entry_size; ++i)
        distance += popcount(feature1[i] ^ feature2[i]);
    return distance;
}

const float *base_arr;

static int compar(const int& a, const int& b)
{
    return base_arr[a] > base_arr[b];
}

void SuperpixelSEEDSImpl::sort_idx(const float* feature, int* idx)
{
    base_arr = feature;

    //sort idx such that largest feature is first

    //TODO: improve this
    for (int i = 0; i < spec_feature_count; ++i)
        idx[i] = i;

    int *end = idx + spec_feature_count;
    std::sort(idx, end, compar);

}


} // namespace ximgproc
} // namespace cv
