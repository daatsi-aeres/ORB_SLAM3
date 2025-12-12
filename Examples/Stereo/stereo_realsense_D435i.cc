/**
* This file is part of ORB-SLAM3
* ... (License Header) ...
*/

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>

#include <condition_variable>

#include <opencv2/core/core.hpp>

#include <librealsense2/rs.hpp>
#include "librealsense2/rsutil.h"

#include <System.h>
#include "PerfLogger.h" // Ensure this is here

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <numeric>

using namespace std;

// ===================================================================
// START: Your Custom DIP Pipeline 
// ===================================================================

struct ClaheParams {
    bool do_clahe = false;
    double clip_limit = 4.0;
    cv::Size kernel_size = cv::Size(8, 8);
};

ClaheParams choose_clahe_params_cpp(const cv::Mat& gray)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    double var = stddev[0] * stddev[0];

    double min_val, max_val;
    cv::minMaxLoc(gray, &min_val, &max_val);
    double range_ = max_val - min_val;

    ClaheParams params;
    if (var < 150.0 && range_ < 40.0) {
        params.clip_limit = 6.0;
        params.kernel_size = cv::Size(4, 4);
        params.do_clahe = true;
    } else if (var < 300.0) {
        params.clip_limit = 3.0;
        params.kernel_size = cv::Size(8, 8);
        params.do_clahe = true;
    } else {
        params.clip_limit = 2.0;
        params.kernel_size = cv::Size(12, 12);
        params.do_clahe = true;
    }
    return params;
}

cv::Mat guided_filter_cpp(const cv::Mat& I_in, const cv::Mat& p_in, int r, double eps)
{
    cv::Mat I, p;
    I_in.convertTo(I, CV_32F, 1.0 / 255.0);
    p_in.convertTo(p, CV_32F, 1.0 / 255.0);
    cv::Size kSize(2 * r + 1, 2 * r + 1);
    cv::Mat mean_I, mean_p, mean_Ip, mean_II;
    
    cv::boxFilter(I, mean_I, CV_32F, kSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(p, mean_p, CV_32F, kSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(I.mul(p), mean_Ip, CV_32F, kSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(I.mul(I), mean_II, CV_32F, kSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);
    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_32F, kSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);
    cv::boxFilter(b, mean_b, CV_32F, kSize, cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    cv::Mat q = mean_a.mul(I) + mean_b;
    cv::Mat result;
    q.convertTo(result, CV_8U, 255.0);
    return result;
}

double calc_blur_score(const cv::Mat& img) {
    cv::Mat lap;
    cv::Laplacian(img, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0]; 
}

cv::Mat get_motion_psf(int length, double angle_deg) {
    cv::Mat psf = cv::Mat::zeros(length, length, CV_32F);
    int center = length / 2;
    double rad = angle_deg * CV_PI / 180.0;
    double vec_len = (double)(length / 2 - 1);
    int x0 = center + (int)(vec_len * cos(rad));
    int y0 = center + (int)(vec_len * sin(rad));
    int x1 = center - (int)(vec_len * cos(rad));
    int y1 = center - (int)(vec_len * sin(rad));
    cv::line(psf, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(1.0), 1);
    cv::Scalar sum_val = cv::sum(psf);
    psf = psf / (sum_val[0] + 1e-6);
    return psf;
}

cv::Mat edgetaper_cpp(const cv::Mat& img, const cv::Mat& psf) {
    cv::Mat tapered;
    img.convertTo(tapered, CV_32F);
    cv::Mat blurred;
    cv::filter2D(tapered, blurred, -1, psf, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

    int h = img.rows;
    int w = img.cols;
    cv::Mat alpha = cv::Mat::ones(h, w, CV_32F);
    int border_h = psf.rows; 
    int border_w = psf.cols; 

    for(int i=0; i<h; i++) {
        for(int j=0; j<border_w; j++) {
            float val = (float)j / border_w;
            alpha.at<float>(i, j) *= val;                
            alpha.at<float>(i, w - 1 - j) *= val;        
        }
    }
    for(int i=0; i<border_h; i++) {
        for(int j=0; j<w; j++) {
            float val = (float)i / border_h;
            alpha.at<float>(i, j) *= val;                
            alpha.at<float>(h - 1 - i, j) *= val;        
        }
    }
    cv::Mat result = alpha.mul(tapered) + (cv::Scalar(1.0) - alpha).mul(blurred);
    return result;
}

void fft_shift(cv::Mat& img) {
    int cx = img.cols / 2;
    int cy = img.rows / 2;
    cv::Mat q0(img, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(img, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(img, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(img, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

cv::Mat wiener_deblur_cpp(const cv::Mat& img, const cv::Mat& psf, double snr_k = 0.02) {
    int H = img.rows;
    int W = img.cols;
    cv::Mat taped_img = edgetaper_cpp(img, psf);
    int m = cv::getOptimalDFTSize(H);
    int n = cv::getOptimalDFTSize(W);
    cv::Mat padded_img;
    cv::copyMakeBorder(img, padded_img, 0, m - H, 0, n - W, cv::BORDER_REFLECT);
    
    cv::Mat psf_full = cv::Mat::zeros(m, n, CV_32F);
    int psf_h = psf.rows;
    int psf_w = psf.cols;
    int cx = psf_w / 2;
    int cy = psf_h / 2;
    psf.copyTo(psf_full(cv::Rect(n/2 - cx, m/2 - cy, psf_w, psf_h)));
    fft_shift(psf_full); 
    
    cv::Mat img_planes[] = {cv::Mat_<float>(padded_img), cv::Mat::zeros(padded_img.size(), CV_32F)};
    cv::Mat img_complex;
    cv::merge(img_planes, 2, img_complex);
    cv::dft(img_complex, img_complex);
    
    cv::Mat psf_planes[] = {psf_full, cv::Mat::zeros(psf_full.size(), CV_32F)};
    cv::Mat psf_complex;
    cv::merge(psf_planes, 2, psf_complex);
    cv::dft(psf_complex, psf_complex);
    
    std::vector<cv::Mat> split_h;
    cv::split(psf_complex, split_h);
    cv::Mat psf_mag_sqr;
    cv::magnitude(split_h[0], split_h[1], psf_mag_sqr);
    cv::pow(psf_mag_sqr, 2, psf_mag_sqr); 
    
    cv::Mat denom = psf_mag_sqr + snr_k;
    cv::Mat w_real, w_imag;
    cv::divide(split_h[0], denom, w_real);
    cv::divide(split_h[1], denom, w_imag);
    cv::multiply(w_imag, -1.0, w_imag); 
    
    cv::Mat w_planes[] = {w_real, w_imag};
    cv::Mat w_complex;
    cv::merge(w_planes, 2, w_complex);
    
    cv::Mat res_complex;
    cv::mulSpectrums(img_complex, w_complex, res_complex, 0);
    cv::idft(res_complex, res_complex, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    cv::Mat result;
    res_complex(cv::Rect(0, 0, W, H)).copyTo(result);
    cv::Mat final_u8;
    result.convertTo(final_u8, CV_8U);
    return final_u8;
}

bool b_continue_session;

void exit_loop_handler(int s){
    cout << "Finishing session" << endl;
    b_continue_session = false;
}

static rs2_option get_sensor_option(const rs2::sensor& sensor)
{
    if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE))
        return RS2_OPTION_ENABLE_AUTO_EXPOSURE;
    return RS2_OPTION_FRAMES_QUEUE_SIZE;
}

// ===================================================================
// MAIN FUNCTION
// ===================================================================
int main(int argc, char **argv) {

    // -------------------------------------------------------
    // 1. SETUP DIP MODE & INIT LOGGER (CORRECT LOCATION)
    // -------------------------------------------------------
    bool using_DIP = true;  // <--- Toggle this for DIP vs Baseline
    PerfLogger::Init(using_DIP); // <--- MUST BE CALLED HERE
    // -------------------------------------------------------

    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./stereo_realsense_D435i path_to_vocabulary path_to_settings (trajectory_file_name)"
             << endl;
        return 1;
    }

    string file_name;
    if (argc == 4) {
        file_name = string(argv[argc - 1]);
    }

    // Signal Handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
    
    b_continue_session = true;

    // RealSense Setup
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        return 0;
    }
    rs2::device selected_device = devices[0];
    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    int index = 0;
    
    for (rs2::sensor sensor : sensors)
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
            ++index;
            if (index == 1) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
                sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_LIMIT,5000);
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0); 
            }
            if (index == 2){
                sensor.set_option(RS2_OPTION_EXPOSURE,100.f);
            }
        }

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    cv::Mat imCV, imRightCV;
    int width_img, height_img;
    double timestamp_image = -1.0;
    bool image_ready = false;
    int count_im_buffer = 0;

    double global_yaw_deg = 0.0;      
    rs2_vector current_gyro = {0,0,0}; 
    double last_gyro_ts = 0.0;

    auto imu_callback = [&](const rs2::frame& frame)
    {
        std::unique_lock<std::mutex> lock(imu_mutex);

        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            count_im_buffer++;
            double new_timestamp_image = fs.get_timestamp()*1e-3;
            if(abs(timestamp_image-new_timestamp_image)<0.001){
                count_im_buffer--;
                return;
            }

            rs2::video_frame ir_frameL = fs.get_infrared_frame(1);
            rs2::video_frame ir_frameR = fs.get_infrared_frame(2);

            width_img = ir_frameL.get_width();
            height_img = ir_frameL.get_height();

            imCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameL.get_data()), cv::Mat::AUTO_STEP);
            imRightCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameR.get_data()), cv::Mat::AUTO_STEP);

            timestamp_image = fs.get_timestamp()*1e-3;
            image_ready = true;

            lock.unlock();
            cond_image_rec.notify_all();
        }
        else if (rs2::motion_frame m_frame = frame.as<rs2::motion_frame>())
        {
            if (m_frame.get_profile().stream_name() == "Gyro")
            {
                rs2_vector gyro = m_frame.get_motion_data();
                current_gyro = gyro; 
                double ts = m_frame.get_timestamp(); 

                if (last_gyro_ts == 0.0) {
                    last_gyro_ts = ts;
                } else {
                    double dt = (ts - last_gyro_ts) / 1000.0; 
                    double deg_per_sec = gyro.z * (180.0 / 3.14159265359);
                    global_yaw_deg += deg_per_sec * dt;
                    last_gyro_ts = ts;
                }
            }
        }
    };

    rs2::pipeline_profile pipe_profile = pipe.start(cfg, imu_callback);

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::STEREO, true, 0, file_name);
    float imageScale = SLAM.GetImageScale();

    double timestamp;
    cv::Mat im, imRight;

    double t_resize = 0.f;
    double t_track = 0.f;

    // ------------------------------------------------------------
    // FIX: Check b_continue_session to respect Ctrl+C
    // ------------------------------------------------------------
    while (!SLAM.isShutDown() && b_continue_session)
    {
        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
                cond_image_rec.wait(lk);

            if(count_im_buffer>1)
                cout << count_im_buffer -1 << " dropped frs\n";
            count_im_buffer = 0;

            timestamp = timestamp_image;
            im = imCV.clone();
            imRight = imRightCV.clone();
            image_ready = false;
        }

        // --- DIP PIPELINE ---
        if (using_DIP) 
        {
            cv::Mat raw_im = im.clone();
            cv::Mat raw_imRight = imRight.clone();
            cv::Mat im_clahe = raw_im.clone();
            cv::Mat im_clahe_right = raw_imRight.clone();
            
            ClaheParams params = choose_clahe_params_cpp(im); 
            if (params.do_clahe) {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->setClipLimit(params.clip_limit);
                clahe->setTilesGridSize(params.kernel_size);
                clahe->apply(raw_im, im_clahe);
                clahe->apply(raw_imRight, im_clahe_right);
            }

            int r_guide = 16; 
            double eps_guide = 50.0 / (255.0 * 255.0); 

            cv::Mat im_guided = guided_filter_cpp(raw_im, im_clahe, r_guide, eps_guide);
            cv::Mat im_guided_right = guided_filter_cpp(raw_imRight, im_clahe_right, r_guide, eps_guide);

            double blur_thresh = 60.0; 
            double blur_score = calc_blur_score(im_guided);
            bool is_blurry = (blur_score < blur_thresh);

            cv::Mat im_final = im_guided;
            cv::Mat im_final_right = im_guided_right;

            if (is_blurry) 
            {
                float vel_x = current_gyro.y; 
                float vel_y = current_gyro.x; 
                double angle_rad = atan2(vel_y, vel_x);
                double angle_deg = angle_rad * (180.0 / 3.14159265359);
                double magnitude = sqrt(vel_x*vel_x + vel_y*vel_y);
                double k = 7.0; 
                int psf_len = (int)(magnitude * k);
                if (psf_len < 3) psf_len = 3;
                if (psf_len > 15) psf_len = 15; 
                if (psf_len % 2 == 0) psf_len++; 

                cv::Mat psf = get_motion_psf(psf_len, angle_deg); 
                // Uncomment if you enabled wiener_deblur_cpp above
                // im_final = wiener_deblur_cpp(im_guided, psf, 0.05);
                // im_final_right = wiener_deblur_cpp(im_guided_right, psf, 0.05);
            }
            im = im_final;
            imRight = im_final_right;
        }

        if(imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
            cv::resize(imRight, imRight, cv::Size(width, height));
        }

        SLAM.TrackStereo(im, imRight, timestamp);
    }

    cout << "System shutdown requested..." << endl;
    
    // --------------------------------------------------------------------
    // CRITICAL FIX: SHUTDOWN -> SAVE TRAJECTORY -> CLOSE LOGGER
    // --------------------------------------------------------------------
    
    // 1. Stop SLAM threads
    SLAM.Shutdown(); 

    // 2. Save Trajectory (Must happen BEFORE closing logger/running python script)
    if(using_DIP) {
        cout << "Saving DIP trajectory to dip/trajectory_dip.txt..." << endl;
        SLAM.SaveTrajectoryTUM("dip/trajectory_dip.txt");
    } else {
        cout << "Saving Baseline trajectory to baseline/trajectory_baseline.txt..." << endl;
        SLAM.SaveTrajectoryTUM("baseline/trajectory_baseline.txt");
    }
    
    // 3. Close Logger & Run RPE Script (Now safe because file exists)
    cout << "Running RPE script..." << endl;
    PerfLogger::Instance().Close();

    return 0;
}