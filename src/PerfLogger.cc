#include "PerfLogger.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib> // Required for std::system

using namespace std;

unique_ptr<PerfLogger> PerfLogger::instance = nullptr;
once_flag PerfLogger::initFlag;

static bool ensure_dir_exists(const string &path) {
    try {
        if (path.empty()) return false;
        std::filesystem::path p(path);
        if (!std::filesystem::exists(p)) {
            std::filesystem::create_directories(p);
        }
        return true;
    } catch (const std::exception &e) {
        cerr << "PerfLogger: filesystem error: " << e.what() << endl;
        return false;
    }
}

void PerfLogger::Init(bool using_DIP) {
    call_once(initFlag, [using_DIP]() {
        instance.reset(new PerfLogger());
        instance->folder = using_DIP ? "dip" : "baseline";
        instance->OpenFiles();
    });
}

PerfLogger& PerfLogger::Instance() {
    if (!instance) {
        Init(false);
    }
    return *instance;
}

PerfLogger::~PerfLogger() {
    Close();
}

void PerfLogger::OpenFiles() {
    lock_guard<mutex> lk(mtx);
    if (initialized) return;

    ensure_dir_exists(folder);

    string path;

    path = folder + "/tracking_state_" + folder + ".csv";
    f_tracking_state.open(path, ios::out | ios::trunc);
    if (f_tracking_state.is_open()) f_tracking_state << "frame_id,state\n";
    else cerr << "PerfLogger: failed open " << path << endl;

    path = folder + "/keypoints_" + folder + ".csv";
    f_keypoints.open(path, ios::out | ios::trunc);
    if (f_keypoints.is_open()) f_keypoints << "frame_id,keypoints,tracked\n";
    else cerr << "PerfLogger: failed open " << path << endl;

    // Inliers removed as requested

    path = folder + "/reprojection_error_" + folder + ".csv";
    f_reprojection.open(path, ios::out | ios::trunc);
    if (f_reprojection.is_open()) f_reprojection << "frame_id,reproj_mean,reproj_std\n";
    else cerr << "PerfLogger: failed open " << path << endl;

    // --- ADDED FOR KEYFRAME FREQUENCY ---
    path = folder + "/keyframe_frequency_" + folder + ".csv";
    f_keyframe_freq.open(path, ios::out | ios::trunc);
    if (f_keyframe_freq.is_open()) f_keyframe_freq << "kf_id,frame_id,timestamp\n";
    else cerr << "PerfLogger: failed open " << path << endl;
    // ------------------------------------

    path = folder + "/map_culling_" + folder + ".csv";
    f_map_culling.open(path, ios::out | ios::trunc);
    if (f_map_culling.is_open()) f_map_culling << "kf_id,created,culled,cull_ratio,mean_obs\n";
    else cerr << "PerfLogger: failed open " << path << endl;

    initialized = true;
}

void PerfLogger::Close() {
    lock_guard<mutex> lk(mtx);
    if (!initialized) return;

    // 1. Close all open file streams
    if (f_tracking_state.is_open()) f_tracking_state.close();
    if (f_keypoints.is_open()) f_keypoints.close();
    if (f_reprojection.is_open()) f_reprojection.close();
    if (f_map_culling.is_open()) f_map_culling.close();
    // --- ADDED FOR KEYFRAME FREQUENCY ---
    if (f_keyframe_freq.is_open()) f_keyframe_freq.close();
    // ------------------------------------
    
    initialized = false;

    // // 2. AUTOMATICALLY RUN RPE SCRIPT
    // // Assumes you run the executable from the ORB_SLAM3 root directory
    // // Command: python3 scripts/compute_rpe.py {folder}/trajectory_{folder}.txt {folder}/rpe_{folder}.csv
    
    // string script_path = "scripts/compute_rpe.py";
    // string traj_file = folder + "/trajectory_" + folder + ".txt";
    // string rpe_output = folder + "/rpe_" + folder + ".csv";

    // string command = "python3 " + script_path + " " + traj_file + " " + rpe_output;

    // cout << endl << "[PerfLogger] Computing RPE..." << endl;
    // cout << "[PerfLogger] Command: " << command << endl;
    
    // int result = std::system(command.c_str());

    // if (result == 0) {
    //     cout << "[PerfLogger] RPE calculated successfully! File saved to: " << rpe_output << endl;
    // } else {
    //     cerr << "[PerfLogger] Error: Failed to run RPE script. Exit code: " << result << endl;
    //     cerr << "[PerfLogger] Make sure you are running from the ORB_SLAM3 root directory and python3/numpy are installed." << endl;
    // }
}

void PerfLogger::LogTrackingState(long frame_id, int state) {
    lock_guard<mutex> lk(mtx);
    if (f_tracking_state.is_open()) {
        f_tracking_state << frame_id << "," << state << "\n";
        if (verbose) cerr << "[PerfLogger] tracking_state " << frame_id << "," << state << endl;
    }
}

void PerfLogger::LogKeypoints(long frame_id, int keypoints, int tracked) {
    lock_guard<mutex> lk(mtx);
    if (f_keypoints.is_open()) {
        f_keypoints << frame_id << "," << keypoints << "," << tracked << "\n";
        if (verbose) cerr << "[PerfLogger] keypoints " << frame_id << "," << keypoints << "," << tracked << endl;
    }
}

void PerfLogger::LogReprojectionError(long frame_id, double mean, double stddev) {
    lock_guard<mutex> lk(mtx);
    if (f_reprojection.is_open()) {
        f_reprojection << frame_id << "," << fixed << setprecision(6) << mean << "," << fixed << setprecision(6) << stddev << "\n";
        if (verbose) cerr << "[PerfLogger] reproj " << frame_id << "," << mean << "," << stddev << endl;
    }
}

void PerfLogger::LogCreationTotal(int kf_id, int nCreated) {
    lock_guard<mutex> lk(mtx);
    if (f_map_culling.is_open()) {
        f_map_culling << kf_id << ",CREATED," << nCreated << "\n";
    }
}

void PerfLogger::LogCullingTotal(int kf_id, int nCulled) {
    lock_guard<mutex> lk(mtx);
    if (f_map_culling.is_open()) {
        f_map_culling << kf_id << ",CULLED," << nCulled << "\n";
    }
}
// Method overload for the existing LogMapCulling usage in LocalMapping.cc if you are still using it
void PerfLogger::LogMapCulling(int kf_id, int created, int culled, double cull_ratio, double mean_obs) {
    lock_guard<mutex> lk(mtx);
    if (f_map_culling.is_open()) {
        f_map_culling << kf_id << "," << created << "," << culled << "," << fixed << setprecision(6) << cull_ratio << "," << fixed << setprecision(6) << mean_obs << "\n";
        if (verbose) cerr << "[PerfLogger] map_culling " << kf_id << "," << created << "," << culled << "," << cull_ratio << "," << mean_obs << endl;
    }
}

// --- ADDED FOR KEYFRAME FREQUENCY ---
void PerfLogger::LogKeyFrame(long kf_id, long frame_id, double timestamp) {
    lock_guard<mutex> lk(mtx);
    if (f_keyframe_freq.is_open()) {
        f_keyframe_freq << kf_id << "," << frame_id << "," << fixed << setprecision(6) << timestamp << "\n";
        // Optionally flush for immediate writing in case of crash, though less efficient
        // f_keyframe_freq.flush(); 
        if (verbose) cerr << "[PerfLogger] keyframe_freq " << kf_id << "," << frame_id << "," << timestamp << endl;
    }
}
// ------------------------------------