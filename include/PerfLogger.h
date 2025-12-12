#pragma once
#ifndef PERFLOGGER_H
#define PERFLOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <memory>

class PerfLogger {
public:
    static void Init(bool using_DIP);
    static PerfLogger& Instance();
    ~PerfLogger(); // Public destructor

    void Close();

    void LogTrackingState(long frame_id, int state);
    void LogKeypoints(long frame_id, int keypoints, int tracked);
    void LogReprojectionError(long frame_id, double mean, double stddev);

    // Creation/Culling helpers
    void LogCreationTotal(int kf_id, int nCreated);
    void LogCullingTotal(int kf_id, int nCulled);
    
    // Keep this if LocalMapping.cc still calls it with 5 arguments
    void LogMapCulling(int kf_id, int created, int culled, double cull_ratio, double mean_obs);

    void SetVerbose(bool v) { verbose = v; }
    // --- ADDED FOR KEYFRAME FREQUENCY ---
    void LogKeyFrame(long kf_id, long frame_id, double timestamp);
    // ------------------------------------

private:
    PerfLogger() = default;
    void OpenFiles();

    static std::unique_ptr<PerfLogger> instance;
    static std::once_flag initFlag;

    std::string folder;
    bool initialized{false};
    bool verbose{false};

    std::ofstream f_tracking_state;
    std::ofstream f_keypoints;
    std::ofstream f_reprojection;
    std::ofstream f_map_culling;

    // --- ADDED FOR KEYFRAME FREQUENCY ---
    std::ofstream f_keyframe_freq; // New file stream
    // ------------------------------------

    std::mutex mtx;
};

#endif // PERFLOGGER_H