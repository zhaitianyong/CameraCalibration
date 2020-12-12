#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core.hpp>
using namespace std;
using namespace Eigen;


bool findHomography(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H, bool isNormal=false);

bool findHomographyByRansac(std::vector<Eigen::Vector2d>& srcPoints, std::vector<Eigen::Vector2d>& dstPoints, Eigen::Matrix3d& H);


void getObjectPoints(const cv::Size& borderSize, const cv::Size2f& squareSize, std::vector<Eigen::Vector3d>& objectPoints);

void computeCameraCalibration(std::vector<std::vector<Eigen::Vector2d>>& imagePoints,
        std::vector<std::vector<Eigen::Vector3d>>& objectPoints,
        cv::Mat& cameraMatrix, cv::Mat& distCoeffs);


void computeFisherCameraCalibration(std::vector<std::vector<Eigen::Vector2d>>& imagePoints,
                                    std::vector<std::vector<Eigen::Vector3d>>& objectPoints,
                                    cv::Mat& cameraMatrix, cv::Mat& distCoeffs);






#endif
