//
// Created by xiang on 22-12-20.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <vector>

#include "common/eigen_types.h"
#include "common/point_cloud_utils.h"
#include "common/lidar_utils.h"
#include "keyframe.h"
#include "ch7/ndt_3d.h"

DEFINE_string(map_path, "../data/ch9/", "导出数据的目录");
DEFINE_double(voxel_size, 0.1, "导出地图分辨率");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    using namespace sad;

    std::map<IdType, KFPtr> keyframes;
    if (!LoadKeyFrames("../data/ch9/keyframes.txt", keyframes)) {
        LOG(ERROR) << "failed to load keyframes";
        return 0;
    }

    std::map<Vec2i, CloudPtr, less_vec<2>> map_data;  // 以网格ID为索引的地图数据
    pcl::VoxelGrid<PointType> voxel_grid_filter;
    float resolution = FLAGS_voxel_size;
    voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

    // 逻辑和dump map差不多，但每个点个查找它的网格ID，没有的话会创建
    for (auto& kfp : keyframes) {
        auto kf = kfp.second;
        kf->LoadScan("../data/ch9/");

        CloudPtr cloud_trans(new PointCloudType);
        pcl::transformPointCloud(*kf->cloud_, *cloud_trans, kf->opti_pose_2_.matrix());

        // voxel size
        CloudPtr kf_cloud_voxeled(new PointCloudType);
        voxel_grid_filter.setInputCloud(cloud_trans);
        voxel_grid_filter.filter(*kf_cloud_voxeled);

        LOG(INFO) << "building kf " << kf->id_ << " in " << keyframes.size();

        // add to grid
        for (const auto& pt : kf_cloud_voxeled->points) {
            int gx = floor((pt.x - 50.0) / 100);
            int gy = floor((pt.y - 50.0) / 100);
            Vec2i key(gx, gy);
            auto iter = map_data.find(key);
            if (iter == map_data.end()) {
                // create point cloud
                CloudPtr cloud(new PointCloudType);
                cloud->points.emplace_back(pt);
                cloud->is_dense = false;
                cloud->height = 1;
                map_data.emplace(key, cloud);
            } else {
                iter->second->points.emplace_back(pt);
            }
        }
    }

    // 存储点云和索引文件
    LOG(INFO) << "saving maps, grids: " << map_data.size();
    std::system("mkdir -p ../data/ch9/map_data/");
    std::system("rm -rf ../data/ch9/map_data/*");

    std::system("mkdir -p ../data/ch9/map_data/res_10/");
    std::system("mkdir -p ../data/ch9/map_data/res_5/");
    std::system("mkdir -p ../data/ch9/map_data/res_4/");
    std::system("mkdir -p ../data/ch9/map_data/res_3/");
    std::system("mkdir -p ../data/ch9/map_data/res_1/");

    std::ofstream fout("../data/ch9/map_data/map_index.txt");
    
    for (auto& dp : map_data) {
        fout << dp.first[0] << " " << dp.first[1] << std::endl;
        dp.second->width = dp.second->size();
        VoxelGrid(dp.second, 0.1);

        pcl::io::savePCDFileBinaryCompressed(
            "../data/ch9/map_data/" + std::to_string(dp.first[0]) + "_" + std::to_string(dp.first[1]) + ".pcd",
            *dp.second);
    }
    fout.close();
    std::vector<double> resolution1{10.0, 5.0, 4.0, 3.0, 1.0};
    for(auto res : resolution1){
        for (auto& dp : map_data) {
            dp.second->width = dp.second->size();
            auto output = VoxelCloud(dp.second, res * 0.1);
            Ndt3d ndt;
            ndt.SetResolution(res);
            ndt.SetTarget(output);
            std::ofstream fout1("../data/ch9/map_data/res_"+ std::to_string((int)res) + "/" + std::to_string(dp.first[0]) +
                                "_" + std::to_string(dp.first[1]) + ".txt");
            for(const auto& grid : ndt.grids_){
                Eigen::Matrix<int, 3, 1> key(grid.first);
                fout1 << key[0] << " " << key[1] << " " << key[2] << " "
                    << grid.second.mu_[0] << " " << grid.second.mu_[1] << " " << grid.second.mu_[2]  
                    << " ";
                for(int row = 0; row < 3; row++){
                    for(int col = 0; col < 3; col++){
                        fout1 << grid.second.info_(row, col) << " ";
                    }
                }
                fout1 << std::endl;
            }
            fout1.close();
        }
    }

    LOG(INFO) << "done.";
    return 0;
}
