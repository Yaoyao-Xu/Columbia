void world2GPS(double r_lat, double r_lon, double x_in_NED, double y_in_NED, double& t_lon, double& t_lat) {
    double y_rad = x_in_NED / C_EARTH;
    t_lat = (r_lat + RAD2DEG(y_rad));
    double x_rad = y_in_NED / C_EARTH / cos(DEG2RAD(t_lat));
    t_lon = r_lon + RAD2DEG(x_rad);
}
 
void trans_NED2GPS(std::map<std::string, cv::Point3d>& all_gps_data) {
    std::map<std::string, cv::Point3d>::iterator it;
    double init_lat = 25.44993764;  // 纬度
    double init_lon = 112.71335009; // 经度
    // std::ofstream out("../data/gps_ccctrans.txt", std::ios::app);
    for (it = all_gps_data.begin(); it != all_gps_data.end(); it++) {
        double lat, lon;
        // std::cout<<"x:"<<(it->second).x<<std::endl;
        // std::cout<<"y:"<<(it->second).y<<std::endl;
        world2GPS(init_lat, init_lon, (it->second).x, (it->second).y, lon, lat);
        it->second.x = lat;
        it->second.y = lon;
        // std::cout<<"lat:"<<lat<<std::endl;
        // std::cout<<"lon:"<<lon<<std::endl<<std::endl;
        // out<<it->first<<" "<<std::setprecision(12)<<lat<<" "<<lon<<" "<<(it->second).z<<std::endl;
    }
    // out.close();
}
 
 