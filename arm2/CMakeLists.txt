# cmake_minimum_required(VERSION 2.8.3)

# project(arm2)

# find_package(catkin REQUIRED)

# catkin_package()

# find_package(roslaunch)

# foreach(dir config launch meshes urdf)
# 	install(DIRECTORY ${dir}/
# 		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
# endforeach(dir)

cmake_minimum_required(VERSION 2.8.3)
project(arm2)

# 必须包含roscpp
find_package(catkin REQUIRED COMPONENTS 
  roscpp
  gazebo_ros
)

catkin_package()

# 显式列出目录
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)
