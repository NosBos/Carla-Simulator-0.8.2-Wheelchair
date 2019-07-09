# Carla-Simulator-0.8.2-Wheelchair

So this is a copy of Carla Simulator 0.8.2 PythonClient folder. 

This is part of a project at Blue Horizon AI to make an autonomous wheelchair. This simulator environment is being created
to set up an environment for collecting data, training and testing autonomous wheelchairs.


On a fresh download Carla has 2 scripts that allow
the simulator to run in server mode.



 1. manual_example.py - This script allows the simulator to be controlled through manual controls
 
 2. client_example.py - This script allows the images from the simulator to be saved
 
 
 I wanted to have a script that allowed me to control the simulator manually and save the data as I drove.
 
 
 
 The main script that is being made is Manual-Control-Data-Collection.py
 
 This script is based off the manual_example.py
 
 Current Features
 1. Manual control through Xbox controller
 2. Saving image data as a .jpg file to a folder
 3. Store the steering angle, throttle and other values such as timestamp to a csv file 
 3. Record data and replay that data to ensure that the recorded data is accurate for imitation learning
 4. Allow the simulator to be controlled in realtime through an autonomous model (using .h5 file and .json file)
 
 Future Features
 1. Have the car model be replaced with a wheelchair
  - This will be done on Unreal Engine 4 editor so this github wont be updated with that feature
 
 
 
HOW TO USE
1. Install Carla simulator 0.8.2
2. Copy the Manual-Control-Data-Collection.py file into your PythonClient folder
3. Mulitple changes may need to be done to ensure that it works on your computer
 - filepaths, controller setup and a couple modules
4. The version which allows models to be used requires a miniconda environment (If you want it message me on github)
5. Run this file as you would run client_example.py following the documentation on Carla wiki for stable branch (8.2)
6. Some documentation is added which is missing in the manual_control.py - The comments I have added may help you
change manual_control.py to what you need it to do
