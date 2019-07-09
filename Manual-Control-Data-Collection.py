#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use Xbox Controller for control.

    RT on Xbox Controller                 : throttle
    LT on Xbox Controller                 : brake
    Right Joystick on Xbox Controller     : steer

    Q                                     : toggle reverse

    Space                                 : hand-brake

    P                                     : toggle autopilot

    M                                     : manual control
    B                                     : autonomous model control
    N                                     : replay control

    R                                     : restart level
    C                                     : change weather

    L                                     : toggle data collection


STARTING in a moment...
"""

from __future__ import print_function
from decimal import Decimal

import argparse
import logging
import random
import time

import csv
import pandas as pd

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_b
    from pygame.locals import K_n
    from pygame.locals import K_m

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#Controller Support
pygame.init()
pygame.joystick.init()
joy = pygame.joystick.Joystick(0)
joy.init()

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

#import prediction file
from real_time_prediction import RealTimePrediction

#Initializing prediction object
model_name = '/home/gill/Carla/CARLA_0.8.2/PythonClient/Model/model.json'
model_weights = '/home/gill/Carla/CARLA_0.8.2/PythonClient/Model/model_weights_15mins.h5'

#Initalizing the model object
p = RealTimePrediction(model_name, model_weights)


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180


#Create data.csv file / overwrites previously recorded data
with open('data.csv', 'w',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Frame', 'Timestamp', 'Steering Angle', 'Speed', 'Throttle']) 
f.close()


#opening replay csv file- uncomment if replay needed
#replay = pd.read_csv("replay.csv")

#Carla setting are set
def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()

    #Camera type and placement is chosen
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)

    #if lidar is chosen, placement of it
    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32,
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=10,
            LowerFovLimit=-30)
        settings.add_sensor(lidar)
    return settings

#class of different times values which are kept track of
class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
          
        #Wheelchair addition
        self._data_collection = False
        self._time_stamp = float(0)
        self._frame = 0
        self._replay_frame = 1
        self._input_control = "Manual"
        self._AI_steer = 0
        self._player_start = args.start

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)

        #allows user to decide starting position for replay
        player_start = self._player_start

        print("Your starting postion is ", player_start)

        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):

        #loop takes place every tick
        self._timer.tick()
        
        #extracts data from simulator
        measurements, sensor_data = self.client.read_data()

        #image for pygame is extracted
        self._main_image = sensor_data.get('CameraRGB', None)

        #control is given by calling _get_keyboard_control function
        control = self._get_keyboard_control(pygame.key.get_pressed())
        
        speed = Decimal(measurements.player_measurements.forward_speed * 3.6)

        #steer and throttle values extracted from control
        steer = Decimal(control.steer)
        throttle = Decimal(control.throttle)

        # Print measurements every chosen amount of time.
        if self._timer.elapsed_seconds_since_lap() > 0.033:
            #timestamp keeps track of how much time has elasped
            self._time_stamp = self._time_stamp + 0.033

            #If input_control is replay, keeps track of frames read
            if self._input_control == "Replay":

                self._replay_frame = self._replay_frame + 1

            #Save Image Data from every # seconds into folder "out"
            if self._data_collection:
                self._frame = self._frame + 1 
                
                for name, measurement in sensor_data.items():
                    filename = '_out/episode_{}'.format(self._frame)
              
                    measurement.save_to_disk(filename)    
                
                #row created for saving data to csv
                row = ["_out/episode_" + str(self._frame), self._time_stamp, round(steer,7), round(speed, 3), round(throttle, 3)]

                #csv file written to
                with open('data.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                csvFile.close()

            #get steering direction from AI
            if self._input_control == "AI":
                for name, measurement in sensor_data.items():
                    #call do_predict, gets steer values from real_time_prediction.py
                    self._AI_steer = p.do_predict(measurement.data)
                
            #lap time is reset to allow this if statment to be called on accurate intervals
            self._timer.lap()


        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """

        #returns NONE to close simulator
        if keys[K_r]:
            return None

        #control is set to its accurate type
        control = VehicleControl()

        #input set to manual
        if self._input_control == "Manual":
            #updates and checks for input from controller
            pygame.event.pump()

            #get_axis values may differ depending on controller
            #values weighted to be used on carla
            control.steer = joy.get_axis(3)

            control.throttle = (joy.get_axis(5) + 1) / 2

            control.brake = (joy.get_axis(2) + 1) / 2

        #replay of previously recorded data
        elif self._input_control == "Replay":

            #read replay.csv file, _replay_frame values gives us correct row
            control.steer = replay.iloc[self._replay_frame,2]
            control.throttle = replay.iloc[self._replay_frame,4]

        #input set to autonomous, steer values from model
        else:
            control.steer = self._AI_steer
            control.throttle = 0.5
            print(control.steer)
          

        if keys[K_l]:
            self._data_collection = not self._data_collection

            if self._data_collection:
                print("Data Collection ON")
            else:
                print("Data Collection OFF")
            time.sleep(0.05)

        #Check for keyboard commands to change input control device

        if keys[K_m]:
            print("Manual Control")
            self._input_control = "Manual"

        if keys[K_n]:
            print("AI Control")
            self._input_control = "AI"

        if keys[K_b]:
            print("Replay Control")
            self._input_control = "Replay"

        if keys[K_SPACE]:
            control.hand_brake = True

        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse

        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot

        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        
        if self._lidar_measurement is not None:
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            #draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface, (10, 10))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    argparser.add_argument(
        '-s', '--start',
        default=1,
        type=int,
        help='Choose the starting postion of the simulator 1-152 (default: 1)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
