#!/usr/bin/env python3

import pygame


pygame.init()

pygame.joystick.init()


joy = pygame.joystick.Joystick(0)

joy.init()

while True:
    pygame.event.pump()
    print(joy.get_axis(5))
