---
layout: project
title: Robot motion planning 
featured-img: robot_motion_planning
project-url: https://github.com/SergiosKar/Robot-Motion-Planning
---

The goal of the project was to simulate a 2D robot (a car for example) and develop a motion planning system in a predefined environment with obstacles. 

The robot can have 2 or 3 degrees of freadom and can move hotizontally, vertically or rotate itself. Using computational geometry techniques such as voronoi diagramm and delauny triangulation, we were able to define the environment and detect the obstacles. Afterwards, we found the shortest route to the destination by constructing the viibility graph and perform a local search to move across the graph nodes.

The whole process was simulated and animated with an in-house library for computer graphics .