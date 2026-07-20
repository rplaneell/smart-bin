# smart-bin
The smart bin is an AI powered appliance that eliminates human error from household recycling. Using computer vision, it instantly classifies waste and mechanically sorts it into the correct chamber via simple gravity based routing. Built by data experts, it provides an affordable, robust, and fully automated solution.

# smart_bin: Autonomous Domestic Recycling

## Vision
An AI powered autonomous recycling system designed specifically for domestic environments. We are bringing the precision of industrial waste sorting directly into the kitchen.

## The Problem Statement
Household recycling is plagued by human error, confusion over local recycling guidelines, and cross contamination. When contaminated items enter the recycling stream, entire batches of otherwise recyclable materials are sent to landfills. Current domestic solutions are passive receptacles that rely entirely on user knowledge.

## Our Solution
The smart_bin removes the human decision making process from waste disposal. By utilizing real time computer vision and a modular mechanical sorting system, the appliance automatically identifies, categorizes, and physically sorts waste into the correct internal compartments.

## Current Milestone
Phase 1: Proof of Concept. 
* Establishing the foundational computer vision model using a static camera.
* Building the initial dataset of common domestic waste items.
* Designing the basic logic flow for the automated decision tree.

## Hardware Architecture
* Compute: Raspberry Pi 5 (acting as the main brain and edge inference device)
* Vision: 1080p wide angle camera module
* Actuation: Standard servo motors for bin lid routing
* Sensors: Ultrasonic proximity sensors for bin capacity monitoring

## Software Stack
* Computer Vision: PyTorch, OpenCV
* Backend Services: Python, FastAPI
* Embedded Systems: MicroPython for microcontroller peripherals
* Deployment: Docker

## Repository Map
* docs: Comprehensive project documentation, specs, and meeting logs.
* software: The core intelligence.
  * vision: Model training and inference scripts.
  * backend: API and state management.
  * firmware: Code for microcontrollers.
* hardware: Physical design files.
  * electronics: Wiring diagrams and PCB designs.
  * cad: 3D models for physical housing.
  * bom: Bill of materials.
* datasets: Images and labels for the vision model.
* experiments: Sandbox for testing new algorithms.

## Getting Started
1. Clone this repository to your local machine.
2. Navigate to the software directory.
3. Install the required Python dependencies using pip install requirements.txt.
4. Run the initial diagnostics script to verify hardware connections.

## Roadmap
Please refer to the docs/02_Roadmap.md file for a detailed timeline of our upcoming development cycles, hardware iteration plans, and dataset expansion goals.
