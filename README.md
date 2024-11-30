# Photogrammetry-project


## 1.Bundle Block Adjustment
### 1.1 Export markers from Metashape
File/Export/Export Markers -> .xml
### 1.2 Set path to your xml file line 12 in bundle_block.py
It will automatically import markers and theirs projections using read_xml.py
### 1.3 Set GCP and tie points line 27-28 in bundle_block.py
At the moment, the program uses CP as tie point, I still need to find out how to export tie points projections from Metashape. Then it may not work if not enough CP.
### 1.4 Run python3 bundle_block.py


## 2. Get station frame
Frame-to-frame least squares optimization to find transformation between two station frames.
### 2.1 Inititialize 'stations' in get_station_frame.py
### 2.2 Check
It need at least 3 common measurements between both stations.
### 2.3 Run python3 ./get_station_frame.py --fro 1 --to 0 --show 1 --write 1
usage: get_station_frame.py [-h] [--fro FRO] [--to TO] [--show SHOW] [--write WRITE]

options:
  -h, --help     show this help message and exit
  --fro FRO      Index of the first station.
  --to TO        Index of the second station.
  --show SHOW    Plot ? (1 or 0)
  --write WRITE  Write points measured from station a expressed in station b. (1 or 0)


## 3. Get station frames
Multi-frame simultaneous optimization to get all frames transformations
### 3.1 Inititialize 'stations' in get_station_frame.py
### 3.2 Run python3 ./get_station_frames.py --write 1
usage: get_station_frames.py [-h] [--write WRITE]

Calculate the transformation between every stations. Change lines 281 and 212 to get the right
transformation order.

options:
  -h, --help     show this help message and exit
  --write WRITE  Write points measured for each station represented in each frame. (1 or 0)
### 3.3 Check!!
It may produce some partially wrong results in cases there are too few common measurements between stations. If it happens, use get_station_frame.py twice with an intermediate frame. For example : 1->2 and 2->0, then store the transformation and use transformation.py to compose transformations, it should correct the result.

