# Cloze
The Cloze is a building metadata generation system.
## Pipeline
1. The building metadata required by model training is stored in the directory `data`. The parsed words of metadata is stored in `data/{}_word_dict.json` and the ground truth is stored in `data/{}_label_dict.json`. 
   The specification file is stored in `data/{}_specification_file.json`.
   
2. Use the `cloze/speicication_file_process.py` to process the specification file.
   
3. Use the `cloze/load_data.py` to process and load the building metadata for training.
   
4. Use the `cloze/training.py` to train and evaluate the models in `model.py`.

5. Functional classes and methods can be found in `cloze/utiliz.py`.
## Installation
### Dependency
+ python 3
+ pip packages: `requirements.txt`
### Install

`git clone https://github.com/fangger4396/Cloze.git`

## Data Model
### Raw Metadata
Example:
```json
{
    "id": "101",
    "raw metadata": "wkgo-bf-cp01.temp1"
    "parsed raw metadata": [
        "wkgo",
        "bf",
        "cp",
        "01",
        "temp",
        "1",
    ],
}
```
### Labels in Brick
Example:
```json
{
    "id": "101",
    "parsed metadata": [
        "bf",
        "cp",
        "01",
        "temp",
        "1",
    ],
    "Brick class": "temperature sensor"
}
```
### Metadata Specification File
The metadata specification file consists of the pieces of metadata and the description of the metadata, that can be organized by key-value pairs.
The key is the pieces of metadata, and the value is the description of the metadata. 

Example of metadata specification file:
```json
{
  "Equipment Code": {
    "CP": "Chiller Plant",
    "CWP":"Condensing Water Pump",
    "TAF":"Transfer Air Fan"
  },
  "Location Code": {
    "BF": "Basement Floor",
    "1F": "First Floor",
    "PLANT": "Plant Room",
    "DINRM": "Dining Room"
  },
  "Point Code": {
    "TEMP": "Temperature",
    "ODT": "Outdoor Air Temperature",
    "OWT": "Outlet Water Temperature",
    "FSPD": "Fan Speed"
  }
}
```
Example for matching specification file with Brick schema tags:
```json
{
  "Equipment Code": {
    "CP": "Chiller",
    "CWP":"Water Pump",
    "TAF":"Air Fan"
  },
  "Location Code": {
    "BF": "Basement Floor",
    "1F": "Floor",
    "PLANT": "Room",
    "DINRM": "Room"
  },
  "Point Code": {
    "TEMP": "Temperature Sensor",
    "ODT": "Outside Air Temperature Sensor",
    "OWT": "Return Water Temperature Sensor",
    "FSPD": "Fan Speed Sensor"
  }
}
```

Example for generating labels for building metadata:
```json
{
  "id": "101",
    "raw metadata": "wkgo-bf-cp01.temp1",
    "parsed metadata": [
        "bf",
        "cp",
        "01",
        "temp",
        "1",
    ],
    "Brick class": "temperature sensor"
}
```
