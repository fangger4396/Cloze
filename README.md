# Cloze

## Pinpeline

## Installation

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
