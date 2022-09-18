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
### Specification File
Example:
```json
{
    "CWP":"Condensing Water Pump",
    "CWVLV":"Condensing Water Valve",
    ...
}
```
