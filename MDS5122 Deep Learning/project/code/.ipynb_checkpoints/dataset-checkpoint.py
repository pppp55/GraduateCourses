"""
1. Extract 300 videos of the three classes (
    'Pushing [something] from right to left'
    'Dropping [something] onto [something]'
    'Covering [something] with [something]')
    respectively according to the train.json file.
    ["id" is the video name, "label" is the description text, "template" is the class label]

2. For each video, extract the first 21 frames and resize them as 96x96 pictures, the first 20 frames
   will be used as input and the last frame as the target output.
   That is, datasets:
        features: 20 images (96x96 or 128x128)
        instructions: 1 text description (corresponding to the "label" field in train.json)
        labels: 1 image (96x96 or 128x128)

tips:
- train.json file contains the list of training videos and their corresponding labels:
    [
        {
            "id": "12345",
            "label": "pushing iphone adapter from left to right",
            "template": "Pushing [something] from right to left"
        },
        {
            "id": "54321",
            "label": "dropping book onto table",
            "template": "Dropping [something] onto [something]"
        },
        ...
    ]

- videos are stored in the "videos" folder, each video is named as "{id}.webm":
    - videos/
        - 12345.webm
        - 54321.webm
        - ...
"""
