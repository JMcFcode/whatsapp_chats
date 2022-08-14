# WhatsApp Chat Analysis
This is a class designed to track statistics of different chats on WhatsApp.

## Installation:
Following imports required:
```python
import pandas as pd
import numpy as np
import re
import time
import os
import string

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
from copy import copy
from functools import reduce
```
All are standard packages installed with anaconda.

## Usage:
Run the `ChatData.py` file to generate data summary printout and summary plots.

Configuration file `chat_config.py` contains settings such as filename etc.

Specify relative path to file.
WhatsApp file must be downloaded without images/docs in a .txt format.

### Currently Supports:
- Reading in data from whatsapp .txt file (two person chats and group chats)
- Looking at histogram of messages sent via hour and via month. Stacked to show what each participant did.
- Looking at reply statistics, by sender. Prints out averages and standard deviations
- View Pie chart that shows number of messages sent.
- Looking at Time Series of reply time and number of words sent over a rolling average period.
- Summary of chat data printed to console.

### Future plans:
Add NLP Sentiment and Emoji Analysis.

Add wordcloud

Add support for other messaging services.
