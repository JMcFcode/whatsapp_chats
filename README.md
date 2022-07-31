# WhatsApp Chat Reader
This is a class designed to track statistics of different chats on WhatsApp.

## Installation:
Following imports required:
```python
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
import seaborn as sns
from collections import Counter
```
All are standard packages installed with anaconda.

## Usage:
Run the `ChatData.py` file to generate data summary printout and summary plots.

Configuration file `chat_config.py` contains settings such as filename etc.

Specify full path or navigate to working directory where the file is located.
WhatsApp file must be downloaded without images/docs in a .txt format.

### Currently Supports:
- Reading in data from whatsapp .txt file (two person chats)
- Looking at histogram of messages sent via hour and via month.
- Looking at reply statistics, by sender. Prints out averages and standard deviations
- Looking at Time Series of reply time and number of words sent over a rolling average period.

### Future plans:
Add support for group-chats.

Add Summary of chat the includes useful statistics.

Add support for other messaging services.
