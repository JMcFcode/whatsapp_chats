"""
Config File
"""
filename: str = 'Chat_Files//ines_chat.txt'

month_map: dict = {1: 'Jan',
                   2: 'Feb',
                   3: 'Mar',
                   4: 'Apr',
                   5: 'May',
                   6: 'Jun',
                   7: 'Jul',
                   8: 'Aug',
                   9: 'Sep',
                   10: 'Oct',
                   11: 'Nov',
                   12: 'Dec'}
day_map: dict = {day: day for day in
                 ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

mov_avg_period: int = 30    # Period of moving average in days.
mov_avg_freq: int = 5
reply_time_max: float = 1   # Time at which replies are considered the start of a new conversation in days
