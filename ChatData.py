#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:02:44 2022

@author: joelmcfarlane
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re
import seaborn as sns
from collections import Counter


sns.set()


class ChatData:
    
    def __init__(self, filename: str):
        self.filename = filename
    
    @staticmethod
    def parse_data(filename: str):
        with open(filename) as f:
            lines = f.readlines()
        new_lines = []
        lines.reverse()
        for i in range(len(lines)):
            line = lines[i]
            dt = line[0: 23].strip()
            if len(re.sub('\D', "", dt)) == 14 and dt.count('/') == 2 and dt.count(':') == 2:
                new_lines.append(line)
            elif i != len(lines) - 1:
                lines[i+1] = lines[i+1] + line
              
        lines.reverse()
        new_lines.reverse()
        return new_lines
    
    @staticmethod
    def into_df(txt: list) -> pd.DataFrame():
        """
        Read in Chat data and parse into a useable format.
        """
        raw_txt = pd.DataFrame(txt)
        raw_txt['Date'] = [line[0: 23].strip() for line in raw_txt[0]]
        raw_txt['Date'] = raw_txt['Date'].str.replace(' ', '')
        dt_list = []
        for dt in raw_txt['Date']:
            dt = re.sub('\D', "", dt)
            dt = datetime.strptime(dt, '%d%m%Y%H%M%S')
            dt_list.append(dt)
        raw_txt['Date'] = dt_list
        
        raw_txt['Msg'] = [line[23:] for line in raw_txt[0]]
        
        raw_txt['Sender'] = [line[0: line.index(':')] for line in raw_txt['Msg']]
        raw_txt['Sender'] = [string.strip() for string in raw_txt['Sender']]
        
        raw_txt['Message'] = [line[line.index(':') + 2: ] for line in raw_txt['Msg']]
        
        data_df = raw_txt[['Date', 'Sender', 'Message']].reset_index(drop=True)
        return data_df
    
    @staticmethod
    def month_hist(df: pd.DataFrame, label: str = None):
        """
        Create a histogram of the times messages are sent.
        """
        month_list = df['Date'].dt.month.to_list()
        month_map = {1: 'Jan',
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
        month_str_list = [month_map[i] for i in month_list]
        count = Counter(month_str_list)
        plot_df = pd.DataFrame.from_dict(count, orient='index').reset_index()
        plot_df.columns = ['Month', 'Count']
        plot_df['m'] = pd.Categorical(plot_df['Month'], categories=month_map.values())
        plot_df = plot_df.sort_values('m')
        plt.bar(plot_df['Month'], plot_df['Count'], label=label)
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        plt.show()
    
    @staticmethod
    def time_hist(df: pd.DataFrame, label=None):
        """
        Create a histogram of the times messages are sent.
        """
        df['Time'] = df['Date'].dt.time
        occurance_list = df['Time']
        hour_list = [t.hour + t.minute / 60 for t in occurance_list]
        numbers = [x for x in range(0, 24)]
        labels = map(lambda x: str(x), numbers)
        plt.xticks(numbers, labels)
        plt.xlim(0, 24)
        plt.hist(hour_list, bins=24*4, label=label, alpha=0.5)
        
    def time_hist_person(self, df: pd.DataFrame):
        """
        Shows who sent messages when.
        """
        for name in pd.unique(df['Sender']):
            mini_df = df[df['Sender'] == name]
            self.time_hist(df=mini_df, label=name)
            
        plt.xlabel('Hour')
        plt.ylabel('Number of Messages')
        
        plt.legend(loc='best')
        
        plt.show()

    def run(self):
        txt = self.parse_data(filename = self.filename)
        data_df = self.into_df(txt=txt)
        
        self.data_df = data_df


if __name__ == '__main__':
    test_class = ChatData(filename='olha_chat.txt')
    test_class.run()
    output = test_class.data_df

    test_class.month_hist(df=output)
