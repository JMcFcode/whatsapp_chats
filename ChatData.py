#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:02:44 2022

@author: joelmcfarlane
"""
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

import chat_config

sns.set()


class ChatData:

    def __init__(self, filename: str):
        self.filename = filename

    @staticmethod
    def parse_data(filename: str):
        with open(os.getcwd() + '//' + filename, 'r', encoding='utf8') as f:
            lines = f.readlines()
        new_lines = []
        lines.reverse()

        for i in range(len(lines)):
            line = lines[i]
            line = line.replace('\u200e', '')
            line = line.replace('\u202a', '')
            line = line.replace('\u202c', '')
            line = line.replace('\xa0', '')
            line = line.replace('\n', '')
            dt = line[0: 23].strip()
            if len(re.sub('\D', "", dt)) == 14 and dt.count('/') == 2 and dt.count(':') == 2:
                new_lines.append(line)
            elif i != len(lines) - 1:
                lines[i + 1] = lines[i + 1] + line

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
        raw_txt['Sender'] = [line[0: line.index(':')] if ':' in line else 'SYSTEM'
                             for line in raw_txt['Msg']]
        raw_txt['Sender'] = [string.strip() for string in raw_txt['Sender']]
        raw_txt['Message'] = [line[line.index(':') + 2:] if ':' in line else line for line in raw_txt['Msg']]

        data_df = raw_txt[['Date', 'Sender', 'Message']]
        data_df = data_df.iloc[1:, :].reset_index(drop=True)
        return data_df

    @staticmethod
    def categorical_sort(list_cat: list, mapping: dict, label: str) -> pd.DataFrame:
        """
        Create a dataframe that summaries the count of data from a df with static values
        :return: out_df
        """
        str_list = [mapping[i] for i in list_cat]
        count = Counter(str_list)
        out_df = pd.DataFrame.from_dict(count, orient='index').reset_index()
        out_df.columns = [label, 'Count']
        out_df['m'] = pd.Categorical(out_df[label], categories=mapping.values())
        out_df = out_df.sort_values('m')
        return out_df

    @staticmethod
    def time_hist(df: pd.DataFrame, ax, ax_val: int, label=None):
        """
        Create a histogram of the times messages are sent.
        """
        df['Time'] = df['Date'].dt.time
        occurance_list = df['Time']
        hour_list = [t.hour + t.minute / 60 for t in occurance_list]
        numbers = [x for x in range(0, 24)]
        labels = map(lambda x: str(x), numbers)
        ax[ax_val].set_xticks(numbers)
        ax[ax_val].set_xticklabels(labels)
        ax[ax_val].set_xlim(0, 24)
        ax[ax_val].hist(hour_list, bins=24 * 4, label=label, alpha=0.5, histtype='bar', fill=True)

    def plot_hist(self, df: pd.DataFrame):
        """
        Plot a set of histograms displaying various attributes.
        :param df:
        """
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        for name in self.names:
            mini_df = df[df['Sender'] == name]
            self.time_hist(df=mini_df, ax_val=0, ax=ax, label=name)
        ax[0].set_xlabel('Hour')
        ax[0].set_ylabel('Number of Messages')
        ax[0].legend(loc='best')

        month_df_list = []
        for name in self.names:
            df_person = df[df['Sender'] == name]
            list_month = df_person['Date'].dt.month.to_list()
            month_df = self.categorical_sort(list_cat=list_month,
                                             mapping=chat_config.month_map, label='Month')
            month_df.rename(columns={'Count': name}, inplace=True)
            month_df.drop(columns=['m'], inplace=True)
            month_df_list.append(month_df)
        all_months_df = reduce(lambda left, right: pd.merge(left, right, on=['Month'], how='outer'), month_df_list)
        all_months_df.plot.bar(x='Month', stacked=True, ax=ax[1])
        ax[1].set_xlabel('Month')
        ax[1].set_ylabel('Number of Messages')

        day_df_list = []
        for name in self.names:
            df_person = df[df['Sender'] == name]
            list_day = df_person['Date'].apply(lambda x: x.strftime('%A'))
            day_df = self.categorical_sort(list_cat=list_day,
                                           mapping=chat_config.day_map, label='Day')
            day_df.rename(columns={'Count': name}, inplace=True)
            day_df.drop(columns=['m'], inplace=True)
            day_df_list.append(day_df)
        all_days_df = reduce(lambda left, right: pd.merge(left, right, on=['Day'], how='outer'), day_df_list)
        all_days_df.plot.bar(x='Day', stacked=True, ax=ax[2])
        ax[2].set_xlabel('Day')
        ax[2].set_ylabel('Number of Messages')

        fig.tight_layout()
        plt.show()

    def reply_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the average reply time in minutes.
        Works for chats where there are two people.
        :param df:
        :return: dict.
        """
        t1 = time.perf_counter()
        res = df.apply(self.reply_single, axis=1, args=(df, ))
        t2 = time.perf_counter()
        print(f'Reply time calc took {t2 -t1} seconds')
        res_df = pd.DataFrame(res.to_list(), columns=['DateTime', 'Reply_Sender', 'Reply_Time'])
        res_df = self.filter_multiple_messages(res_df=res_df)
        print('\n')
        return res_df

    @staticmethod
    def filter_multiple_messages(res_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out times when a series of messages has been sent.
        """
        s1 = time.perf_counter()
        bool_list = []
        for i in res_df.index.values:
            if i + 1 == len(res_df):
                bool_list.append(True)
            elif res_df['Reply_Sender'].iloc[i + 1] == res_df['Reply_Sender'].iloc[i]:
                bool_list.append(False)
            else:
                bool_list.append(True)
        res_df = res_df[bool_list].reset_index(drop=True)
        s2 = time.perf_counter()
        print(f'Filter multiple messages took {s2 -s1} seconds')
        return res_df

    @staticmethod
    def reply_single(row: pd.Series, df: pd.DataFrame) -> tuple:
        time_sent = row['Date']
        sender = row['Sender']
        temp_df = df[df['Date'] > time_sent]
        temp_df = temp_df[temp_df['Sender'] != sender]
        if len(temp_df) != 0:
            reply_datetime = temp_df['Date'].iloc[0]
            reply_sender = temp_df['Sender'].iloc[0]
            delta_time = (reply_datetime - time_sent).total_seconds() / 60  # Value in minutes.
            return time_sent, reply_sender, delta_time

    @staticmethod
    def calc_data_amount(df: pd.DataFrame) -> int:
        """
        Calculate the number of words sent in a dataframe
        :param df:
        :return: int
        """
        big_str = ' '.join(df['Message'].to_list())
        word_count = len(big_str)
        return word_count

    def time_series_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how many words sent in a rolling average period.
        """
        res_dict = {}
        for date in self.date_list_mov_avg:
            temp_df = df[(df['Date'] > date) & (df['Date'] < date + timedelta(chat_config.mov_avg_period))]
            word_count = self.calc_data_amount(df=temp_df)
            res_dict[date] = word_count
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df = res_df.reset_index()
        res_df.columns = ['DateTime', 'Word_Count']
        return res_df

    def time_series_reply(self, reply_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the average reply time in a rolling average period.
        """
        res_dict = {}
        for date in self.date_list_mov_avg:
            temp_df = reply_df[(reply_df['DateTime'] > date) & (reply_df['DateTime']
                                                                < date + timedelta(chat_config.mov_avg_period))]
            avg_reply = np.mean(temp_df['Reply_Time'])
            res_dict[date] = avg_reply
        res_df = pd.DataFrame.from_dict(res_dict, orient='index')
        res_df = res_df.reset_index()
        res_df.columns = ['DateTime', 'Avg_Reply_Time']
        return res_df

    def plot_time_series(self):
        """
        Display a time series of:
        - Average reply time rolling average.
        - Average amount of content sent.
        """
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        for name in self.dict_data.keys():
            word_time_df = self.time_series_size(df=self.dict_data[name])
            ax[0].plot(word_time_df['DateTime'], word_time_df['Word_Count'], label=name)
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel(f'Word Count per {chat_config.mov_avg_period} days')
        ax[0].legend(loc='best')

        reply_df = self.reply_time(df=self.dict_data['All'])
        reply_df = reply_df[reply_df['Reply_Time'] < 60 * 24 * chat_config.reply_time_max]
        self.print_reply_stats(reply_df=reply_df)
        for name in self.dict_data.keys():
            if name != 'All':
                reply_temp_df = reply_df[reply_df['Reply_Sender'] == name]
            else:
                reply_temp_df = copy(reply_df)
            word_time_df = self.time_series_reply(reply_df=reply_temp_df)
            ax[1].plot(word_time_df['DateTime'], word_time_df['Avg_Reply_Time'], label=name)
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel(f'Average Reply Time (mins)')
        ax[1].legend(loc='best')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def print_reply_stats(reply_df: pd.DataFrame):
        for name in pd.unique(reply_df['Reply_Sender']):
            temp_df = reply_df[reply_df['Reply_Sender'] == name]
            mean = np.mean(temp_df['Reply_Time'])
            temp_std_dev = np.std(temp_df['Reply_Time'])
            print(f'{name} has average reply time of {round(float(mean), 1)} minutes and standard deviation of'
                  f' {round(float(temp_std_dev), 1)} minutes \n ')

    @staticmethod
    def summary_all(data_df: pd.DataFrame, print_logs: bool = True) -> dict:
        """
        Summary of all the whole chat.
        """
        list_active = [1 if date in data_df['Date'].dt.date.values else 0 for date in
                       pd.date_range(start=data_df['Date'].values[0], end=data_df['Date'].values[-1]).date]

        days_active = sum(list_active)
        days_inactive = len(data_df) - days_active

        string_data = data_df['Message'].str.cat(sep=' ')
        num_words = sum([i.strip(string.punctuation).isalpha() for i in string_data.split()])

        dict_out = {'Number of Messages': len(data_df),
                    'Word Count': num_words,
                    'Period': f"{pd.to_datetime(data_df['Date'].values[0]).date()} -> "
                              f"{pd.to_datetime(data_df['Date'].values[-1]).date()}",
                    'Days Active': days_active,
                    'Days Inactive': days_inactive}
        dict_out['Word_per_message'] = dict_out['Word Count'] / dict_out['Number of Messages']

        if print_logs:
            print('\n')
            for key, values in dict_out.items():
                print(f'{key}: {values}')
            print('\n')
        return dict_out

    def draw_pie(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.set_title('Number of Messages')
        list_messages = [self.dict_summary[name]['Number of Messages'] for name in self.names]
        ax.pie(list_messages, labels=self.names, startangle=90)

        plt.show()

    def run(self):
        txt = self.parse_data(filename=self.filename)
        data_df = self.into_df(txt=txt)
        if 'SYSTEM' in data_df['Sender'].values:
            self.full_group_df = data_df
            data_df = data_df[data_df['Sender'] != 'SYSTEM']
        else:
            self.full_group_df = None

        self.names = pd.unique(data_df['Sender'])
        start_date = data_df['Date'].iloc[0]
        self.date_list_mov_avg = pd.date_range(start_date, datetime.today(), freq=str(chat_config.mov_avg_freq) + 'D')

        self.dict_summary_all = self.summary_all(data_df=data_df)

        self.dict_data = {'All': data_df}
        self.data_df = data_df
        self.dict_summary = {name: self.summary_all(data_df=data_df[data_df['Sender'] == name], print_logs=False)
                             for name in self.names}

        for name in self.names:
            self.dict_data[name] = data_df[data_df['Sender'] == name]


if __name__ == '__main__':
    chat_class = ChatData(filename=chat_config.filename)
    chat_class.run()
    output = chat_class.data_df

    chat_class.plot_hist(df=output)
    chat_class.plot_time_series()
    chat_class.draw_pie()
