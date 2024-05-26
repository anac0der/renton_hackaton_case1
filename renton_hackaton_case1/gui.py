import tkinter as tk
from tkcalendar import Calendar
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import pickle
import datetime

plt.style.use('seaborn-v0_8')
class Anoms_GUI:
    def __init__(self):
        width, height = 1200, 950
        self.root = tk.Tk()
        self.root.title('Anomaly detection @ 2Pandas team')
        self.root.geometry(f'{width}x{height}')
        self.figure = plt.figure(figsize=(12, 9))
        self.axes = plt.axes()

        self.detector = pickle.load(open('./models/filename_detector.pickle', 'rb'))
        self.forecaster = pickle.load(open('./models/filename_forecaster.pickle', 'rb'))
        self.train_data = pickle.load(open('./saved_data/train_data.pickle', 'rb'))
        self.test_data = pickle.load(open('./saved_data/test_data.pickle', 'rb'))
        self.train_anom = pickle.load(open('./saved_data/train_anom.pickle', 'rb'))
        self.curr_x = self.train_data
        self.curr_features = None
        self.curr_target = self.train_data['CPU usage [%]']
        self.curr_anom = pd.Series(np.array([0 for _ in range(self.train_data.shape[0])]))
        self.is_train_plotted = False
        self.is_annotated = True
        self.curr_med = None
        self.window = None
        self.button_start = None
        self.button_end = None
        self.date_start = datetime.datetime(day=1, month=1, year=1970, hour=0, minute=0, second=0)
        self.date_end = datetime.datetime(day=14, month=1, year=1970, hour=0, minute=0, second=0)

        frame = tk.Frame(width=1200, height=900, borderwidth=1)
        frame.grid(row=12, column=4)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=10, columnspan=4)
        btn = tk.Button(frame, text="Load train data", command = self.plot_train_data)
        btn.grid(row=10, column=0, sticky=tk.NSEW)
        btn = tk.Button(frame, text="Annotate anomaly samples", command = self.annotate_anomalies)
        btn.grid(row=10, column=1, sticky=tk.NSEW)
        btn = tk.Button(frame, text="Choose forecasting interval", command = self.choice_date)
        btn.grid(row=10, column=2, sticky=tk.NSEW)
        btn = tk.Button(frame, text="Forecast on interval", command = self.forecast)
        btn.grid(row=10, column=3, sticky=tk.NSEW)
        btn = tk.Button(frame, text="Save current data and predictions", command = self.save_curr_data)
        btn.grid(row=11, column=1, columnspan=2, sticky=tk.NSEW)
        btn = tk.Button(frame, text="Load test data", command = self.plot_test_data)
        btn.grid(row=11, column=0, sticky=tk.NSEW)
    
    def plot_test_data(self):
        self.curr_x = self.test_data
        self.curr_target = pd.Series(self.test_data['CPU usage [%]'])
        time_index = pd.to_datetime(self.curr_x['Time (DD.MM.YYYY HH:MM:SS)'], format='%d.%m.%Y %H:%M:%S')
        self.date_start = time_index.iloc[0]
        self.date_end = time_index.iloc[-1] + datetime.timedelta(hours=1)

        self.curr_features, _ = self.generate_features_forecast() 
        self.axes.clear()
        self.is_annotated = False
        self.axes.plot(self.curr_target, label='CPU usage, %', zorder=2)
        self.is_train_plotted = False
        self.curr_med = max(self.curr_target.median(), 10)
        self.axes.legend()
        self.canvas.draw()

    def save_curr_data(self):
        self.curr_x.to_csv('./saved_data/curr_x.csv')
        self.curr_target.to_csv('./saved_data/curr_target.csv')
        self.curr_anom.to_csv('./saved_data/curr_anomaly_prediction.csv')        

    def forecast(self):
        data_x, self.curr_x = self.generate_features_forecast()
        self.curr_features = data_x
        target = self.forecaster.predict(data_x)
        self.curr_target = pd.Series(target)
        self.curr_med = max(np.median(target), 10)
        self.curr_x.insert(1, 'CPU usage [%]', target)
        self.axes.clear()
        self.axes.plot(self.curr_x['CPU usage [%]'], label='CPU usage, %', zorder=2)
        self.axes.legend()
        self.is_train_plotted = False
        self.is_annotated = False
        self.canvas.draw()

    def generate_x_data(self, date_list):
        x_data = pd.DataFrame()
        date_list = date_list.dt.strftime('%Y.%m.%d %H:%M:%S')
        x_data.insert(0, 'Time', date_list)
        return x_data
        
    def generate_features_forecast(self):
        base = datetime.datetime(day=self.date_start.day, month=self.date_start.month, 
                                 year=self.date_start.year, hour=0)
        hours_amount = (self.date_end - self.date_start).days  * 24 + (self.date_end - self.date_start).seconds // 3600
        date_list = [base + datetime.timedelta(hours=x) for x in range(hours_amount)]
        date_list = pd.Series(date_list)
        features = pd.DataFrame()
        features.insert(0, 'hour', date_list.dt.hour)
        features.insert(1, 'day', date_list.dt.day)
        features.insert(2, 'month', date_list.dt.month)
        features.insert(3, 'dayofweek', date_list.dt.dayofweek)
        return features, self.generate_x_data(date_list)

    def plot_train_data(self):
        self.curr_x = self.train_data
        self.axes.clear()
        self.is_annotated = False
        self.axes.plot(self.train_data['CPU usage [%]'], label='CPU usage, %', zorder=2)
        self.is_train_plotted = True
        self.curr_med = max(self.train_data['CPU usage [%]'].median(), 10)
        self.axes.legend()
        self.canvas.draw()

    def annotate_anomalies(self):
        if not self.is_annotated:
            if self.is_train_plotted:
                self.curr_anom = self.train_anom
                self.curr_target = self.train_data['CPU usage [%]']
            else:
                self.curr_features.insert(4, 'cpu', self.curr_target)
                self.curr_anom = pd.Series(self.detector.predict(self.curr_features))
                self.curr_anom *= self.curr_target > self.curr_med
            self.draw_anom_samples()
            self.is_annotated = True
        else:
            pass

    def draw_anom_samples(self):
        anomaly_inds = self.curr_anom > 0
        anomaly_inds *= self.curr_target > self.curr_med
        self.axes.scatter(pd.Series(self.curr_target[anomaly_inds].index), 
                              self.curr_target[anomaly_inds], color='r', label='Anomaly points',
                              zorder=3)
        self.axes.legend()
        self.canvas.draw()
    
    def choice_date(self):
        self.window = tk.Tk()
        self.window.geometry('200x200')
        self.window.title('Set forecasting interval')
        self.cal = Calendar(self.window, selectmode = 'day',
               year = 2024)
        self.cal.grid(row = 0, sticky=tk.NSEW)
        self.button_start = tk.Button(self.window, text = "Get Start Date",
                command = self.assign_date_start).grid(row=1, sticky=tk.NSEW)
        self.button_end = tk.Button(self.window, text = "Get End Date",
                command = self.assign_date_end).grid(row=2, sticky=tk.NSEW)

    def assign_date_start(self):
        t = tuple(self.cal.get_date().split('/'))
        month, day, year = int(t[0]), int(t[1]), 2000 + int(t[2])
        self.date_start = datetime.datetime(day=day, month=month, year=year, hour=0, minute=0, second=0)

    def assign_date_end(self):
        t = tuple(self.cal.get_date().split('/'))
        month, day, year = int(t[0]), int(t[1]), 2000 + int(t[2])
        self.date_end = datetime.datetime(day=day, month=month, year=year, hour=23, minute=0, second=0) + datetime.timedelta(hours=1)

    def destroy(self):
        self.root.destroy()
        exit()

    def start(self):
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self.root.mainloop()

if __name__ == '__main__':
    gui = Anoms_GUI()
    gui.start()