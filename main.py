from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import tkinter as tk
from tkinter import messagebox, ttk
from tabulate import tabulate
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import ImageTk, Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn import linear_model

class StockDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Stock Data')
        self.root.iconbitmap("stock.ico")
        self.root.geometry('500x300')

        # Create the GUI elements
        self.prompt_label = tk.Label(self.root, text='Search symbol', font=("Arial", 14))
        self.prompt_label.pack()

        self.search_entry = tk.Entry(self.root, font=('Arial', 12), width=20)
        self.search_entry.pack(pady=3)

        self.symbol = self.search_entry.get().strip()

        self.time_label = tk.Label(self.root, text='Search time period', font=('Arial', 14))
        self.time_label.pack(pady=10)

        # Using a dropdown menu the select the timeframes
        self.selected_timeframe = tk.StringVar(root)
        self.selected_timeframe.set("ALL")
        self.dropdown_timeframe = tk.OptionMenu(root, self.selected_timeframe, "1Y", "3Y", "5Y", "ALL")
        self.dropdown_timeframe.pack(pady=4)

        self.table_button = tk.Button(self.root, text='Show table', command=self.Show)
        self.table_button.pack(pady=10)

        self.graph_button = tk.Button(self.root, text='Show graph', command=self.Show_graph)
        self.graph_button.pack(pady=10)

        # Create another button for the techincal analysis
        self.ta_button = tk.Button(self.root, text='Techincal Analysis', command=self.Open_ta)
        self.ta_button.pack(pady=10)

    # Create a dark style format which can be recalled later in the plottings
    def Get_dark_style(self): # This function only returns a dictionary containing the parameters for the dark style
        return  {
                "axes.facecolor": "#222222",
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "figure.facecolor": "#222222",
                "grid.color": "gray",
                "text.color": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "xtick.major.size": 7,
                "ytick.major.size": 7,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.linewidth": 0.5,
                }

    def Get_data(self, API='I7P51WLUI128TBGE'):

        self.symbol = self.search_entry.get().strip()

        # Get the data and handling wrong symbol inputs
        if not self.symbol:
            messagebox.showerror("Error", "Please enter a stock symbol.")
            return None
        try:
            ts = TimeSeries(key=API, output_format='pandas')
            symbol_data = ts.get_weekly_adjusted(self.symbol)[0]
            symbol_data = symbol_data.rename(columns=
                                               {"1. open": "Open",
                                                "2. high": "High",
                                                "3. low": "Low",
                                                "4. close": "Raw_close",
                                                "5. adjusted close": "Close",
                                                "6. volume": "Volume",
                                                "7. divident amount": "Divident amount"
                                                })
            return symbol_data
        except KeyError:
            messagebox.showerror("Error", f"Invalid stock symbol: '{self.symbol}'. Check the stock market to find a real symbol.")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occured: {e}")

        # Filtering the data and error handling
        timeframe = self.selected_timeframe.get()

        len_data = len(symbol_data)
        if timeframe == "ALL":
            filtered_data = symbol_data.copy()
        elif timeframe == "1Y" and len_data >= 52:
            filtered_data = symbol_data.head(52)
            return filtered_data
        elif timeframe == "1Y" and len_data < 52:
            messagebox.showerror("Error", f"Stock does not have {timeframe[:1]} years of data.")
            return None
        elif timeframe == "3Y" and len_data >= 156:
            filtered_data = symbol_data.head(156)
            return filtered_data
        elif timeframe == "3Y" and len_data < 156:
            messagebox.showerror("Error", f"Stock does not have {timeframe[:1]} years of data.")
            return None
        elif timeframe == "5Y" and len_data >= 260:
            filtered_data = symbol_data.head(260)
            return filtered_data
        elif timeframe == "5Y" and len_data < 260:
            messagebox.showerror("Error", f"Stock does not have {timeframe[:1]} years of data.")
            return None

    def Show(self):
        # Get the data
        filtered_data = self.Get_data()

        # Create the formatted table
        formatted_table = tabulate(filtered_data, headers='keys', tablefmt='pretty', showindex=True)

        # Create a new window to show the data
        show_window = tk.Toplevel(self.root)
        show_window.title("Data Table")
        show_window.iconbitmap("stock.ico")
        show_window.geometry("1000x400")

        # Fill the window with the data and a slider
        slider = tk.Scale(root, resolution=2, orient="vertical")
        slider.pack
        text_widget = tk.Text(show_window, wrap="none", font=('Courier New', 10))
        text_widget.pack(expand=True, fill="both", padx=10, pady=10)
        text_widget.insert("1.0", formatted_table)
        text_widget.config(state="disabled")

    def Show_graph(self):
        # Get the data
        filtered_data = self.Get_data()

        # Create a new window for the graph
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Stock Graph")
        graph_window.iconbitmap("stock.ico")
        graph_window.geometry("700x600")

        # Create the Matplotlib figure using the predifned dark_stlye
        dark_style = self.Get_dark_style()
        plt.style.use(dark_style)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Closing Price')
        ax.plot(filtered_data.index, filtered_data['Close'], label='Adjusted Closing Price', color='lime', linewidth = '2')
        ax.legend()

        # Embed the figure in the Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add Matplotlib navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def Open_ta(self): # Helper function
        self.ta = self.Technical_analysis(self)

    class Technical_analysis(): # Adding a nested class so that later the app can be upgraded more easily
        def __init__(self, app):
            self.app = app
            # Initialize the dark style method
            self.dark_style = self.app.Get_dark_style()
            plt.style.use(self.dark_style)

            # Create the new window
            self.ta_window = tk.Toplevel(self.app.root)
            self.ta_window.title("Technical Analysis Toolbox")
            self.ta_window.iconbitmap("analysis.ico")
            self.ta_window.geometry("800x600")

            # Fetch the data and save as a copy
            # Format the data so that it is working with the API and the function too and can be reused in the indicators
            self.ta_data = pd.DataFrame(self.app.Get_data().copy())
            self.ta_data = self.ta_data.rename(columns=
                                               {"1. open": "open",
                                                "2. high": "high",
                                                "3. low": "low",
                                                "4. close": "raw_close",
                                                "5. adjusted close": "close",
                                                "6. volume": "volume",
                                                "7. divident amount": "divident amount"
                                                })
            # NOTE: I need to flip the whole dataframe, as the API returns the current date as the first row, and for the calculations I need the last row as the first one
            self.ta_data = self.ta_data.iloc[::-1]

            # Fill the window with elements
            self.ta_label = tk.Label(self.ta_window, text=f'You have successfully fetched the data for the {self.app.symbol} stock. Select tools from below.', wraplength=400, justify="center", font=("Arial", 14))
            self.ta_label.pack(pady=10)

            # Create the buttons (analysis tools) on the window
            self.MACD_button = tk.Button(self.ta_window, text="MACD", command=self.MACD)
            self.MACD_button.pack(pady=10, padx=10)

            self.RSI_button = tk.Button(self.ta_window, text="RSI", command=self.RSI)
            self.RSI_button.pack(pady=10, padx=5)

            self.ATR_button = tk.Button(self.ta_window, text="ATR", command=self.ATR)
            self.ATR_button.pack(pady=10, padx=0)

            self.BB_button = tk.Button(self.ta_window, text="BBs", command=self.BBs)
            self.BB_button.pack(pady=20, padx=7.5)

            self.VWAP_button = tk.Button(self.ta_window, text="VWAP", command=self.VWAP)
            self.VWAP_button.pack(pady=20, padx=2.5)

            self.Strat_button = tk.Button(self.ta_window, text="Suggestion / Complete analysis", command=self.Suggestion)
            self.Strat_button.pack(pady=10)

            # Create the button for the predictions
            self.pred_button = tk.Button(self.ta_window, text='Simple Linear Regression', command=self.LinReg)
            self.pred_button.pack(pady=10)

            self.help_icon = tk.PhotoImage(file="resized_question.png")
            self.help_button = tk.Button(self.ta_window, image=self.help_icon, command=self.Open_help)
            self.help_button.image = self.help_icon
            self.help_button.place(relx=1.0, x=-10, y=10, anchor="ne")

            self.dark_style = self.dark_style


        def Open_help(self):
            
            help_window = tk.Toplevel(self.ta_window)
            help_window.title("Help for the indicators")
            help_window.iconbitmap("question.ico")
            help_window.geometry("600x300")
            self.help_label = tk.Label(help_window, text="This is a short guide for the indicators used in the programme", justify='center', font=('Arial', 14))
            self.help_label.pack(pady=10)

            # Create a Text widget to display the file contents
            text_area = tk.Text(help_window, wrap='word', font=('Arial', 12))
            text_area.pack(expand=True, fill='both', padx=10, pady=10)

            # Try to open and load the contents of the file into the Text widget
            with open("indicators.txt", "r", encoding="utf-8") as file:
                file_content = file.read()
                text_area.insert("1.0", file_content)



        def MACD(self):
            # Calculate MACD
            self.macddf = self.ta_data.ta.macd(fast=12, slow=26, signal=9, min_periods=None, append=True) 

            # Prepare the data for plotting
            histogram = self.macddf['MACDh_12_26_9']

            # Filter histogram into positive and negative values
            positive_hist = histogram[histogram > 0]
            negative_hist = histogram[histogram < 0]

            # Plot MACD and signal lines
            plt.figure(figsize=(14, 7))
            plt.plot(self.macddf['MACD_12_26_9'], label='MACD Line', color='blue')
            plt.plot(self.macddf['MACDs_12_26_9'], label='Signal Line', color='orange')

            # Plot histogram bars
            plt.bar(positive_hist.index, positive_hist, color='green', label="Positive Histogram", width=1.5)
            plt.bar(negative_hist.index, negative_hist, color='red', label="Negative Histogram", width=1.5)

            # Finalize the plot
            plt.legend(loc='best')
            plt.title('MACD and Signal Line with Histogram')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.show()

        def RSI(self):
            # Calculate RSI
            self.rsidf = self.ta_data.ta.rsi(close=self.ta_data["Close"], lenght=14)
            
            # Plot RSI
            plt.figure(figsize=(14, 7))
            plt.plot(self.rsidf, label='RSI', color='blue')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.ylim(0, 100)
            plt.ylabel('RSI')
            plt.legend(loc='best')

            plt.show()
    
        def ATR(self):
            # Calculate ATR
            self.atrdf = self.ta_data.ta.atr(high=self.ta_data["High"], low=self.ta_data["Low"], close=self.ta_data["Close"])

            # Plot ATR
            plt.figure(figsize=(14, 7))
            plt.plot(self.atrdf, label='ATR', color='orange')
            plt.ylabel('ATR')
            plt.xlabel('Date')
            plt.legend(loc='best')

            plt.title('Average True Range (ATR)', color='white')
            plt.show()


        def BBs(self):
            # Calc Bollinger Bands
            self.bbsdf = self.ta_data.ta.bbands(close = self.ta_data["Close"], length=14 )

            # Plot BBs
            plt.figure(figsize=(14, 7))
            plt.plot(self.bbsdf["BBL_14_2.0"], label="Bands", color="red")
            plt.plot(self.bbsdf["BBU_14_2.0"], color="red")
            plt.plot(self.ta_data["Close"], label="Price", color="blue")
            plt.ylabel("Price")
            plt.xlabel('Date')
            plt.legend(loc='best')
            
            plt.title("Price with Bollinger Bands", color='white')
            plt.show()

        def VWAP(self):
            # Note: the VWAP is commonly used in intraday trading, which with the free Alphavantage API I do not have acces to, but with the paid version it can be used
            # Also the function has an "anchor" argument, which can be set to weekly, so this is how I am going to use it, altough it might not the most effective
            # Calc VWAP
            self.vwapdf = self.ta_data.ta.vwap(high=self.ta_data["High"], low=self.ta_data["Low"], close=self.ta_data["Raw_close"], volume=self.ta_data["Volume"], anchor="W")

            # Plot VWAP
            plt.figure(figsize=(14, 7))
            plt.plot(self.ta_data["Raw_close"], label="Price", color="blue")
            plt.plot(self.vwapdf, label="VWAP Weekly", color="orange")
            plt.ylabel("Price")
            plt.xlabel("Date")
            plt.legend(loc='best')

            plt.title("Price with Volume-Weighted Average Price", color='white')
            plt.show()

        def Suggestion(self): # This function creates a pandas-ta strategy from the above indicators, uses is to create the trading algorithm and displays the result
            
            self.ta_data.drop("7. dividend amount", axis=1, inplace=True) # drop the divident column as it is not needed in the calculations

            # First use the pandas-ta Strategy function to calculate all the desired indicators in one dataframe
            ta_strategy = ta.Strategy(name='TA Strategy',
                                        ta=[
                                            {"kind": "macd", "fast" : 12, "slow" : 26, "signal" : 9},
                                            {"kind": "rsi", "lenght" : 14},
                                            {"kind": "atr"},
                                            {"kind": "bbands", "length": 14},
                                            {"kind": "vwap", "anchor" : "W"}                              
                                        ]
                                        )
            self.ta_data.ta.strategy(ta_strategy)

            # Now create the indicator specific columns
            MACD_signal = []
            for i in range (0, len(self.ta_data['MACD_12_26_9'])):
                if self.ta_data['MACDh_12_26_9'][i] > 0 and self.ta_data['MACDh_12_26_9'][i-1] < 0:
                    MACD_signal.append(+1)
                elif self.ta_data['MACDh_12_26_9'][i] < 0 and self.ta_data['MACDh_12_26_9'][i-1] > 0:
                    MACD_signal.append(-1)
                else:
                    MACD_signal.append(0)
            self.ta_data['MACD_signal'] = MACD_signal

            RSI_signal = []
            for i in range (0, len(self.ta_data['RSI_14'])):
                if self.ta_data['RSI_14'][i] < 30:
                    RSI_signal.append(+1)
                elif self.ta_data['RSI_14'][i] > 70:
                    RSI_signal.append(-1)
                else:
                    RSI_signal.append(0)
            self.ta_data['RSI_signal'] = RSI_signal

            BBs_signal = []
            for i in range (0, len(self.ta_data['BBM_14_2.0'])):
                if self.ta_data['Close'][i] < self.ta_data['BBL_14_2.0'][i]:
                    BBs_signal.append(+1)
                elif self.ta_data['Close'][i] > self.ta_data['BBU_14_2.0'][i]:
                    BBs_signal.append(-1)
                else:
                    BBs_signal.append(0)
            self.ta_data['BBs_signal'] = BBs_signal

            VWAP_signal = []
            for i in range (0, len(self.ta_data['VWAP_W'])):
                if self.ta_data['VWAP_W'][i] < self.ta_data['Close'][i]:
                    VWAP_signal.append(+1)
                elif self.ta_data['VWAP_W'][i] > self.ta_data['Close'][i]:
                    VWAP_signal.append(-1)
                else:
                    VWAP_signal.append(0)
            self.ta_data['VWAP_signal'] = VWAP_signal

            # First create the rolling average for the ATR's signal multiplier's base (with the last year's data (if available))
            self.ta_data['Rolling_ATR_base'] = self.ta_data['ATRr_14'].rolling(window=53, min_periods=1).mean()
            # Then calculate the ATR_signal (multiplier)
            self.ta_data['ATR_signal'] = (self.ta_data['ATRr_14'] / self.ta_data['Rolling_ATR_base']).round()

            # Now just add the final signal / suggestion column that will give the algorithm's suggestion

            self.ta_data['Final_signal_all'] = (self.ta_data['MACD_signal'] + self.ta_data['RSI_signal'] + self.ta_data['BBs_signal'] + self.ta_data['VWAP_signal']) * self.ta_data['ATR_signal']
            # Taking the average of the last 5 suggestions to have a more certain and conservative model
            self.ta_data['Final_signal'] = (self.ta_data['Final_signal_all'].rolling(window=5, min_periods=1).mean()).round()

            # Create the final suggestion
            suggestion = []
            for value in self.ta_data['Final_signal']:
                if value >= 5 or value == 4:
                    suggestion.append('Strong Buy')
                elif value == 3 or value == 2:
                    suggestion.append('Buy')
                elif value == 1 or value == 0 or value == -1:
                    suggestion.append('Hold')
                elif value == -2 or value == -3:
                    suggestion.append('Sell')
                elif value == -4 or value <= -5:
                    suggestion.append('Strong Sell')
                else:
                    suggestion.append('Undefined')
            self.ta_data['Suggestion'] = suggestion

            #print(self.ta_data[['Close', 'MACD_signal', 'RSI_signal', 'BBs_signal', 'VWAP_signal', 'ATR_signal', 'Final_signal_all', 'Final_signal', 'Suggestion']][500:540])

            # Displaying final result (what to do with the stock now)
            Suggestion_window = tk.Toplevel(self.ta_window)
            Suggestion_window.title("Suggestion")
            Suggestion_window.iconbitmap("calc.ico")
            Suggestion_window.geometry("350x300")

            label = tk.Label(Suggestion_window, text=f"The result of the algorithmic technical analysis of the {self.app.symbol} stock is:", font=('Arial', 14), wraplength=250)
            label.pack(pady=20)

            text = self.ta_data['Suggestion'][-1]
            color = ("green" if text in ["Strong Buy", "Buy"] else
                    "gray" if text == "Hold" else
                    "red")
            suggestion_label = tk.Label(Suggestion_window, text=text, font=("Trade Gothic Next Heavy", 16), fg=color, padx=10, pady=5)
            suggestion_label.pack(pady=10)

            bull = Image.open("bull-market.png")
            bull = ImageTk.PhotoImage(bull)

            bull_label = tk.Label(Suggestion_window, image=bull)
            bull_label.pack(padx=5)

            return self.ta_data

        def LinReg(self):
            # Linear Regression
            X = np.array((self.ta_data.index - self.ta_data.index[0]).days).reshape(-1, 1)  # Independent variable
            y = self.ta_data['Close'].values  # Dependent variable

            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # Visualization
            plt.style.use(self.dark_style)

            plt.figure(figsize=(10, 6))
            plt.plot(self.ta_data.index, y, label='Actual Prices', color='blue', alpha=0.8)
            plt.plot(self.ta_data.index, y_pred, label='Linear Regression Line', color='green')
            plt.title('Price Trend with Linear Regression', color='white')
            plt.text(self.ta_data.index[5], max(y) - 55, f"Slope: {model.coef_[0]:.4f}", color='white', fontsize=12, bbox=dict(facecolor='#333333', alpha=0.7))
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend(facecolor='#222222', edgecolor='white', labelcolor='white')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = StockDataApp(root)
    root.mainloop()
