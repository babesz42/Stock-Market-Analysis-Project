from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import pandas_ta as ta
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
from sklearn import linear_model

class StockDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Stock Data')
        self.root.iconbitmap("stock.ico")
        self.root.geometry('500x450')


        # Create the GUI elements
        self.api_label = tk.Label(self.root, text='Enter AlphaVantage API key', font=('Arial', '14'))
        self.api_label.pack()
        self.api_entry = tk.Entry(self.root, font=('Arial', 12), width=20)
        self.api_entry.pack(pady=3)

        self.prompt_label = tk.Label(self.root, text='Search symbol', font=("Arial", 14))
        self.prompt_label.pack(pady=5)

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

        self.import_button = tk.Button(self.root, text="Import as .csv file", command=self.Import_csv)
        self.import_button.pack(pady=5)

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

    def Get_data(self):
        # Get the required data from the entries
        API = self.api_entry.get().strip()
        if not API:
            messagebox.showerror("Error", "Please enter an API key.\nYou can redeem a free API key from https://www.alphavantage.co/support/#api-key")
            return None

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
            return filtered_data
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
        
    def Import_csv(self):
        # Get the data
        self.filtered_data = self.Get_data()

        # Create a new window where the user can input the filepath
        import_window = tk.Toplevel(self.root)
        import_window.title(f"Import {self.symbol} data")
        import_window.iconbitmap("download.ico")
        import_window.geometry("400x200")

        # Add the widgets
        filepath_label = tk.Label(import_window, text="Enter filepath for saving location", font=('Arial', 14))
        filepath_label.pack(pady=10)

        filepath_entry = tk.Entry(import_window, font=('Arial', 12), width=30)
        filepath_entry.pack(pady=5)

        def Import_command(): # This function saves the data and can be recalled for the button
            filepath = str(filepath_entry.get())
            if filepath:
                try:
                    self.filtered_data.to_csv(filepath, index=False)
                    tk.messagebox.showinfo("Success", f"Data saved to {filepath}")
                    import_window.destroy()
                except Exception as e:
                    tk.messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            else:
                tk.messagebox.showwarning("Warning", "Please enter a valid file path.")

        import_button = tk.Button(import_window, text="Import", command=Import_command)
        import_button.pack(pady=5)



    def Show(self):
        # Get the data
        filtered_data = self.Get_data()

        # Create the formatted table
        formatted_table = tabulate(filtered_data, headers='keys', tablefmt='pretty', showindex=True)

        # Create a new window to show the data
        show_window = tk.Toplevel(self.root)
        show_window.title(f"Data Table of {self.symbol}")
        show_window.iconbitmap("stock.ico")
        show_window.geometry("1000x400")
        # Create grid layout
        show_window.columnconfigure(0, weight=1)  # Make the first column expandable
        show_window.rowconfigure(0, weight=1)    # Make the first row expandable

        # Add the Text widget
        text_widget = tk.Text(show_window, wrap="none", font=('Courier New', 10))
        text_widget.grid(row=0, column=0, sticky="nsew", padx=(10, 10), pady=10)
        text_widget.insert("1.0", formatted_table)
        text_widget.config(state="disabled")
        show_window.columnconfigure(1, weight=0)

        # Add the vertical scrollbar
        scrollbar = tk.Scrollbar(show_window, orient="vertical", command=text_widget.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 20))  # Attach to the right of the Text widget with space
        text_widget.config(yscrollcommand=scrollbar.set)

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
        ax.plot(filtered_data.index, filtered_data['Close'], label='Adjusted Closing Price', color='blue', linewidth = '2')
        ax.legend()
        ax.set_title(f"Adjusted closing price of {self.symbol}")

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

            # Visualize the plot, with the closing prices as subplot
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Plot stock price on the first subplot
            axes[0].plot(self.ta_data.index, self.ta_data['Close'], label='Stock Price', color='blue')
            axes[0].set_title('Stock Price')
            axes[0].set_ylabel('Price')
            axes[0].legend(loc='best')
            axes[0].grid(True, linestyle='--', linewidth=0.5)

            # Plot MACD and histogram on the second subplot
            axes[1].plot(self.macddf.index, self.macddf['MACD_12_26_9'], label='MACD Line', color='cyan')
            axes[1].plot(self.macddf.index, self.macddf['MACDs_12_26_9'], label='Signal Line', color='orange')
            axes[1].bar(positive_hist.index, positive_hist, color='green', label='Positive Histogram', width=1.5)
            axes[1].bar(negative_hist.index, negative_hist, color='red', label='Negative Histogram', width=1.5)
            axes[1].set_title('MACD and Signal Line with Histogram')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Value')
            axes[1].legend(loc='best')
            axes[1].grid(True, linestyle='--', linewidth=0.5)

            # Finalize plot
            plt.tight_layout()
            plt.show()

        def RSI(self):
            # Calculate RSI
            self.rsidf = self.ta_data.ta.rsi(close=self.ta_data["Close"], lenght=14)
            
            # Create subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)  # 2 rows, 1 column, shared x-axis

            # Plot stock price on the first subplot
            axes[0].plot(self.ta_data.index, self.ta_data['Close'], label='Stock Price', color='blue')
            axes[0].set_title('Stock Price')
            axes[0].set_ylabel('Price')
            axes[0].legend(loc='best')
            axes[0].grid(True, linestyle='--', linewidth=0.5)

            # Plot RSI on the second subplot
            axes[1].plot(self.rsidf.index, self.rsidf, label='RSI', color='lime')
            axes[1].axhline(70, color='red', linestyle='--', label='Overbought (70)')
            axes[1].axhline(30, color='green', linestyle='--', label='Oversold (30)')
            axes[1].set_ylim(0, 100)
            axes[1].set_title('RSI (Relative Strength Index)')
            axes[1].set_ylabel('RSI')
            axes[1].legend(loc='best')
            axes[1].grid(True, linestyle='--', linewidth=0.5)

            # Finalize plot
            plt.tight_layout()
            plt.show()
    
        def ATR(self):
            # Calculate ATR
            self.atrdf = self.ta_data.ta.atr(high=self.ta_data["High"], low=self.ta_data["Low"], close=self.ta_data["Close"])
            
            # Create subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)  # 2 rows, 1 column, shared x-axis

            # Plot stock price on the first subplot
            axes[0].plot(self.ta_data.index, self.ta_data['Close'], label='Stock Price', color='blue')
            axes[0].set_title('Stock Price')
            axes[0].set_ylabel('Price')
            axes[0].legend(loc='best')
            axes[0].grid(True, linestyle='--', linewidth=0.5)

            # Plot ATR on the second subplot
            axes[1].plot(self.atrdf.index, self.atrdf, label='ATR', color='orange')
            axes[1].set_title('Average True Range (ATR)')
            axes[1].set_ylabel('ATR')
            axes[1].legend(loc='best')
            axes[1].grid(True, linestyle='--', linewidth=0.5)

            # Finalize plot
            plt.tight_layout()
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
    tk.messagebox.showwarning("Disclaimer", "This is not financial advice, the creater does not take any responsibility for the program's wrong predictions/mistakes.")
    root.mainloop()
