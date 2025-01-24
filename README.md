# Stock-Market-Analysis-Project
## This is a solo project that showcases my skills and my interests in the field of finance, programming data- analysis and visualization
## The followings are demonstrated via the project:
### Python object-oriented programming:
  Using classes and functions and also utilizing nest classes and nested functions when needed
### API usage (AlphaVantage)
### The following python libraries:
  * **pandas** data analysis and data handling
  * **pandas_ta** technical, financial analysis
  * **numpy** data handling for scikit-learn
  * **matplotlib** data visualization
  * **tkinter** user interference and application
  * **scikit-learn** linear regression model
### Finance:
  The project is based on the stock market, as it is a field of interest of mine.
  In the project, I use an algorithmic trading model based on the indicators. You can find a more detailed description in the *indicators.txt* file 

## Guide:
  When running the program, a disclaimer shows up and the main file starts running. On the first window we need to input our Alphavantage API key and the stock symbol we are interested in. Then we have the dropdown menu of the time selection. This works with the 'Import as .csv file' and 'Show table' buttons, where we can either download the *.csv* file or view it in another tkiner window. Under that we have the 'Show graph' option which shows an interactive plot of the adjusted closing prices of the chosen stock. If we go to the 'Technical analyis' button, the program opens up another window made for the technical part. In this we have 5 buttons for indicators, they show - on an interactive graph - the price and the indicator for visualization. Then we have our 'Suggestion / Complete analysis' button, which runs the trading algorithm model on the searched stock and gives back a rating from 'Strong Sell' to 'Strong Buy'. The last button is a simple linear regression.
