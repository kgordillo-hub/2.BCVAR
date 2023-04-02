import yfinance as yf

class YahooData:
    def __init__(self, startDate, endDate):
        self.startDate = startDate
        self.endDate = endDate

    def getYahooData(self, keyword):
        historical_data = yf.download(keyword, start=self.startDate, end=self.endDate, progress=False)
        return historical_data