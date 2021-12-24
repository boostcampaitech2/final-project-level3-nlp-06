import mpl_finance as matfin
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
import base64 
from io import BytesIO
import FinanceDataReader as fdr
    
def visualize(start_date, end_date, stock_code):
    print(stock_code, type(stock_code))
    stock = fdr.DataReader(stock_code, start_date, end_date)
    fig = Figure(figsize=[12, 5])
    ax = fig.add_subplot(111)

    day_list = []
    name_list = []
    for i, day in enumerate(stock.index):
        if day.dayofweek == 0:
            day_list.append(i)
            name_list.append(day.strftime('%Y-%m-%d'))

    ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
    buf = BytesIO()
    matfin.candlestick2_ohlc(ax, stock['Open'], stock['High'], stock['Low'], stock['Close'], width=0.5, colorup='r', colordown='b')
    fig.savefig(buf, format='png')
    data = base64.b64encode(buf.getbuffer()).decode('ascii')
    return f'data:image/png;base64,{data}'