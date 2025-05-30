import yfinance as yf
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords

def test_talib():
    """Test TA-Lib installation"""
    close = np.array([90.0, 92.5, 91.8, 94.2, 93.5], dtype=float)
    rsi = talib.RSI(close, timeperiod=3)
    assert not np.isnan(rsi[-1]), "TA-Lib RSI calculation failed"

def test_yfinance():
    """Test yfinance data fetching"""
    data = yf.download("AAPL", period="1d", progress=False)
    assert not data.empty, "Failed to fetch stock data"

def test_nltk():
    """Test NLTK resources"""
    assert 'the' in stopwords.words('english'), "NLTK stopwords missing"
    
def test_textblob():
    """Test sentiment analysis"""
    blob = TextBlob("Excellent quarterly results")
    assert blob.sentiment.polarity > 0.7, "Sentiment analysis failed"