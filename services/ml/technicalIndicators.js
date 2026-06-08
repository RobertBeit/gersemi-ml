// src/services/technicalIndicators.js
// This file contains functions for calculating various technical indicators

/**
 * Calculates Simple Moving Average
 * @param {Array} data - Array of price data
 * @param {Number} period - Period for SMA calculation
 * @returns {Array} - Array of SMA values
 */
export const calculateSMA = (data, period) => {
    const sma = [];

    // Use a partial rolling window for early rows so features can be built
    // much earlier in the series instead of dropping the first ~period rows.
    for (let i = 0; i < data.length; i++) {
      const window = Math.min(period, i + 1);
      let sum = 0;
      for (let j = 0; j < window; j++) {
        sum += data[i - j];
      }
      sma.push(sum / window);
    }
    
    return sma;
  };
  
  /**
   * Calculates Exponential Moving Average
   * @param {Array} data - Array of price data
   * @param {Number} period - Period for EMA calculation
   * @returns {Array} - Array of EMA values
   */
  export const calculateEMA = (data, period) => {
    const ema = [];
    const multiplier = 2 / (period + 1);
    
    // Start with SMA for the first EMA value
    let initialSMA = 0;
    for (let i = 0; i < period; i++) {
      initialSMA += data[i];
    }
    initialSMA /= period;
    
    // Fill initial positions with NaN
    for (let i = 0; i < period - 1; i++) {
      ema.push(NaN);
    }
    
    // First EMA is the SMA
    ema.push(initialSMA);
    
    // Calculate EMA for remaining data
    for (let i = period; i < data.length; i++) {
      ema.push((data[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1]);
    }
    
    return ema;
  };
  
  /**
   * Calculates Moving Average Convergence Divergence (MACD)
   * @param {Array} data - Array of price data
   * @param {Number} fastPeriod - Fast EMA period (default: 12)
   * @param {Number} slowPeriod - Slow EMA period (default: 26)
   * @param {Number} signalPeriod - Signal line period (default: 9)
   * @returns {Object} - Object containing MACD line, signal line, and histogram
   */
  export const calculateMACD = (data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) => {
    const fastEMA = calculateEMA(data, fastPeriod);
    const slowEMA = calculateEMA(data, slowPeriod);
    
    // Calculate MACD line (fast EMA - slow EMA)
    const macdLine = [];
    for (let i = 0; i < data.length; i++) {
      if (isNaN(fastEMA[i]) || isNaN(slowEMA[i])) {
        macdLine.push(NaN);
      } else {
        macdLine.push(fastEMA[i] - slowEMA[i]);
      }
    }
    
    // Calculate signal line (EMA of MACD line)
    // First filter out NaN values for signal calculation
    const validMacdValues = macdLine.filter(val => !isNaN(val));
    const signalLine = calculateEMA(validMacdValues, signalPeriod);
    
    // Prepare final signal line with proper alignment
    const fullSignalLine = [];
    for (let i = 0; i < data.length - validMacdValues.length; i++) {
      fullSignalLine.push(NaN);
    }
    fullSignalLine.push(...signalLine);
    
    // Calculate histogram (MACD line - signal line)
    const histogram = [];
    for (let i = 0; i < data.length; i++) {
      if (isNaN(macdLine[i]) || isNaN(fullSignalLine[i])) {
        histogram.push(NaN);
      } else {
        histogram.push(macdLine[i] - fullSignalLine[i]);
      }
    }
    
    return {
      macdLine,
      signalLine: fullSignalLine,
      histogram
    };
  };
  
  /**
   * Calculates Relative Strength Index (RSI)
   * @param {Array} data - Array of price data
   * @param {Number} period - Period for RSI calculation (default: 14)
   * @returns {Array} - Array of RSI values
   */
  export const calculateRSI = (data, period = 14) => {
    const rsi = [];
    const gains = [];
    const losses = [];
    
    // Calculate initial price changes
    for (let i = 1; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    // Fill initial positions with NaN (we need period+1 data points to calculate first RSI)
    for (let i = 0; i < period; i++) {
      rsi.push(NaN);
    }
    
    // Calculate first average gain and loss
    let avgGain = gains.slice(0, period).reduce((sum, val) => sum + val, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((sum, val) => sum + val, 0) / period;
    
    // Calculate first RSI
    let rs = avgGain / (avgLoss === 0 ? 0.001 : avgLoss); // Avoid division by zero
    rsi.push(100 - (100 / (1 + rs)));
    
    // Calculate remaining RSI values using smoothed method
    for (let i = period; i < gains.length; i++) {
      avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
      avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
      
      rs = avgGain / (avgLoss === 0 ? 0.001 : avgLoss); // Avoid division by zero
      rsi.push(100 - (100 / (1 + rs)));
    }
    
    return rsi;
  };

/**
 * Builds training labels from local extrema with a required minimum move.
 * This yields denser and more meaningful bottom/peak supervision than strict
 * indicator-rule intersections.
 */
export const detectLocalExtremaSignals = (close, lookback = 5, lookforward = 5, minMovePct = 0.015) => {
  const bottomSignals = new Array(close.length).fill(0);
  const peakSignals = new Array(close.length).fill(0);

  for (let i = lookback; i < close.length - lookforward; i++) {
    const current = close[i];
    const prevWindow = close.slice(i - lookback, i + 1);
    const nextWindow = close.slice(i, i + lookforward + 1);

    const localMin = Math.min(...prevWindow, ...nextWindow);
    const localMax = Math.max(...prevWindow, ...nextWindow);

    const futureHigh = Math.max(...close.slice(i + 1, i + lookforward + 1));
    const futureLow = Math.min(...close.slice(i + 1, i + lookforward + 1));

    const bouncePct = current > 0 ? (futureHigh - current) / current : 0;
    const dropPct = current > 0 ? (current - futureLow) / current : 0;

    if (current <= localMin && bouncePct >= minMovePct) {
      bottomSignals[i] = 1;
    }

    if (current >= localMax && dropPct >= minMovePct) {
      peakSignals[i] = 1;
    }
  }

  return { bottomSignals, peakSignals };
};
  
  /**
   * Detects potential bottom/peak patterns based on technical indicators
   * @param {Object} stockData - Object containing OHLCV data
   * @returns {Object} - Object with features data for LSTM
   */
  export const extractTechnicalFeatures = (stockData) => {
    const { close, high, low, volume } = stockData;
    
    // Calculate SMAs
    const sma20 = calculateSMA(close, 20);
    const sma50 = calculateSMA(close, 50);
    const sma200 = calculateSMA(close, 200);
    
    // Calculate MACD
    const macd = calculateMACD(close);
    
    // Calculate RSI
    const rsi = calculateRSI(close);
    
    // Calculate Volatility (20-day standard deviation of returns)
    const returns = [];
    returns.push(0); // First day has no return
    for (let i = 1; i < close.length; i++) {
      returns.push((close[i] / close[i - 1]) - 1);
    }
    
    const volatility = [];
    for (let i = 0; i < returns.length; i++) {
      if (i < 20) {
        volatility.push(NaN);
      } else {
        const windowReturns = returns.slice(i - 20, i);
        const mean = windowReturns.reduce((sum, val) => sum + val, 0) / 20;
        const squaredDiffs = windowReturns.map(val => Math.pow(val - mean, 2));
        const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / 20;
        volatility.push(Math.sqrt(variance));
      }
    }
    
    // Calculate volume ratio (current volume / 20-day avg volume)
    const volSMA20 = calculateSMA(volume, 20);
    const volumeRatio = volume.map((vol, i) => isNaN(volSMA20[i]) ? NaN : vol / volSMA20[i]);
    
    // Calculate price position (where current price is relative to its 52-week range)
    // Uses a growing window up to 252 days, starting once 20 data points are available
    const pricePosition = [];
    const maxLookback = 252;
    const minLookback = 20;
    
    for (let i = 0; i < close.length; i++) {
      if (i < minLookback - 1) {
        pricePosition.push(NaN);
      } else {
        const windowSize = Math.min(maxLookback, i + 1);
        const window = close.slice(i - windowSize + 1, i + 1);
        const min = Math.min(...window);
        const max = Math.max(...window);
        pricePosition.push(max === min ? 0.5 : (close[i] - min) / (max - min));
      }
    }
    
    const { bottomSignals, peakSignals } = detectLocalExtremaSignals(close, 5, 5, 0.015);
    
    // Compile all features into a single dataset
    const features = [];
    
    for (let i = 0; i < close.length; i++) {
      // Skip entries with NaN values
      if (isNaN(sma20[i]) || isNaN(sma50[i]) || isNaN(sma200[i]) || 
          isNaN(macd.macdLine[i]) || isNaN(macd.signalLine[i]) || 
          isNaN(rsi[i]) || isNaN(volatility[i]) || isNaN(volumeRatio[i]) || 
          isNaN(pricePosition[i])) {
        continue;
      }
      
      // Create feature vector
      features.push({
        date: stockData.dates[i],
        close: close[i],
        sma20Ratio: close[i] / sma20[i],
        sma50Ratio: close[i] / sma50[i],
        sma200Ratio: close[i] / sma200[i],
        macd: macd.macdLine[i],
        macdSignal: macd.signalLine[i],
        macdHistogram: macd.histogram[i],
        rsi: rsi[i],
        volatility: volatility[i],
        volumeRatio: volumeRatio[i],
        pricePosition: pricePosition[i],
        bottomSignal: bottomSignals[i],
        peakSignal: peakSignals[i]
      });
    }
    
    return features;
  };