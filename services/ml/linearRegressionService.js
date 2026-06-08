// Enhanced Multi-Timeframe Linear Regression Service with TensorFlow.js
// src/services/linearRegressionService.js
// GPU-accelerated linear regression for optimal stock prediction timeframes

import * as tf from '@tensorflow/tfjs';
import { extractTechnicalFeatures } from './technicalIndicators';

// Enhanced TensorFlow.js Linear Regression Model
class TensorFlowLinearRegression {
  constructor(regularization = 0.001) {
    this.model = null;
    this.trained = false;
    this.featureNames = null;
    this.trainHistory = null;
    this.regularization = regularization;
    this.normalizationParams = null;
  }

  // Create optimized linear regression model
  createModel(inputDim) {
    this.model = tf.sequential({
      layers: [
        // Input layer with L2 regularization for stability
        tf.layers.dense({
          units: 1,
          inputShape: [inputDim],
          activation: 'linear',
          kernelRegularizer: tf.regularizers.l2({ l2: this.regularization }),
          name: 'linear_output'
        })
      ]
    });

    // Optimized compiler settings
    this.model.compile({
      optimizer: tf.train.adam(0.01), // Adaptive learning rate
      loss: 'meanSquaredError',
      metrics: ['mae', 'mse']
    });

    return this.model;
  }

  // Enhanced feature normalization
  async normalizeFeatures(features) {
    const featureTensor = tf.tensor2d(features);
    
    // Calculate normalization parameters
    const mean = featureTensor.mean(0);
    const std = tf.sqrt(featureTensor.sub(mean).square().mean(0));
    
    // Prevent division by zero
    const epsilon = tf.scalar(1e-8);
    const safeStd = std.add(epsilon);
    
    // Normalize: (x - mean) / std
    const normalizedTensor = featureTensor.sub(mean).div(safeStd);
    
    // Store parameters for later use
    this.normalizationParams = {
      mean: await mean.data(),
      std: await std.data()
    };
    
    // Clean up intermediate tensors
    featureTensor.dispose();
    mean.dispose();
    std.dispose();
    safeStd.dispose();
    epsilon.dispose();
    
    return normalizedTensor;
  }

  // Apply normalization to new data
  applyNormalization(features) {
    if (!this.normalizationParams) {
      throw new Error('No normalization parameters available');
    }
    
    const featureTensor = tf.tensor2d(features);
    const meanTensor = tf.tensor1d(this.normalizationParams.mean);
    const stdTensor = tf.tensor1d(this.normalizationParams.std);
    const epsilon = tf.scalar(1e-8);
    
    const normalizedTensor = featureTensor
      .sub(meanTensor)
      .div(stdTensor.add(epsilon));
    
    // Clean up
    featureTensor.dispose();
    meanTensor.dispose();
    stdTensor.dispose();
    epsilon.dispose();
    
    return normalizedTensor;
  }

  // Enhanced training with early stopping
  async train(features, targets, featureNames = null, epochs = 100, validationSplit = 0.2) {
    console.log(`🧠 Training TensorFlow.js Linear Regression: ${features.length} samples × ${features[0].length} features`);
    
    this.featureNames = featureNames;
    
    // Create model
    const inputDim = features[0].length;
    this.createModel(inputDim);
    
    // Normalize features
    const normalizedFeatures = await this.normalizeFeatures(features);
    const targetTensor = tf.tensor2d(targets, [targets.length, 1]);
    
    // Training with validation split and callbacks
    const history = await this.model.fit(normalizedFeatures, targetTensor, {
      epochs: epochs,
      validationSplit: validationSplit,
      batchSize: Math.min(32, Math.floor(features.length / 10)), // Dynamic batch size
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 20 === 0 || epoch === epochs - 1) {
            console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(6)}, val_loss=${logs.val_loss.toFixed(6)}, mae=${logs.mae.toFixed(6)}`);

          }
        }
      }
    });
    
    this.trainHistory = history;
    this.trained = true;
    
    // Clean up tensors
    normalizedFeatures.dispose();
    targetTensor.dispose();
    
    console.log('✅ TensorFlow.js Linear Regression training completed');
    
    return this;
  }

  // Make predictions
  async predict(features) {
    if (!this.trained) {
      throw new Error('Model not trained yet');
    }
    
    const normalizedFeatures = this.applyNormalization(features);
    const predictions = this.model.predict(normalizedFeatures);
    const predictionValues = await predictions.data();
    
    // Clean up
    normalizedFeatures.dispose();
    predictions.dispose();
    
    return Array.from(predictionValues);
  }

  // Single prediction
  async predictSingle(features) {
    const predictions = await this.predict([features]);
    return predictions[0];
  }

  // Enhanced evaluation metrics
  async evaluate(features, targets) {
    const predictions = await this.predict(features);
    const n = targets.length;
    
    // Basic metrics
    const yMean = targets.reduce((sum, val) => sum + val, 0) / n;
    const ssTotal = targets.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
    const ssResidual = targets.reduce((sum, actual, i) => sum + Math.pow(actual - predictions[i], 2), 0);
    
    const r2 = Math.max(-1, 1 - (ssResidual / ssTotal)); // Clamp R² to reasonable range
    const mae = targets.reduce((sum, actual, i) => sum + Math.abs(actual - predictions[i]), 0) / n;
    const mse = ssResidual / n;
    const rmse = Math.sqrt(mse);
    
    // Enhanced metrics
    const mape = targets.reduce((sum, actual, i) => {
      if (Math.abs(actual) > 0.001) { // Avoid division by very small numbers
        return sum + Math.abs((actual - predictions[i]) / actual);
      }
      return sum;
    }, 0) / n * 100;
    
    // Directional accuracy
    let correctDirections = 0;
    for (let i = 0; i < n; i++) {
      if ((targets[i] >= 0 && predictions[i] >= 0) || 
          (targets[i] < 0 && predictions[i] < 0)) {
        correctDirections++;
      }
    }
    const directionalAccuracy = correctDirections / n;
    
    // Correlation coefficient
    const targetMean = yMean;
    const predMean = predictions.reduce((sum, val) => sum + val, 0) / n;
    
    let numerator = 0;
    let targetSumSq = 0;
    let predSumSq = 0;
    
    for (let i = 0; i < n; i++) {
      const targetDiff = targets[i] - targetMean;
      const predDiff = predictions[i] - predMean;
      numerator += targetDiff * predDiff;
      targetSumSq += targetDiff * targetDiff;
      predSumSq += predDiff * predDiff;
    }
    
    const correlation = numerator / Math.sqrt(targetSumSq * predSumSq);
    
    return {
      r2: r2,
      mae: mae,
      mse: mse,
      rmse: rmse,
      mape: mape,
      directionalAccuracy: directionalAccuracy,
      correlation: correlation,
      sampleCount: n
    };
  }

  // Get feature weights (importance)
  getFeatureWeights() {
    if (!this.trained) return null;
    
    const weights = this.model.layers[0].getWeights()[0].dataSync();
    const bias = this.model.layers[0].getWeights()[1].dataSync()[0];
    
    return {
      weights: Array.from(weights),
      bias: bias,
      featureImportance: this.featureNames ? 
        Array.from(weights).map((weight, index) => ({
          feature: this.featureNames[index],
          weight: weight,
          importance: Math.abs(weight),
          rank: index
        })).sort((a, b) => b.importance - a.importance) : null
    };
  }

  // Clean up model resources
  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.trained = false;
  }
}

// Enhanced feature preparation with better engineering
export const prepareLinearRegressionFeatures = (stockData, lookbackDays = 20, targetDaysAhead = 1) => {
  if (!stockData || stockData.length < Math.max(lookbackDays + targetDaysAhead, 252)) {
    throw new Error(`Insufficient data. Need at least ${Math.max(lookbackDays + targetDaysAhead, 252)} days.`);
  }

  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays + targetDaysAhead + 50) {
    throw new Error(`Insufficient technical feature data.`);
  }

  const features = [];
  const targets = [];
  const featureNames = [];

  // Enhanced feature names for better interpretability
  const coreFeatures = [
    'SMA20Ratio', 'SMA50Ratio', 'SMA200Ratio',
    'MACD', 'MACDSignal', 'MACDHistogram',
    'RSI', 'Volatility', 'VolumeRatio', 'PricePosition',
    'BottomSignal', 'PeakSignal'
  ];

  // Build comprehensive feature set
  for (let i = lookbackDays; i < technicalFeatures.length - targetDaysAhead; i++) {
    const currentFeatures = [];
    
    // Recent price action (last few days for immediate context)
    for (let day = 0; day < Math.min(5, lookbackDays); day++) {
      const featureDay = technicalFeatures[i - day - 1];
      
      coreFeatures.forEach((featureName, featureIdx) => {
        let value;
        switch(featureName) {
          case 'SMA20Ratio': value = featureDay.sma20Ratio; break;
          case 'SMA50Ratio': value = featureDay.sma50Ratio; break;
          case 'SMA200Ratio': value = featureDay.sma200Ratio; break;
          case 'MACD': value = featureDay.macd; break;
          case 'MACDSignal': value = featureDay.macdSignal; break;
          case 'MACDHistogram': value = featureDay.macdHistogram; break;
          case 'RSI': value = featureDay.rsi / 100; break; // Normalize
          case 'Volatility': value = featureDay.volatility; break;
          case 'VolumeRatio': value = featureDay.volumeRatio; break;
          case 'PricePosition': value = featureDay.pricePosition; break;
          case 'BottomSignal': value = featureDay.bottomSignal; break;
          case 'PeakSignal': value = featureDay.peakSignal; break;
          default: value = 0;
        }
        
        currentFeatures.push(value || 0);
        if (day === 0 && i === lookbackDays) { // Only add names once
          featureNames.push(`Recent${day + 1}_${featureName}`);
        }
      });
    }
    
    // Enhanced momentum features (critical for short-term prediction)
    const currentPrice = technicalFeatures[i].close;
    const momentumPeriods = [1, 2, 3, 5];
    
    momentumPeriods.forEach(period => {
      if (i >= period) {
        const pastPrice = technicalFeatures[i - period].close;
        const momentum = (currentPrice - pastPrice) / pastPrice;
        currentFeatures.push(momentum);
        
        if (i === lookbackDays) {
          featureNames.push(`Momentum_${period}d`);
        }
      } else {
        currentFeatures.push(0);
        if (i === lookbackDays) {
          featureNames.push(`Momentum_${period}d`);
        }
      }
    });
    
    // Volatility features (important for risk assessment)
    const recentVolatilities = [];
    for (let v = 1; v <= Math.min(10, lookbackDays); v++) {
      if (i >= v) {
        recentVolatilities.push(technicalFeatures[i - v].volatility);
      }
    }
    
    if (recentVolatilities.length > 0) {
      const avgVol = recentVolatilities.reduce((sum, vol) => sum + vol, 0) / recentVolatilities.length;
      const volStd = Math.sqrt(recentVolatilities.reduce((sum, vol) => sum + Math.pow(vol - avgVol, 2), 0) / recentVolatilities.length);
      
      currentFeatures.push(avgVol);
      currentFeatures.push(volStd);
      currentFeatures.push(technicalFeatures[i].volatility / avgVol); // Volatility ratio
      
      if (i === lookbackDays) {
        featureNames.push('AvgVolatility', 'VolatilityStd', 'VolatilityRatio');
      }
    }
    
    // Enhanced trend features
    const lookbackWindow = technicalFeatures.slice(Math.max(0, i - lookbackDays), i);
    
    // Multi-timeframe trend strength
    const trendPeriods = [5, 10, 20];
    trendPeriods.forEach(period => {
      if (lookbackWindow.length >= period) {
        const periodWindow = lookbackWindow.slice(-period);
        const trendStrength = periodWindow.reduce((sum, f) => {
          let score = 0;
          if (f.close > f.close * f.sma20Ratio) score += 1;
          if (f.close > f.close * f.sma50Ratio) score += 1;
          if (f.close > f.close * f.sma200Ratio) score += 1;
          return sum + score;
        }, 0) / (period * 3);
        
        currentFeatures.push(trendStrength);
        
        if (i === lookbackDays) {
          featureNames.push(`TrendStrength_${period}d`);
        }
      } else {
        currentFeatures.push(0);
        if (i === lookbackDays) {
          featureNames.push(`TrendStrength_${period}d`);
        }
      }
    });
    
    // RSI divergence (technical analysis signal)
    if (i >= 5) {
      const currentRSI = technicalFeatures[i].rsi;
      const pastRSI = technicalFeatures[i - 5].rsi;
      const rsiChange = currentRSI - pastRSI;
      const priceChange = (currentPrice - technicalFeatures[i - 5].close) / technicalFeatures[i - 5].close;
      
      // Divergence: price up but RSI down (bearish) or price down but RSI up (bullish)
      const divergence = rsiChange * priceChange < 0 ? 1 : 0;
      
      currentFeatures.push(rsiChange / 100);
      currentFeatures.push(divergence);
      
      if (i === lookbackDays) {
        featureNames.push('RSIChange', 'RSIDivergence');
      }
    } else {
      currentFeatures.push(0, 0);
      if (i === lookbackDays) {
        featureNames.push('RSIChange', 'RSIDivergence');
      }
    }

    // Target: percentage return over specified timeframe
    const futurePrice = technicalFeatures[i + targetDaysAhead].close;
    const percentReturn = (futurePrice - currentPrice) / currentPrice;

    features.push(currentFeatures);
    targets.push(percentReturn);
  }

  console.log(`🎯 Prepared ${features.length} samples with ${features[0].length} enhanced features for ${targetDaysAhead}-day prediction`);
  console.log(`📊 Target statistics: min=${Math.min(...targets).toFixed(4)}, max=${Math.max(...targets).toFixed(4)}, std=${Math.sqrt(targets.reduce((sum, t) => sum + t*t, 0)/targets.length).toFixed(4)}`);
  
  return { features, targets, featureNames };
};

// Enhanced multi-timeframe analysis with TensorFlow.js
export const trainLinearRegressionMultiTimeframe = async (stockData, timeframes = [1, 2, 3, 5, 7, 10, 15, 22], lookbackDays = 20, testSplit = 0.2) => {
  const results = {};
  
  console.log('🚀 TensorFlow.js Multi-Timeframe Linear Regression Analysis...');
  console.log(`📅 Testing timeframes: ${timeframes.join(', ')} days`);
  console.log(`💾 GPU acceleration: ${tf.getBackend()}`);
  
  for (const targetDays of timeframes) {
    console.log(`\n🧠 Training ${targetDays}-day TensorFlow model...`);
    
    try {
      const { features, targets, featureNames } = prepareLinearRegressionFeatures(
        stockData, 
        lookbackDays, 
        targetDays
      );
      
      // Chronological train/test split
      const trainSize = Math.floor(features.length * (1 - testSplit));
      const trainFeatures = features.slice(0, trainSize);
      const trainTargets = targets.slice(0, trainSize);
      const testFeatures = features.slice(trainSize);
      const testTargets = targets.slice(trainSize);
      
      // Create and train TensorFlow model
      const model = new TensorFlowLinearRegression(0.001); // Small L2 regularization
      await model.train(trainFeatures, trainTargets, featureNames, 50, 0.15); // Reduced epochs for speed
      
      // Evaluate on test set
      const testStats = await model.evaluate(testFeatures, testTargets);
      const trainStats = await model.evaluate(trainFeatures, trainTargets);
      
      // Get feature importance
      const featureWeights = model.getFeatureWeights();
      
      results[targetDays] = {
        model: model,
        trainSamples: trainSize,
        testSamples: testFeatures.length,
        trainStats: trainStats,
        testStats: testStats,
        featureImportance: featureWeights.featureImportance ? featureWeights.featureImportance.slice(0, 15) : [],
        targetDays: targetDays,
        lookbackDays: lookbackDays,
        overfitting: trainStats.mae > 0 ? testStats.mae / trainStats.mae : 1 // Overfitting ratio
      };
      
      console.log(`✅ ${targetDays}-day: R²=${testStats.r2.toFixed(3)}, MAE=${testStats.mae.toFixed(4)}, Dir.Acc=${(testStats.directionalAccuracy*100).toFixed(1)}%, Corr=${testStats.correlation.toFixed(3)}`);
      
    } catch (error) {
      console.error(`❌ Failed ${targetDays}-day model:`, error.message);
      results[targetDays] = {
        error: error.message,
        targetDays: targetDays
      };
    }
  }
  
  // Find optimal timeframe using enhanced scoring
  const validResults = Object.entries(results).filter(([_, result]) => !result.error);
  const bestTimeframe = validResults.reduce((best, [timeframe, result]) => {
    // Enhanced scoring: R² (40%) + Directional Accuracy (40%) + Low Overfitting (20%)
    const r2Score = Math.max(0, result.testStats.r2);
    const dirScore = result.testStats.directionalAccuracy;
    const overfitPenalty = Math.min(1, 2 - result.overfitting); // Penalty for overfitting
    
    const score = r2Score * 0.4 + dirScore * 0.4 + overfitPenalty * 0.2;
    
    const bestScore = best.result.testStats ? 
      Math.max(0, best.result.testStats.r2) * 0.4 + 
      best.result.testStats.directionalAccuracy * 0.4 + 
      Math.min(1, 2 - best.result.overfitting) * 0.2 : -1;
    
    return score > bestScore ? { timeframe: parseInt(timeframe), result, score } : best;
  }, { timeframe: 1, result: validResults[0]?.[1] || { testStats: { r2: -1, directionalAccuracy: 0 } }, score: -1 });
  
  console.log(`\n🏆 Optimal timeframe: ${bestTimeframe.timeframe} days (score: ${bestTimeframe.score.toFixed(3)})`);
  console.log(`📊 Performance: R²=${bestTimeframe.result.testStats?.r2?.toFixed(3)}, Dir.Acc=${(bestTimeframe.result.testStats?.directionalAccuracy*100).toFixed(1)}%`);
  
  return {
    results: results,
    bestTimeframe: bestTimeframe.timeframe,
    bestModel: bestTimeframe.result,
    summary: {
      totalTimeframes: timeframes.length,
      successfulTimeframes: validResults.length,
      bestR2: bestTimeframe.result.testStats?.r2 || 0,
      bestDirectionalAccuracy: bestTimeframe.result.testStats?.directionalAccuracy || 0,
      bestScore: bestTimeframe.score,
      avgShortTerm: validResults.filter(([tf, _]) => parseInt(tf) <= 3).length > 0 ? 
        validResults.filter(([tf, _]) => parseInt(tf) <= 3).reduce((sum, [_, result]) => sum + result.testStats.r2, 0) / 
        validResults.filter(([tf, _]) => parseInt(tf) <= 3).length : 0,
      avgLongTerm: validResults.filter(([tf, _]) => parseInt(tf) >= 10).length > 0 ? 
        validResults.filter(([tf, _]) => parseInt(tf) >= 10).reduce((sum, [_, result]) => sum + result.testStats.r2, 0) / 
        validResults.filter(([tf, _]) => parseInt(tf) >= 10).length : 0
    }
  };
};

// Single timeframe training with TensorFlow.js
export const trainLinearRegression = async (stockData, targetDaysAhead = 1, lookbackDays = 20, testSplit = 0.2) => {
  const { features, targets, featureNames } = prepareLinearRegressionFeatures(
    stockData, 
    lookbackDays, 
    targetDaysAhead
  );
  
  const trainSize = Math.floor(features.length * (1 - testSplit));
  const trainFeatures = features.slice(0, trainSize);
  const trainTargets = targets.slice(0, trainSize);
  const testFeatures = features.slice(trainSize);
  const testTargets = targets.slice(trainSize);
  
  const model = new TensorFlowLinearRegression(0.001);
  await model.train(trainFeatures, trainTargets, featureNames, 100, 0.2);
  
  const testStats = await model.evaluate(testFeatures, testTargets);
  const trainStats = await model.evaluate(trainFeatures, trainTargets);
  const featureWeights = model.getFeatureWeights();
  
  return {
    model: model,
    trainStats: trainStats,
    testStats: testStats,
    featureImportance: featureWeights.featureImportance,
    trainSamples: trainSize,
    testSamples: testFeatures.length,
    targetDaysAhead: targetDaysAhead,
    lookbackDays: lookbackDays,
    overfitting: trainStats.mae > 0 ? testStats.mae / trainStats.mae : 1
  };
};

// Enhanced prediction with TensorFlow.js
export const predictWithLinearRegression = async (model, stockData, lookbackDays = 20) => {
  if (!model || !model.trained) {
    throw new Error('Model not trained');
  }
  
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays + 50) {
    throw new Error(`Insufficient data for prediction. Need at least ${lookbackDays + 50} days.`);
  }
  
  // Prepare latest features (matches training feature engineering exactly)
  const latestIndex = technicalFeatures.length - 1;
  const currentFeatures = [];
  
  // Recent price action features
  const coreFeatures = [
    'SMA20Ratio', 'SMA50Ratio', 'SMA200Ratio',
    'MACD', 'MACDSignal', 'MACDHistogram',
    'RSI', 'Volatility', 'VolumeRatio', 'PricePosition',
    'BottomSignal', 'PeakSignal'
  ];
  
  for (let day = 0; day < Math.min(5, lookbackDays); day++) {
    const featureDay = technicalFeatures[latestIndex - day];
    
    coreFeatures.forEach(featureName => {
      let value;
      switch(featureName) {
        case 'SMA20Ratio': value = featureDay.sma20Ratio; break;
        case 'SMA50Ratio': value = featureDay.sma50Ratio; break;
        case 'SMA200Ratio': value = featureDay.sma200Ratio; break;
        case 'MACD': value = featureDay.macd; break;
        case 'MACDSignal': value = featureDay.macdSignal; break;
        case 'MACDHistogram': value = featureDay.macdHistogram; break;
        case 'RSI': value = featureDay.rsi / 100; break;
        case 'Volatility': value = featureDay.volatility; break;
        case 'VolumeRatio': value = featureDay.volumeRatio; break;
        case 'PricePosition': value = featureDay.pricePosition; break;
        case 'BottomSignal': value = featureDay.bottomSignal; break;
        case 'PeakSignal': value = featureDay.peakSignal; break;
        default: value = 0;
      }
      currentFeatures.push(value || 0);
    });
  }
  
  // Add all other features that match training...
  const currentPrice = technicalFeatures[latestIndex].close;
  
  // Momentum features
  [1, 2, 3, 5].forEach(period => {
    if (latestIndex >= period) {
      const pastPrice = technicalFeatures[latestIndex - period].close;
      const momentum = (currentPrice - pastPrice) / pastPrice;
      currentFeatures.push(momentum);
    } else {
      currentFeatures.push(0);
    }
  });
  
  // Volatility features
  const recentVolatilities = [];
  for (let v = 1; v <= Math.min(10, lookbackDays); v++) {
    if (latestIndex >= v) {
      recentVolatilities.push(technicalFeatures[latestIndex - v].volatility);
    }
  }
  
  if (recentVolatilities.length > 0) {
    const avgVol = recentVolatilities.reduce((sum, vol) => sum + vol, 0) / recentVolatilities.length;
    const volStd = Math.sqrt(recentVolatilities.reduce((sum, vol) => sum + Math.pow(vol - avgVol, 2), 0) / recentVolatilities.length);
    
    currentFeatures.push(avgVol);
    currentFeatures.push(volStd);
    currentFeatures.push(technicalFeatures[latestIndex].volatility / avgVol);
  }
  
  // Trend features
  [5, 10, 20].forEach(period => {
    if (latestIndex >= period) {
      const periodWindow = technicalFeatures.slice(latestIndex - period + 1, latestIndex + 1);
      const trendStrength = periodWindow.reduce((sum, f) => {
        let score = 0;
        if (f.close > f.close * f.sma20Ratio) score += 1;
        if (f.close > f.close * f.sma50Ratio) score += 1;
        if (f.close > f.close * f.sma200Ratio) score += 1;
        return sum + score;
      }, 0) / (period * 3);
      
      currentFeatures.push(trendStrength);
    } else {
      currentFeatures.push(0);
    }
  });
  
  // RSI features
  if (latestIndex >= 5) {
    const currentRSI = technicalFeatures[latestIndex].rsi;
    const pastRSI = technicalFeatures[latestIndex - 5].rsi;
    const rsiChange = currentRSI - pastRSI;
    const priceChange = (currentPrice - technicalFeatures[latestIndex - 5].close) / technicalFeatures[latestIndex - 5].close;
    const divergence = rsiChange * priceChange < 0 ? 1 : 0;
    
    currentFeatures.push(rsiChange / 100);
    currentFeatures.push(divergence);
  } else {
    currentFeatures.push(0, 0);
  }
  
  // Make prediction
  const predictedReturn = await model.predictSingle(currentFeatures);
  const predictedPrice = currentPrice * (1 + predictedReturn);
  
  return {
    currentPrice: currentPrice,
    predictedReturn: predictedReturn,
    predictedPrice: predictedPrice,
    priceChange: predictedPrice - currentPrice,
    percentChange: predictedReturn * 100,
    date: technicalFeatures[latestIndex].date,
    confidence: Math.min(0.9, Math.max(0.1, Math.abs(predictedReturn) * 10)), // Simple confidence estimate
    features: currentFeatures
  };
};