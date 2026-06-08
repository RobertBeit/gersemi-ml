// Improved Long-Term Naive Bayes for Price Prediction
// src/services/longTermNaiveBayesService.js

import { extractTechnicalFeatures } from './technicalIndicators';

/**
 * Enhanced feature preparation with better normalization and feature engineering
 */
export const prepareLongTermNBFeatures = (stockData, lookbackDays = 3, targetDaysAhead = 1) => {
  if (!stockData || stockData.length < Math.max(lookbackDays + targetDaysAhead, 252 + targetDaysAhead)) {
    throw new Error(`Insufficient data. Need at least ${252 + targetDaysAhead} days for long-term prediction.`);
  }

  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays + targetDaysAhead) {
    throw new Error(`Insufficient technical feature data for long-term prediction.`);
  }

  const features = [];
  const targets = [];
  const validReturns = []; // Track all returns for better statistics

  for (let i = lookbackDays; i < technicalFeatures.length - targetDaysAhead; i++) {
    const currentFeatures = [];
    
    // Get lookback window
    const lookbackWindow = technicalFeatures.slice(i - lookbackDays, i);
    const currentPoint = technicalFeatures[i];
    
    // ULTRA SIMPLIFIED FEATURES - Only 6 most important ones
    
    // 1. Simple price trend (just current vs recent average)
    const recentPrices = lookbackWindow.map(f => f.close);
    const avgPrice = recentPrices.reduce((sum, p) => sum + p, 0) / recentPrices.length;
    const priceRatio = currentPoint.close / avgPrice - 1; // Normalized price position
    currentFeatures.push(Math.tanh(priceRatio * 10)); // Bounded -1 to 1
    
    // 2. RSI signal (simple)
    const currentRSI = currentPoint.rsi;
    currentFeatures.push((currentRSI - 50) / 50); // Normalized -1 to 1
    
    // 3. Volume signal
    const avgVolume = lookbackWindow.reduce((sum, f) => sum + f.volumeRatio, 0) / lookbackDays;
    currentFeatures.push(Math.tanh(avgVolume - 1)); // Normalized volume
    
    // 4. Moving average trend
    const smaSignal = currentPoint.sma20Ratio - 1;
    currentFeatures.push(Math.tanh(smaSignal * 10));
    
    // 5. Volatility (simple)
    currentFeatures.push(Math.min(currentPoint.volatility * 50, 1)); // Capped volatility
    
    // 6. MACD signal
    currentFeatures.push(Math.tanh(currentPoint.macd * 100));

    // Calculate target return
    const currentPrice = currentPoint.close;
    const futurePrice = technicalFeatures[i + targetDaysAhead].close;
    const percentReturn = (futurePrice - currentPrice) / currentPrice;
    
    // Store all valid returns for statistics
    validReturns.push(percentReturn);
    
    // Less aggressive filtering - allow up to 10% moves
    if (Math.abs(percentReturn) <= 0.10) {
      features.push(currentFeatures);
      targets.push(percentReturn);
      
      if (Math.abs(percentReturn) > 0.03) {
        console.log(`Notable return: ${(percentReturn * 100).toFixed(2)}% on ${currentPoint.date}`);
      }
    }
  }

  // Calculate return statistics
  const sortedReturns = validReturns.slice().sort((a, b) => a - b);
  console.log('Enhanced Return Distribution:');
  console.log(`Samples: ${features.length} (filtered from ${validReturns.length})`);
  console.log(`Min: ${(sortedReturns[0] * 100).toFixed(2)}%`);
  console.log(`5th percentile: ${(sortedReturns[Math.floor(sortedReturns.length * 0.05)] * 100).toFixed(2)}%`);
  console.log(`Median: ${(sortedReturns[Math.floor(sortedReturns.length * 0.5)] * 100).toFixed(2)}%`);
  console.log(`95th percentile: ${(sortedReturns[Math.floor(sortedReturns.length * 0.95)] * 100).toFixed(2)}%`);
  console.log(`Max: ${(sortedReturns[sortedReturns.length - 1] * 100).toFixed(2)}%`);
  console.log(`Feature count: ${features[0]?.length || 0}`);

  return { features, targets };
};

/**
 * Enhanced discretization with outlier handling
 */
export const discretizeLongTermFeatures = (features, targets, bins = 5) => {
  if (!features.length || !targets.length) {
    throw new Error('No features or targets provided for discretization');
  }

  const numFeatures = features[0].length;
  const discretizedFeatures = [];
  const discretizedTargets = [];
  
  // Calculate robust statistics for each feature (using percentiles to handle outliers)
  const featureStats = [];
  for (let f = 0; f < numFeatures; f++) {
    const values = features.map(sample => sample[f]).sort((a, b) => a - b);
    const p5 = values[Math.floor(values.length * 0.05)];
    const p95 = values[Math.floor(values.length * 0.95)];
    
    featureStats.push({
      min: p5,
      max: p95,
      range: p95 - p5
    });
  }
  
  // Calculate robust target statistics
  const sortedTargets = targets.slice().sort((a, b) => a - b);
  const targetP5 = sortedTargets[Math.floor(sortedTargets.length * 0.05)];
  const targetP95 = sortedTargets[Math.floor(sortedTargets.length * 0.95)];
  const targetRange = targetP95 - targetP5;
  
  // Discretize features and targets
  features.forEach((sample, sampleIndex) => {
    const discretizedSample = sample.map((value, featureIndex) => {
      const stats = featureStats[featureIndex];
      if (stats.range === 0) return Math.floor(bins / 2); // Middle bin for constant features
      
      // Clamp to robust range and discretize
      const clampedValue = Math.max(stats.min, Math.min(stats.max, value));
      const normalizedValue = (clampedValue - stats.min) / stats.range;
      const binIndex = Math.min(Math.floor(normalizedValue * bins), bins - 1);
      
      return binIndex;
    });
    
    // Discretize target
    const clampedTarget = Math.max(targetP5, Math.min(targetP95, targets[sampleIndex]));
    const normalizedTarget = targetRange > 0 ? (clampedTarget - targetP5) / targetRange : 0.5;
    const targetBinIndex = Math.min(Math.floor(normalizedTarget * bins), bins - 1);
    
    discretizedFeatures.push(discretizedSample);
    discretizedTargets.push(targetBinIndex);
  });
  
  return {
    discretizedFeatures,
    discretizedTargets,
    metadata: { 
      featureStats,
      targetMin: targetP5,
      targetMax: targetP95,
      targetRange,
      bins,
      originalSamples: features.length
    }
  };
};

/**
 * Enhanced Naive Bayes with better probability handling
 */
export class LongTermNaiveBayesRegressor {
  constructor(numTargetBins = 5, smoothing = 1.0) {
    this.numTargetBins = numTargetBins;
    this.smoothing = smoothing; // Laplace smoothing parameter
    this.targetPriors = Array(numTargetBins).fill(0);
    this.featureCounts = [];
    this.targetCounts = Array(numTargetBins).fill(0);
    this.trained = false;
    this.metadata = null;
  }
  
  train(discretizedFeatures, discretizedTargets, metadata) {
    this.metadata = metadata;
    const numSamples = discretizedFeatures.length;
    const numFeatures = discretizedFeatures[0].length;
    const { bins } = metadata;
    
    // Initialize feature counts
    for (let i = 0; i < numFeatures; i++) {
      this.featureCounts.push(
        Array(this.numTargetBins).fill().map(() => Array(bins).fill(0))
      );
    }
    
    // Count occurrences
    for (let i = 0; i < numSamples; i++) {
      const targetBin = discretizedTargets[i];
      this.targetCounts[targetBin]++;
      
      for (let j = 0; j < numFeatures; j++) {
        const featureBin = discretizedFeatures[i][j];
        this.featureCounts[j][targetBin][featureBin]++;
      }
    }
    
    // Calculate target priors with smoothing
    const totalSamples = numSamples + this.numTargetBins * this.smoothing;
    for (let t = 0; t < this.numTargetBins; t++) {
      this.targetPriors[t] = (this.targetCounts[t] + this.smoothing) / totalSamples;
    }
    
    this.trained = true;
    console.log(`Model trained with ${numSamples} samples, ${numFeatures} features, ${this.numTargetBins} target bins`);
    return this;
  }
  
  predictProba(discretizedFeatures) {
    if (!this.trained) {
      throw new Error("Model not trained yet!");
    }
    
    const probabilities = [];
    
    discretizedFeatures.forEach(sample => {
      const logProbs = Array(this.numTargetBins).fill(0);
      
      for (let t = 0; t < this.numTargetBins; t++) {
        // Start with log of target prior
        logProbs[t] = Math.log(this.targetPriors[t]);
        
        // Add log of conditional probabilities for each feature
        for (let f = 0; f < sample.length; f++) {
          const bin = sample[f];
          
          // Enhanced Laplace smoothing
          const featureCountForTargetAndBin = this.featureCounts[f][t][bin] + this.smoothing;
          const totalFeaturesForTarget = this.targetCounts[t] + this.metadata.bins * this.smoothing;
          
          const probability = featureCountForTargetAndBin / totalFeaturesForTarget;
          logProbs[t] += Math.log(probability);
        }
      }
      
      // Convert from log probabilities to probabilities
      const maxLogProb = Math.max(...logProbs);
      const expProbs = logProbs.map(lp => Math.exp(lp - maxLogProb));
      const sumExpProbs = expProbs.reduce((a, b) => a + b, 0);
      
      // Normalize to get probabilities
      const normalizedProbs = expProbs.map(p => p / (sumExpProbs || 1));
      probabilities.push(normalizedProbs);
    });
    
    return probabilities;
  }
  
  predict(discretizedFeatures) {
    const probabilities = this.predictProba(discretizedFeatures);
    const { targetMin, targetRange, bins } = this.metadata;
    
    return probabilities.map(probs => {
      let weightedSum = 0;
      
      for (let t = 0; t < this.numTargetBins; t++) {
        // Calculate bin center
        const binCenter = targetMin + (t + 0.5) * targetRange / bins;
        weightedSum += probs[t] * binCenter;
      }
      
      return weightedSum;
    });
  }
  
  predictWithConfidence(discretizedFeatures) {
    const probabilities = this.predictProba(discretizedFeatures);
    const { targetMin, targetRange, bins } = this.metadata;
    
    return probabilities.map(probs => {
      let weightedSum = 0;
      let weightedSquareSum = 0;
      
      for (let t = 0; t < this.numTargetBins; t++) {
        const binCenter = targetMin + (t + 0.5) * targetRange / bins;
        weightedSum += probs[t] * binCenter;
        weightedSquareSum += probs[t] * binCenter * binCenter;
      }
      
      const variance = weightedSquareSum - (weightedSum * weightedSum);
      const stdDev = Math.sqrt(Math.max(variance, 0));
      
      // Calculate confidence based on probability concentration
      const maxProb = Math.max(...probs);
      const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
      const maxEntropy = Math.log(this.numTargetBins);
      const confidence = Math.min(maxProb * 2, 1 - (entropy / maxEntropy));
      
      return {
        prediction: weightedSum,
        confidence: confidence,
        stdDev: stdDev,
        confidenceInterval: [weightedSum - 1.96 * stdDev, weightedSum + 1.96 * stdDev],
        probabilities: probs
      };
    });
  }
  
  evaluate(discretizedFeatures, trueContinuousTargets) {
    const predictions = this.predict(discretizedFeatures);
    const n = predictions.length;
    
    if (n === 0) return { mae: 0, mse: 0, rmse: 0, r2: 0, mape: 0 };
    
    const meanTrue = trueContinuousTargets.reduce((sum, val) => sum + val, 0) / n;
    
    // Mean Absolute Error
    const mae = predictions.reduce((sum, pred, i) => sum + Math.abs(pred - trueContinuousTargets[i]), 0) / n;
    
    // Mean Squared Error
    const mse = predictions.reduce((sum, pred, i) => sum + Math.pow(pred - trueContinuousTargets[i], 2), 0) / n;
    
    // Root Mean Squared Error
    const rmse = Math.sqrt(mse);
    
    // R-squared
    const ssTot = trueContinuousTargets.reduce((sum, val) => sum + Math.pow(val - meanTrue, 2), 0);
    const ssRes = predictions.reduce((sum, pred, i) => sum + Math.pow(trueContinuousTargets[i] - pred, 2), 0);
    const r2 = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    
    // Mean Absolute Percentage Error (handle division by zero)
    const mape = predictions.reduce((sum, pred, i) => {
      const actual = Math.abs(trueContinuousTargets[i]);
      if (actual > 0.001) {
        return sum + Math.abs((trueContinuousTargets[i] - pred) / trueContinuousTargets[i]);
      }
      return sum;
    }, 0) / n * 100;
    
    // Directional accuracy
    let correctDirection = 0;
    for (let i = 0; i < n; i++) {
      const actualDirection = trueContinuousTargets[i] > 0 ? 1 : -1;
      const predictedDirection = predictions[i] > 0 ? 1 : -1;
      if (actualDirection === predictedDirection) correctDirection++;
    }
    const directionalAccuracy = correctDirection / n;
    
    return {
      mae: mae * 100, // Convert to percentage points
      mse,
      rmse: rmse * 100,
      r2,
      mape,
      directionalAccuracy,
      meanPrediction: (predictions.reduce((sum, val) => sum + val, 0) / n) * 100,
      meanActual: meanTrue * 100,
      sampleCount: n
    };
  }
}

/**
 * Discretize prediction sample using training metadata
 */
export const discretizeLongTermPredictionSample = (sample, metadata) => {
  const { featureStats, bins } = metadata;
  
  return sample.map((value, featureIndex) => {
    const stats = featureStats[featureIndex];
    if (stats.range === 0) return Math.floor(bins / 2);
    
    const clampedValue = Math.max(stats.min, Math.min(stats.max, value));
    const normalizedValue = (clampedValue - stats.min) / stats.range;
    const binIndex = Math.min(Math.floor(normalizedValue * bins), bins - 1);
    
    return Math.max(0, binIndex);
  });
};

/**
 * Enhanced latest features preparation
 */
export const prepareLatestLongTermNBFeatures = (stockData, lookbackDays = 3) => {
  if (!stockData || stockData.length < Math.max(lookbackDays, 252)) {
    throw new Error(`Insufficient data for prediction. Need at least 252 days.`);
  }
  
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays) {
    throw new Error(`Insufficient technical feature data for prediction.`);
  }
  
  // Use the same simplified feature engineering as in training
  const lookbackWindow = technicalFeatures.slice(technicalFeatures.length - lookbackDays);
  const currentPoint = technicalFeatures[technicalFeatures.length - 1];
  const features = [];
  
  // SAME 6 FEATURES AS TRAINING
  
  // 1. Price trend
  const recentPrices = lookbackWindow.map(f => f.close);
  const avgPrice = recentPrices.reduce((sum, p) => sum + p, 0) / recentPrices.length;
  const priceRatio = currentPoint.close / avgPrice - 1;
  features.push(Math.tanh(priceRatio * 10));
  
  // 2. RSI signal
  features.push((currentPoint.rsi - 50) / 50);
  
  // 3. Volume signal
  const avgVolume = lookbackWindow.reduce((sum, f) => sum + f.volumeRatio, 0) / lookbackDays;
  features.push(Math.tanh(avgVolume - 1));
  
  // 4. Moving average trend
  features.push(Math.tanh((currentPoint.sma20Ratio - 1) * 10));
  
  // 5. Volatility
  features.push(Math.min(currentPoint.volatility * 50, 1));
  
  // 6. MACD signal
  features.push(Math.tanh(currentPoint.macd * 100));
  
  return features;
};

/**
 * Main training function with enhanced parameters
 */
export const trainLongTermNaiveBayes = (stockData, lookbackDays = 3, targetDaysAhead = 1, bins = 3, testSplit = 0.2) => {
  console.log('🎯 Starting Enhanced Long-Term Naive Bayes Training...');
  
  const { features, targets } = prepareLongTermNBFeatures(stockData, lookbackDays, targetDaysAhead);
  
  if (features.length === 0) {
    throw new Error('No valid features generated. Check your data and parameters.');
  }
  
  console.log(`Prepared ${features.length} samples with ${features[0].length} features for ${targetDaysAhead}-day prediction`);
  
  const { discretizedFeatures, discretizedTargets, metadata } = discretizeLongTermFeatures(features, targets, bins);
  
  const totalSamples = discretizedFeatures.length;
  const testSize = Math.floor(totalSamples * testSplit);
  const trainSize = totalSamples - testSize;
  
  const trainFeatures = discretizedFeatures.slice(0, trainSize);
  const trainTargets = discretizedTargets.slice(0, trainSize);
  const testFeatures = discretizedFeatures.slice(trainSize);
  const testTargets = discretizedTargets.slice(trainSize);
  const testContinuousTargets = targets.slice(trainSize);
  
  console.log(`Training set: ${trainSize} samples, Test set: ${testSize} samples`);
  
  const model = new LongTermNaiveBayesRegressor(bins, 1.0);
  model.train(trainFeatures, trainTargets, metadata);
  
  const metrics = model.evaluate(testFeatures, testContinuousTargets);
  
  console.log('✅ Enhanced Naive Bayes training completed');
  console.log(`Performance - R²: ${metrics.r2.toFixed(3)}, RMSE: ${metrics.rmse.toFixed(2)}%, Directional: ${(metrics.directionalAccuracy * 100).toFixed(1)}%`);
  
  return {
    model,
    metadata,
    metrics,
    lookbackDays,
    targetDaysAhead,
    bins,
    trainingSamples: trainSize,
    testSamples: testSize
  };
};

export const predictWithLongTermNaiveBayes = (
  model,
  metadata,
  stockData,
  lookbackDays = 3,
  targetDaysAhead = 1
) => {
  if (!model || typeof model.predictWithConfidence !== 'function') {
    throw new Error('Long-term Naive Bayes model is not available for prediction');
  }

  if (!metadata) {
    throw new Error('Long-term Naive Bayes metadata is required for prediction');
  }

  const latestFeatures = prepareLatestLongTermNBFeatures(stockData, lookbackDays);
  const discretized = discretizeLongTermPredictionSample(latestFeatures, metadata);
  const predictionResult = model.predictWithConfidence([discretized])?.[0];

  if (!predictionResult) {
    throw new Error('Long-term Naive Bayes did not return a prediction result');
  }

  const currentPrice = Number(stockData?.[stockData.length - 1]?.close ?? 0);
  if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
    throw new Error('Unable to resolve current price for long-term Naive Bayes prediction');
  }

  const expectedReturn = Number(predictionResult.prediction ?? 0);
  const predictedPrice = currentPrice * (1 + expectedReturn);

  return {
    currentPrice,
    predictedPrice,
    predictedReturn: expectedReturn,
    percentChange: expectedReturn * 100,
    priceChange: predictedPrice - currentPrice,
    direction: expectedReturn > 0 ? 'UP' : (expectedReturn < 0 ? 'DOWN' : 'FLAT'),
    confidence: Number(predictionResult.confidence ?? 0),
    stdDev: Number(predictionResult.stdDev ?? 0),
    confidenceInterval: predictionResult.confidenceInterval || null,
    probabilities: predictionResult.probabilities || null,
    horizonDays: Math.max(1, Number(targetDaysAhead) || 1),
  };
};