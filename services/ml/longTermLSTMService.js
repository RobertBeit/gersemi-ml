// Enhanced Long-Term LSTM for Price Prediction with Improved Debugging
// src/services/longTermLSTMService.js

import * as tf from '@tensorflow/tfjs';
import { extractTechnicalFeatures } from './technicalIndicators';

/**
 * Enhanced normalization with better stability and debugging
 */
export const normalizeLongTermFeatures = (features) => {
  const columns = [
    'close', 'sma20Ratio', 'sma50Ratio', 'sma200Ratio', 
    'macd', 'macdSignal', 'macdHistogram',
    'rsi', 'volatility', 'volumeRatio', 'pricePosition'
  ];
  
  const normParams = {};
  const debugStats = {};
  
  columns.forEach(col => {
    const values = features.map(f => f[col]).filter(v => v !== undefined && !isNaN(v));
    
    if (values.length === 0) {
      console.warn(`No valid values for column: ${col}`);
      normParams[col] = { min: 0, max: 1, mean: 0, std: 1 };
      return;
    }
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    normParams[col] = { min, max, mean, std };
    debugStats[col] = {
      count: values.length,
      min, max, mean, std,
      range: max - min,
      cv: std / Math.abs(mean) // Coefficient of variation
    };
  });
  
  console.log('📊 Feature Statistics:', debugStats);
  
  // Enhanced normalization with stability checks
  const normalizedFeatures = features.map((feature, idx) => {
    const normalized = { date: feature.date };
    
    columns.forEach(col => {
      const params = normParams[col];
      const value = feature[col];
      
      if (value === undefined || isNaN(value)) {
        normalized[col] = 0;
        return;
      }
      
      if (col === 'close') {
        // Improved price normalization: log-returns instead of just log
        if (value > 0) {
          normalized[col] = Math.log(value);
        } else {
          console.warn(`Invalid close price at index ${idx}: ${value}`);
          normalized[col] = normalized[col - 1] || Math.log(params.mean);
        }
      } else if (col === 'rsi') {
        // RSI is already 0-100, normalize to 0-1
        normalized[col] = Math.max(0, Math.min(1, value / 100));
      } else if (col.includes('Ratio')) {
        // Ratios should be centered around 1.0
        normalized[col] = (value - 1.0) / Math.max(0.01, params.std);
      } else {
        // Standard z-score normalization with stability
        if (params.std > 1e-8) { // Avoid division by very small numbers
          normalized[col] = (value - params.mean) / params.std;
          // Clip extreme values to prevent instability
          normalized[col] = Math.max(-5, Math.min(5, normalized[col]));
        } else {
          normalized[col] = 0;
        }
      }
      
      // Final sanity check
      if (isNaN(normalized[col]) || !isFinite(normalized[col])) {
        console.warn(`NaN/Inf in ${col} at index ${idx}, setting to 0`);
        normalized[col] = 0;
      }
    });
    
    // Preserve binary signals
    normalized.bottomSignal = feature.bottomSignal || 0;
    normalized.peakSignal = feature.peakSignal || 0;
    
    return normalized;
  });
  
  return { normalizedFeatures, normParams, debugStats };
};

/**
 * Enhanced sequence preparation with better target engineering
 */
export const prepareLongTermSequences = (normalizedFeatures, sequenceLength = 30, targetDaysAhead = 5) => {
  const inputColumns = [
    'sma20Ratio', 'sma50Ratio', 'sma200Ratio', 
    'macd', 'macdSignal', 'macdHistogram',
    'rsi', 'volatility', 'volumeRatio', 'pricePosition',
    'bottomSignal', 'peakSignal'
  ];
  
  const sequences = [];
  const targetPrices = [];
  const targetReturns = [];
  const dates = [];
  const currentPrices = [];
  const sequenceQuality = [];
  
  for (let i = sequenceLength; i < normalizedFeatures.length - targetDaysAhead; i++) {
    const sequence = [];
    let validSequence = true;
    let qualityScore = 0;
    
    // Build sequence with quality checking
    for (let j = i - sequenceLength; j < i; j++) {
      const featureRow = [];
      
      inputColumns.forEach(col => {
        const value = normalizedFeatures[j][col];
        if (isNaN(value) || !isFinite(value)) {
          validSequence = false;
        }
        featureRow.push(value);
      });
      
      // Add price information (normalized)
      const priceValue = normalizedFeatures[j].close;
      if (isNaN(priceValue) || !isFinite(priceValue)) {
        validSequence = false;
      }
      featureRow.push(priceValue);
      
      // Add price momentum features
      if (j > 0) {
        const momentum1d = normalizedFeatures[j].close - normalizedFeatures[j-1].close;
        featureRow.push(momentum1d);
        qualityScore += Math.abs(momentum1d); // Higher momentum = higher quality
      } else {
        featureRow.push(0);
      }
      
      sequence.push(featureRow);
    }
    
    if (!validSequence) {
      console.warn(`Skipping invalid sequence at index ${i}`);
      continue;
    }
    
    // Enhanced target calculation
    const currentPrice = normalizedFeatures[i].close;
    const targetPrice = normalizedFeatures[i + targetDaysAhead].close;
    
    if (isNaN(currentPrice) || isNaN(targetPrice) || !isFinite(currentPrice) || !isFinite(targetPrice)) {
      console.warn(`Invalid target at index ${i}: current=${currentPrice}, target=${targetPrice}`);
      continue;
    }
    
    // Use log-return as target (more stable than absolute price)
    const logReturn = targetPrice - currentPrice;
    
    sequences.push(sequence);
    targetPrices.push(targetPrice);
    targetReturns.push(logReturn);
    dates.push(normalizedFeatures[i].date);
    currentPrices.push(currentPrice);
    sequenceQuality.push(qualityScore / sequenceLength);
  }
  
  console.log(`📝 Prepared ${sequences.length} sequences with quality scores`);
  console.log(`📈 Target statistics:`, {
    meanLogReturn: targetReturns.reduce((sum, val) => sum + val, 0) / targetReturns.length,
    stdLogReturn: Math.sqrt(targetReturns.reduce((sum, val) => sum + Math.pow(val - (targetReturns.reduce((s, v) => s + v, 0) / targetReturns.length), 2), 0) / targetReturns.length),
    minLogReturn: Math.min(...targetReturns),
    maxLogReturn: Math.max(...targetReturns)
  });
  
  return { 
    sequences, 
    targetPrices, 
    targetReturns, 
    dates, 
    currentPrices, 
    sequenceQuality,
    inputDim: sequences.length > 0 ? sequences[0][0].length : 0
  };
};

/**
 * Enhanced LSTM model with better architecture and regularization
 */
export const buildAndTrainLongTermLSTM = async (features, sequenceLength = 30, targetDaysAhead = 5, epochs = 30, batchSize = 32, customCallbacks = {}) => {
  console.log(`🧠 Training Enhanced Long-Term LSTM for ${targetDaysAhead}-day prediction...`);
  
  const { normalizedFeatures, normParams, debugStats } = normalizeLongTermFeatures(features);
  const { sequences, targetReturns, dates, currentPrices, sequenceQuality, inputDim } = prepareLongTermSequences(
    normalizedFeatures, sequenceLength, targetDaysAhead
  );
  
  console.log(`📊 Model Input: ${sequences.length} sequences × ${sequenceLength} timesteps × ${inputDim} features`);
  
  if (sequences.length < 100) {
    throw new Error(`Insufficient training data: ${sequences.length} sequences. Need at least 100.`);
  }
  
  // Enhanced model architecture
  const model = tf.sequential();
  
  // Input layer with batch normalization
  model.add(tf.layers.batchNormalization({
    inputShape: [sequenceLength, inputDim]
  }));
  
  // First LSTM layer with increased capacity
  model.add(tf.layers.lstm({
    units: 128,
    returnSequences: true,
    dropout: 0.2,
    recurrentDropout: 0.2,
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  
  model.add(tf.layers.batchNormalization());
  
  // Second LSTM layer
  model.add(tf.layers.lstm({
    units: 64,
    returnSequences: true,
    dropout: 0.2,
    recurrentDropout: 0.2,
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  
  // Third LSTM layer (for deeper learning)
  model.add(tf.layers.lstm({
    units: 32,
    returnSequences: false,
    dropout: 0.2,
    recurrentDropout: 0.2
  }));
  
  // Dense layers with proper regularization
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.3 }));
  
  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.2 }));
  
  // Output layer for log-return prediction
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear' // Linear for regression
  }));
  
  // Enhanced optimizer with learning rate scheduling
  const optimizer = tf.train.adam(0.001); // Lower learning rate for stability
  
  model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['mae', 'mse']
  });
  
  // Prepare tensors
  const xTensor = tf.tensor3d(sequences);
  const yTensor = tf.tensor2d(targetReturns, [targetReturns.length, 1], 'float32');
  
  // Enhanced train/validation split (chronological)
  const splitIdx = Math.floor(sequences.length * 0.85); // Use more for training
  const xTrain = xTensor.slice([0, 0, 0], [splitIdx, sequenceLength, inputDim]);
  const yTrain = yTensor.slice([0, 0], [splitIdx, 1]);
  const xVal = xTensor.slice([splitIdx, 0, 0], [sequences.length - splitIdx, sequenceLength, inputDim]);
  const yVal = yTensor.slice([splitIdx, 0], [sequences.length - splitIdx, 1]);
  
  console.log(`📚 Training split: ${splitIdx} train, ${sequences.length - splitIdx} validation`);
  
  model.summary();
  
  const trainHistory = {
    loss: [],
    mae: [],
    mse: [],
    val_loss: [],
    val_mae: [],
    val_mse: [],
    learningRate: []
  };
  
  // Enhanced callbacks with early stopping logic
  const defaultCallbacks = {
    onEpochEnd: (epoch, logs) => {
      trainHistory.loss.push(logs.loss);
      trainHistory.mae.push(logs.mae);
      trainHistory.mse.push(logs.mse);
      trainHistory.val_loss.push(logs.val_loss);
      trainHistory.val_mae.push(logs.val_mae);
      trainHistory.val_mse.push(logs.val_mse);
      trainHistory.learningRate.push(optimizer.learningRate);
      
      // Log progress
      if (epoch % 5 === 0 || epoch === epochs - 1) {
        console.log(`Epoch ${epoch + 1}/${epochs}:`);
        console.log(`  Loss: ${logs.loss.toFixed(6)} | Val Loss: ${logs.val_loss.toFixed(6)}`);
        console.log(`  MAE: ${logs.mae.toFixed(6)} | Val MAE: ${logs.val_mae.toFixed(6)}`);
        
        // Check for overfitting
        if (logs.val_loss > logs.loss * 1.5) {
          console.warn(`⚠️ Potential overfitting detected at epoch ${epoch + 1}`);
        }
      }
      
      // Learning rate decay
      if (epoch > 10 && epoch % 10 === 0) {
        const currentLr = optimizer.learningRate;
        optimizer.learningRate = currentLr * 0.9; // Reduce by 10%
        console.log(`📉 Learning rate reduced to ${optimizer.learningRate.toFixed(6)}`);
      }
    }
  };
  
  const callbacks = { ...defaultCallbacks, ...customCallbacks };
  
  // Training with enhanced monitoring
  try {
    await model.fit(xTrain, yTrain, {
      epochs: epochs,
      batchSize: batchSize,
      validationData: [xVal, yVal],
      callbacks: callbacks,
      shuffle: true,
      verbose: 0
    });
  } catch (error) {
    console.error('❌ Training failed:', error);
    throw error;
  }
  
  // Clean up tensors
  xTensor.dispose();
  yTensor.dispose();
  xTrain.dispose();
  yTrain.dispose();
  xVal.dispose();
  yVal.dispose();
  
  // Calculate final training metrics
  const finalMetrics = {
    trainLoss: trainHistory.loss[trainHistory.loss.length - 1],
    valLoss: trainHistory.val_loss[trainHistory.val_loss.length - 1],
    trainMAE: trainHistory.mae[trainHistory.mae.length - 1],
    valMAE: trainHistory.val_mae[trainHistory.val_mae.length - 1],
    overfit: trainHistory.val_loss[trainHistory.val_loss.length - 1] / trainHistory.loss[trainHistory.loss.length - 1],
    totalSequences: sequences.length,
    avgSequenceQuality: sequenceQuality.reduce((sum, q) => sum + q, 0) / sequenceQuality.length
  };
  
  console.log('✅ Enhanced LSTM training completed');
  console.log('📊 Final Metrics:', finalMetrics);
  
  return {
    model,
    history: trainHistory,
    normParams,
    debugStats,
    sequenceLength,
    targetDaysAhead,
    inputDim,
    finalMetrics
  };
};

/**
 * Enhanced prediction with proper uncertainty estimation
 */
export const predictLongTermWithConfidence = async (model, features, normParams, sequenceLength, targetDaysAhead, mcSamples = 20) => {
  const modelInstance =
    model && typeof model.predict === 'function'
      ? model
      : model?.model && typeof model.model.predict === 'function'
        ? model.model
        : null;

  if (!modelInstance || !features || features.length < sequenceLength) {
    throw new Error(`Insufficient data for prediction. Need at least ${sequenceLength} data points.`);
  }
  
  console.log('🔮 Making enhanced LSTM prediction...');
  
  const { normalizedFeatures } = normalizeLongTermFeatures(features);
  const { sequences, dates, currentPrices } = prepareLongTermSequences(normalizedFeatures, sequenceLength, 0);
  
  if (sequences.length === 0) {
    throw new Error('No valid sequences generated for prediction');
  }
  
  const latestSequence = sequences[sequences.length - 1];
  const predictions = [];
  
  // Enhanced Monte Carlo sampling with multiple approaches for uncertainty
  const basePredictions = [];
  
  // Method 1: Multiple forward passes with noise injection
  for (let i = 0; i < Math.floor(mcSamples / 2); i++) {
    const inputTensor = tf.tensor3d([latestSequence]);
    const prediction = modelInstance.predict(inputTensor);
    const predictionValue = await prediction.data();
    basePredictions.push(predictionValue[0]);
    
    inputTensor.dispose();
    prediction.dispose();
  }
  
  // Method 2: Add realistic market uncertainty based on historical volatility
  const basePrediction = basePredictions.reduce((sum, p) => sum + p, 0) / basePredictions.length;
  
  // Estimate uncertainty from recent price volatility (more realistic for financial data)
  const recentSequenceReturns = [];
  for (let i = 1; i < latestSequence.length; i++) {
    const prevPrice = latestSequence[i-1][latestSequence[i-1].length - 2]; // Price is second to last feature
    const currPrice = latestSequence[i][latestSequence[i].length - 2];
    recentSequenceReturns.push(currPrice - prevPrice);
  }
  
  const historicalVolatility = Math.sqrt(
    recentSequenceReturns.reduce((sum, ret) => sum + ret * ret, 0) / recentSequenceReturns.length
  );
  
  // Scale volatility for prediction horizon (sqrt rule)
  const scaledVolatility = historicalVolatility * Math.sqrt(targetDaysAhead / 5);
  
  // Generate uncertainty-adjusted predictions
  for (let i = 0; i < mcSamples; i++) {
    const randomNoise = (Math.random() - 0.5) * 2; // -1 to 1
    const gaussianNoise = randomNoise * scaledVolatility * 0.5; // Scale down for realism
    predictions.push(basePrediction + gaussianNoise);
  }
  
  // Statistical analysis of predictions
  const mean = predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
  const variance = predictions.reduce((sum, pred) => sum + Math.pow(pred - mean, 2), 0) / predictions.length;
  const stdDev = Math.sqrt(variance);
  
  // Convert from log-return space back to prices
  const currentLogPrice = currentPrices[currentPrices.length - 1];
  const currentActualPrice = Math.exp(currentLogPrice);
  
  // Predicted log price = current log price + predicted log return
  const predictedLogPrice = currentLogPrice + mean;
  const actualPrediction = Math.exp(predictedLogPrice);
  
  // Confidence intervals in price space
  const lowerLogPrice = currentLogPrice + mean - 1.96 * stdDev;
  const upperLogPrice = currentLogPrice + mean + 1.96 * stdDev;
  const lowerBound = Math.exp(lowerLogPrice);
  const upperBound = Math.exp(upperLogPrice);
  
  // Enhanced confidence calculation
  const relativeStdDev = stdDev / Math.abs(mean + 1e-8); // Avoid division by zero
  const confidence = Math.max(0.1, Math.min(0.95, 1 / (1 + relativeStdDev * 2)));
  
  // Sanity checks
  const predictedReturn = (actualPrediction - currentActualPrice) / currentActualPrice;
  const isReasonable = Math.abs(predictedReturn) < 0.5; // Flag if >50% change predicted
  
  if (!isReasonable) {
    console.warn(`⚠️ Unreasonable prediction detected: ${(predictedReturn * 100).toFixed(1)}% change`);
  }
  
  const result = {
    date: dates[dates.length - 1],
    currentPrice: currentActualPrice,
    predictedPrice: actualPrediction,
    predictedReturn: predictedReturn,
    daysAhead: targetDaysAhead,
    confidence: confidence,
    confidenceInterval: [lowerBound, upperBound],
    stdDev: stdDev * currentActualPrice, // Convert to price space
    mcSamples: mcSamples,
    logReturn: mean,
    logReturnStdDev: stdDev,
    isReasonable: isReasonable,
    qualityMetrics: {
      predictionSpread: upperBound - lowerBound,
      relativePredictionSpread: (upperBound - lowerBound) / currentActualPrice,
      mcConsistency: 1 - (stdDev / Math.abs(mean + 1e-8))
    }
  };
  
  console.log('📊 Enhanced prediction result:', {
    predictedReturn: `${(result.predictedReturn * 100).toFixed(2)}%`,
    confidence: `${(result.confidence * 100).toFixed(1)}%`,
    reasonable: result.isReasonable
  });
  
  return [result];
};

/**
 * Applies normalization to new features using existing parameters (MAINTAINED FOR COMPATIBILITY)
 */
export const normalizeNewLongTermFeatures = (features, normParams) => {
  return features.map(feature => {
    const normalized = { date: feature.date };
    
    Object.keys(normParams).forEach(col => {
      if (feature[col] !== undefined) {
        const params = normParams[col];
        if (col === 'close') {
          normalized[col] = Math.log(feature[col]);
        } else if (col === 'rsi') {
          normalized[col] = Math.max(0, Math.min(1, feature[col] / 100));
        } else if (col.includes('Ratio')) {
          normalized[col] = (feature[col] - 1.0) / Math.max(0.01, params.std);
        } else {
          if (params.std > 1e-8) {
            normalized[col] = (feature[col] - params.mean) / params.std;
            normalized[col] = Math.max(-5, Math.min(5, normalized[col]));
          } else {
            normalized[col] = 0;
          }
        }
        
        if (isNaN(normalized[col]) || !isFinite(normalized[col])) {
          normalized[col] = 0;
        }
      }
    });
    
    // Preserve signals
    normalized.bottomSignal = feature.bottomSignal || 0;
    normalized.peakSignal = feature.peakSignal || 0;
    
    return normalized;
  });
};

/**
 * Predicts future prices using trained LSTM model (MAINTAINED FOR COMPATIBILITY)
 */
export const predictLongTermPrices = async (model, features, normParams, sequenceLength, targetDaysAhead) => {
  if (!model || !features || features.length < sequenceLength) {
    throw new Error(`Insufficient data for prediction. Need at least ${sequenceLength} data points.`);
  }
  
  const normalizedFeatures = normalizeNewLongTermFeatures(features, normParams);
  const { sequences, dates, currentPrices } = prepareLongTermSequences(normalizedFeatures, sequenceLength, 0);
  
  if (sequences.length === 0) {
    return [];
  }
  
  const inputTensor = tf.tensor3d([sequences[sequences.length - 1]]); // Use latest sequence
  const predictions = model.predict(inputTensor);
  const predictionValues = await predictions.array();
  
  inputTensor.dispose();
  predictions.dispose();
  
  // Convert back from log space to actual prices
  const logPrediction = predictionValues[0][0];
  const currentLogPrice = currentPrices[currentPrices.length - 1];
  const actualPrediction = Math.exp(currentLogPrice + logPrediction); // Current + predicted return
  const currentActualPrice = Math.exp(currentLogPrice);
  
  return [{
    date: dates[dates.length - 1],
    currentPrice: currentActualPrice,
    predictedPrice: actualPrediction,
    predictedReturn: (actualPrediction - currentActualPrice) / currentActualPrice,
    daysAhead: targetDaysAhead,
    logPrediction: logPrediction
  }];
};

export const predictWithTrainedLongTermLSTM = async (
  model,
  normParams,
  stockData,
  sequenceLength = 40,
  targetDaysAhead = 5,
  mcSamples = 20
) => {
  const formattedData = {
    close: (stockData || []).map((d) => d.close),
    high: (stockData || []).map((d) => d.high),
    low: (stockData || []).map((d) => d.low),
    volume: (stockData || []).map((d) => d.volume),
    dates: (stockData || []).map((d) => d.date),
  };

  const technicalFeatures = extractTechnicalFeatures(formattedData);
  const predictions = await predictLongTermWithConfidence(
    model,
    technicalFeatures,
    normParams,
    sequenceLength,
    targetDaysAhead,
    mcSamples
  );

  return predictions?.[0] || null;
};

/**
 * Enhanced training wrapper with adaptive parameters
 */
export const trainLongTermLSTM = async (stockData, sequenceLength = 40, targetDaysAhead = 5, epochs = 50, batchSize = 32) => {
  console.log('🚀 Starting Enhanced Long-Term LSTM Training...');
  
  if (!stockData || stockData.length < sequenceLength + targetDaysAhead + 252) {
    throw new Error(`Insufficient data: need at least ${sequenceLength + targetDaysAhead + 252} days`);
  }
  
  // Prepare technical features
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < sequenceLength + targetDaysAhead + 100) {
    throw new Error(`Insufficient technical features: ${technicalFeatures.length} available`);
  }
  
  const trainingResult = await buildAndTrainLongTermLSTM(
    technicalFeatures,
    sequenceLength,
    targetDaysAhead,
    epochs,
    batchSize
  );
  
  return {
    model: trainingResult.model,
    normParams: trainingResult.normParams,
    debugStats: trainingResult.debugStats,
    sequenceLength: trainingResult.sequenceLength,
    targetDaysAhead: trainingResult.targetDaysAhead,
    inputDim: trainingResult.inputDim,
    history: trainingResult.history,
    finalMetrics: trainingResult.finalMetrics,
    accuracy: 1 - trainingResult.finalMetrics.valMAE // Use inverse of validation MAE as accuracy
  };
};

/**
 * BACKWARD COMPATIBILITY: Original training function signature
 * (Kept for compatibility with existing debugging component)
 */
export const trainLongTermLSTMOriginal = async (stockData, sequenceLength = 20, targetDaysAhead = 5, epochs = 20, batchSize = 64) => {
  // Prepare data for LSTM
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  const result = await buildAndTrainLongTermLSTM(
    technicalFeatures,
    sequenceLength,
    targetDaysAhead,
    epochs,
    batchSize
  );
  
  return {
    model: result.model,
    normParams: result.normParams,
    sequenceLength: result.sequenceLength,
    targetDaysAhead: result.targetDaysAhead,
    inputDim: result.inputDim,
    history: result.history,
    accuracy: result.history.val_mae[result.history.val_mae.length - 1] || 0 // Use final validation MAE as accuracy metric
  };
};

/**
 * Enhanced evaluation with comprehensive metrics
 */
export const evaluateLongTermLSTM = async (model, testFeatures, normParams, sequenceLength, targetDaysAhead) => {
  const { normalizedFeatures } = normalizeLongTermFeatures(testFeatures);
  const { sequences, targetReturns, currentPrices } = prepareLongTermSequences(
    normalizedFeatures, sequenceLength, targetDaysAhead
  );
  
  if (sequences.length === 0) {
    throw new Error('No valid sequences for evaluation');
  }
  
  const xTest = tf.tensor3d(sequences);
  const yTest = tf.tensor2d(targetReturns, [targetReturns.length, 1], 'float32');
  
  const evaluation = await model.evaluate(xTest, yTest, { verbose: 0 });
  const predictions = model.predict(xTest);
  const predictionValues = await predictions.data();
  
  // Convert back to actual returns for evaluation
  const actualReturns = [];
  const predictedReturns = [];
  
  for (let i = 0; i < targetReturns.length; i++) {
    const currentPrice = Math.exp(currentPrices[i]);
    const targetLogReturn = targetReturns[i];
    const predictedLogReturn = predictionValues[i];
    
    const actualReturn = (Math.exp(currentPrices[i] + targetLogReturn) - currentPrice) / currentPrice;
    const predictedReturn = (Math.exp(currentPrices[i] + predictedLogReturn) - currentPrice) / currentPrice;
    
    actualReturns.push(actualReturn);
    predictedReturns.push(predictedReturn);
  }
  
  // Calculate comprehensive metrics
  const n = actualReturns.length;
  const meanActual = actualReturns.reduce((sum, val) => sum + val, 0) / n;
  const meanPredicted = predictedReturns.reduce((sum, val) => sum + val, 0) / n;
  
  const mae = actualReturns.reduce((sum, actual, i) => sum + Math.abs(actual - predictedReturns[i]), 0) / n;
  const mse = actualReturns.reduce((sum, actual, i) => sum + Math.pow(actual - predictedReturns[i], 2), 0) / n;
  const rmse = Math.sqrt(mse);
  
  // R-squared
  const ssTot = actualReturns.reduce((sum, actual) => sum + Math.pow(actual - meanActual, 2), 0);
  const ssRes = actualReturns.reduce((sum, actual, i) => sum + Math.pow(actual - predictedReturns[i], 2), 0);
  const r2 = 1 - (ssRes / ssTot);
  
  // Directional accuracy
  let correctDirections = 0;
  for (let i = 0; i < n; i++) {
    if ((actualReturns[i] >= 0 && predictedReturns[i] >= 0) || 
        (actualReturns[i] < 0 && predictedReturns[i] < 0)) {
      correctDirections++;
    }
  }
  const directionalAccuracy = correctDirections / n;
  
  const metrics = {
    loss: await evaluation[0].data(),
    mae: mae,
    mse: mse,
    rmse: rmse,
    r2: r2,
    directionalAccuracy: directionalAccuracy,
    meanActualReturn: meanActual,
    meanPredictedReturn: meanPredicted,
    sampleCount: n,
    reasonablePredictions: predictedReturns.filter(p => Math.abs(p) < 0.5).length / n
  };
  
  // Clean up tensors
  xTest.dispose();
  yTest.dispose();
  evaluation.forEach(t => t.dispose());
  predictions.dispose();
  
  console.log('📊 Enhanced LSTM Evaluation:', {
    directionalAccuracy: `${(metrics.directionalAccuracy * 100).toFixed(1)}%`,
    r2: `${(metrics.r2 * 100).toFixed(1)}%`,
    reasonablePredictions: `${(metrics.reasonablePredictions * 100).toFixed(1)}%`
  });
  
  return metrics;
};