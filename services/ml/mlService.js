// Enhanced Stock Market Prediction using Naive Bayes for Intraday Trading
// src/services/mlService.js

// Function to prepare features from raw stock data (ORIGINAL FUNCTION)
export const prepareFeatures = (stockData, lookbackDays = 5) => {
  if (!stockData || stockData.length < lookbackDays + 1) {
    throw new Error(`Insufficient data. Need at least ${lookbackDays + 1} days of data.`);
  }

  const features = [];
  const labels = [];

  // We'll create features based on previous N days' patterns
  // and predict whether the stock will go up or down
  for (let i = lookbackDays; i < stockData.length; i++) {
    const currentFeatures = [];
    
    // Calculate features from the lookback period
    for (let j = i - lookbackDays; j < i; j++) {
      // Skip the first day as we need the previous day for some calculations
      if (j > i - lookbackDays) {
        // 1. Price momentum (% change)
        const priceChange = (stockData[j].close - stockData[j-1].close) / stockData[j-1].close;
        currentFeatures.push(priceChange);
        
        // 2. Daily volatility (high-low spread)
        const volatility = (stockData[j].high - stockData[j].low) / stockData[j].open;
        currentFeatures.push(volatility);
        
        // 3. Volume change
        const volumeChange = (stockData[j].volume - stockData[j-1].volume) / stockData[j-1].volume;
        currentFeatures.push(volumeChange);
        
        // 4. Open-Close difference
        const openCloseChange = (stockData[j].close - stockData[j].open) / stockData[j].open;
        currentFeatures.push(openCloseChange);
      }
    }

    // Label: 1 if next day's close price is higher than today's, 0 otherwise
    const label = stockData[i].close > stockData[i-1].close ? 1 : 0;
    
    features.push(currentFeatures);
    labels.push(label);
  }

  return { features, labels };
};

// Function to discretize continuous features for Naive Bayes (ORIGINAL FUNCTION)
export const discretizeFeatures = (features, bins = 5) => {
  const numFeatures = features[0].length;
  const discretizedFeatures = [];
  
  // Calculate min and max for each feature column
  const mins = Array(numFeatures).fill(Infinity);
  const maxs = Array(numFeatures).fill(-Infinity);
  
  features.forEach(sample => {
    sample.forEach((value, featureIndex) => {
      mins[featureIndex] = Math.min(mins[featureIndex], value);
      maxs[featureIndex] = Math.max(maxs[featureIndex], value);
    });
  });
  
  // Calculate bin sizes for each feature
  const binSizes = mins.map((min, index) => (maxs[index] - min) / bins);
  
  // Discretize each feature
  features.forEach(sample => {
    const discretizedSample = sample.map((value, featureIndex) => {
      if (binSizes[featureIndex] === 0) return 0; // Handle constant features
      
      // Calculate bin index (0 to bins-1)
      const binIndex = Math.min(
        Math.floor((value - mins[featureIndex]) / binSizes[featureIndex]),
        bins - 1
      );
      
      return binIndex;
    });
    
    discretizedFeatures.push(discretizedSample);
  });
  
  return {
    discretizedFeatures,
    metadata: { mins, maxs, binSizes, bins }
  };
};

// Naive Bayes Implementation (ORIGINAL CLASS)
export class NaiveBayesClassifier {
  constructor(numClasses = 2, featureSpace = null) {
    this.numClasses = numClasses;
    this.classPriors = Array(numClasses).fill(0);
    this.featureSpace = featureSpace;
    this.featureCounts = []; // Array of matrices, one per feature
    this.classCount = Array(numClasses).fill(0);
    this.trained = false;
  }
  
  // Train the classifier with discretized features
  train(discretizedFeatures, labels, metadata) {
    this.metadata = metadata;
    
    const numSamples = discretizedFeatures.length;
    const numFeatures = discretizedFeatures[0].length;
    const { bins } = metadata;
    
    // Initialize feature counts - for each feature, we need a matrix of [class, bin]
    for (let i = 0; i < numFeatures; i++) {
      this.featureCounts.push(
        Array(this.numClasses).fill().map(() => Array(bins).fill(0))
      );
    }
    
    // Count occurrences
    for (let i = 0; i < numSamples; i++) {
      const label = labels[i];
      this.classCount[label]++;
      
      for (let j = 0; j < numFeatures; j++) {
        const binValue = discretizedFeatures[i][j];
        this.featureCounts[j][label][binValue]++;
      }
    }
    
    // Calculate class priors
    for (let c = 0; c < this.numClasses; c++) {
      this.classPriors[c] = this.classCount[c] / numSamples;
    }
    
    this.trained = true;
    return this;
  }
  
  // Predict using the trained model - apply Bayes' theorem
  predict(discretizedFeatures) {
    if (!this.trained) {
      throw new Error("Model not trained yet!");
    }
    
    const predictions = [];
    
    discretizedFeatures.forEach(sample => {
      // Calculate probabilities for each class
      const logProbs = Array(this.numClasses).fill(0);
      
      for (let c = 0; c < this.numClasses; c++) {
        // Start with log of class prior
        logProbs[c] = Math.log(this.classPriors[c]);
        
        // Add log of conditional probabilities for each feature
        for (let f = 0; f < sample.length; f++) {
          const bin = sample[f];
          
          // Apply Laplace smoothing for zero probabilities
          const featureCountForClassAndBin = this.featureCounts[f][c][bin] + 1;
          const totalFeaturesForClass = this.classCount[c] + this.metadata.bins;
          
          const probability = featureCountForClassAndBin / totalFeaturesForClass;
          logProbs[c] += Math.log(probability);
        }
      }
      
      // Find the class with the highest log probability
      const predictedClass = logProbs.indexOf(Math.max(...logProbs));
      predictions.push(predictedClass);
    });
    
    return predictions;
  }
  
  // Get prediction probabilities for each class
  predictProba(discretizedFeatures) {
    if (!this.trained) {
      throw new Error("Model not trained yet!");
    }
    
    const probabilities = [];
    
    discretizedFeatures.forEach(sample => {
      // Calculate probabilities for each class
      const logProbs = Array(this.numClasses).fill(0);
      
      for (let c = 0; c < this.numClasses; c++) {
        // Start with log of class prior
        logProbs[c] = Math.log(this.classPriors[c]);
        
        // Add log of conditional probabilities for each feature
        for (let f = 0; f < sample.length; f++) {
          const bin = sample[f];
          
          // Apply Laplace smoothing for zero probabilities
          const featureCountForClassAndBin = this.featureCounts[f][c][bin] + 1;
          const totalFeaturesForClass = this.classCount[c] + this.metadata.bins;
          
          const probability = featureCountForClassAndBin / totalFeaturesForClass;
          logProbs[c] += Math.log(probability);
        }
      }
      
      // Convert from log probabilities to probabilities
      const maxLogProb = Math.max(...logProbs);
      const expProbs = logProbs.map(lp => Math.exp(lp - maxLogProb));
      const sumExpProbs = expProbs.reduce((a, b) => a + b, 0);
      
      // Normalize to get probabilities
      const normalizedProbs = expProbs.map(p => p / sumExpProbs);
      probabilities.push(normalizedProbs);
    });
    
    return probabilities;
  }
  
  // Evaluate the model performance
  evaluate(discretizedFeatures, trueLabels) {
    const predictions = this.predict(discretizedFeatures);
    
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === trueLabels[i]) {
        correct++;
      }
    }
    
    const accuracy = correct / predictions.length;
    
    // Calculate precision, recall and F1 for each class
    const metrics = { accuracy };
    
    // For binary classification, calculate additional metrics
    if (this.numClasses === 2) {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      
      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === 1 && trueLabels[i] === 1) tp++;
        if (predictions[i] === 1 && trueLabels[i] === 0) fp++;
        if (predictions[i] === 0 && trueLabels[i] === 1) fn++;
        if (predictions[i] === 0 && trueLabels[i] === 0) tn++;
      }
      
      const precision = tp / (tp + fp) || 0;
      const recall = tp / (tp + fn) || 0;
      const f1 = 2 * (precision * recall) / (precision + recall) || 0;
      
      metrics.precision = precision;
      metrics.recall = recall;
      metrics.f1 = f1;
      metrics.confusionMatrix = [
        [tn, fp], // [TN, FP]
        [fn, tp]  // [FN, TP]
      ];
    }
    
    return metrics;
  }
}

// Function to discretize a single prediction sample using the same bins (ORIGINAL FUNCTION)
export const discretizePredictionSample = (sample, metadata) => {
  const { mins, binSizes, bins } = metadata;
  
  return sample.map((value, featureIndex) => {
    if (binSizes[featureIndex] === 0) return 0; // Handle constant features
    
    // Calculate bin index (0 to bins-1)
    const binIndex = Math.min(
      Math.floor((value - mins[featureIndex]) / binSizes[featureIndex]),
      bins - 1
    );
    
    return binIndex;
  });
};

// Function to train and evaluate the model using cross-validation (ORIGINAL FUNCTION)
export const trainAndEvaluate = (stockData, lookbackDays = 5, bins = 5, testSplit = 0.2) => {
  // Prepare features
  const { features, labels } = prepareFeatures(stockData, lookbackDays);
  
  // Discretize features
  const { discretizedFeatures, metadata } = discretizeFeatures(features, bins);
  
  // Split into training and testing sets
  const totalSamples = discretizedFeatures.length;
  const testSize = Math.floor(totalSamples * testSplit);
  const trainSize = totalSamples - testSize;
  
  const trainFeatures = discretizedFeatures.slice(0, trainSize);
  const trainLabels = labels.slice(0, trainSize);
  const testFeatures = discretizedFeatures.slice(trainSize);
  const testLabels = labels.slice(trainSize);
  
  // Train the model
  const model = new NaiveBayesClassifier(2);
  model.train(trainFeatures, trainLabels, metadata);
  
  // Evaluate
  const metrics = model.evaluate(testFeatures, testLabels);
  
  // Return model, metrics, and metadata for later predictions
  return {
    model,
    metrics,
    metadata
  };
};

// Function to prepare the latest data for prediction (ORIGINAL FUNCTION)
export const prepareLatestForPrediction = (stockData, lookbackDays = 5) => {
  if (!stockData || stockData.length < lookbackDays) {
    throw new Error(`Insufficient data for prediction. Need at least ${lookbackDays} days.`);
  }
  
  const latestData = stockData.slice(stockData.length - lookbackDays);
  const features = [];
  
  // Calculate features from the latest data
  for (let j = 1; j < lookbackDays; j++) {  // Start from index 1 to have a previous day
    // 1. Price momentum (% change)
    const priceChange = (latestData[j].close - latestData[j-1].close) / latestData[j-1].close;
    features.push(priceChange);
    
    // 2. Daily volatility (high-low spread)
    const volatility = (latestData[j].high - latestData[j].low) / latestData[j].open;
    features.push(volatility);
    
    // 3. Volume change
    const volumeChange = (latestData[j].volume - latestData[j-1].volume) / latestData[j-1].volume;
    features.push(volumeChange);
    
    // 4. Open-Close difference
    const openCloseChange = (latestData[j].close - latestData[j].open) / latestData[j].open;
    features.push(openCloseChange);
  }
  
  return features;
};

// Function to get current market time info
export const getMarketTimeInfo = () => {
  const now = new Date();
  const marketOpen = new Date(now);
  marketOpen.setHours(9, 30, 0, 0); // 9:30 AM ET
  const marketClose = new Date(now);
  marketClose.setHours(16, 0, 0, 0); // 4:00 PM ET
  
  const isMarketHours = now >= marketOpen && now <= marketClose;
  const minutesFromOpen = isMarketHours ? Math.floor((now - marketOpen) / (1000 * 60)) : 0;
  const timeOfDay = getTimeOfDaySegment(minutesFromOpen);
  
  return {
    isMarketHours,
    minutesFromOpen,
    timeOfDay,
    marketOpen,
    marketClose
  };
};

// Categorize time of day into trading segments
const getTimeOfDaySegment = (minutesFromOpen) => {
  if (minutesFromOpen < 30) return 'OPENING'; // 9:30-10:00
  if (minutesFromOpen < 120) return 'MORNING'; // 10:00-11:30
  if (minutesFromOpen < 210) return 'MIDDAY'; // 11:30-1:00
  if (minutesFromOpen < 330) return 'AFTERNOON'; // 1:00-3:00
  return 'CLOSING'; // 3:00-4:00
};

// Enhanced feature preparation for intraday trading
export const prepareIntradayFeatures = (stockData, currentPrice = null, currentVolume = null, lookbackDays = 5) => {
  if (!stockData || stockData.length < lookbackDays + 1) {
    throw new Error(`Insufficient data. Need at least ${lookbackDays + 1} days of data.`);
  }

  const features = [];
  const labels = [];
  
  // Use market time info
  const marketInfo = getMarketTimeInfo();

  for (let i = lookbackDays; i < stockData.length; i++) {
    const currentFeatures = [];
    
    // Historical pattern features (same as before but enhanced)
    for (let j = i - lookbackDays; j < i; j++) {
      if (j > i - lookbackDays) {
        // 1. Price momentum (% change)
        const priceChange = (stockData[j].close - stockData[j-1].close) / stockData[j-1].close;
        currentFeatures.push(priceChange);
        
        // 2. Daily volatility (high-low spread)
        const volatility = (stockData[j].high - stockData[j].low) / stockData[j].open;
        currentFeatures.push(volatility);
        
        // 3. Volume change
        const volumeChange = (stockData[j].volume - stockData[j-1].volume) / stockData[j-1].volume;
        currentFeatures.push(volumeChange);
        
        // 4. Open-Close difference
        const openCloseChange = (stockData[j].close - stockData[j].open) / stockData[j].open;
        currentFeatures.push(openCloseChange);
        
        // NEW: 5. Gap from previous close
        const gap = (stockData[j].open - stockData[j-1].close) / stockData[j-1].close;
        currentFeatures.push(gap);
      }
    }
    
    // NEW: Current day intraday features (if available)
    if (currentPrice && currentVolume && i === stockData.length - 1) {
      // 6. Current price vs open
      const currentVsOpen = (currentPrice - stockData[i].open) / stockData[i].open;
      currentFeatures.push(currentVsOpen);
      
      // 7. Current price vs yesterday's close
      const currentVsYesterday = (currentPrice - stockData[i-1].close) / stockData[i-1].close;
      currentFeatures.push(currentVsYesterday);
      
      // 8. Time of day factor
      const timeOfDayNumeric = marketInfo.timeOfDay === 'OPENING' ? 0 :
                              marketInfo.timeOfDay === 'MORNING' ? 1 :
                              marketInfo.timeOfDay === 'MIDDAY' ? 2 :
                              marketInfo.timeOfDay === 'AFTERNOON' ? 3 : 4;
      currentFeatures.push(timeOfDayNumeric / 4); // Normalize to 0-1
      
      // 9. Volume pace (current volume vs typical for this time)
      const minutesElapsed = marketInfo.minutesFromOpen;
      const expectedVolumeRatio = minutesElapsed / 390; // 390 minutes in trading day
      const actualVolumeRatio = currentVolume / stockData[i-1].volume;
      const volumePace = actualVolumeRatio / expectedVolumeRatio;
      currentFeatures.push(Math.min(volumePace, 5)); // Cap at 5x normal pace
    }

    // Enhanced labels for different prediction horizons
    if (i < stockData.length - 1) {
      // Label: 1 if next period's price is higher, 0 otherwise
      const label = stockData[i+1].close > stockData[i].close ? 1 : 0;
      labels.push(label);
    } else if (currentPrice) {
      // For current prediction, we can't know the future label
      labels.push(null);
    }
    
    features.push(currentFeatures);
  }

  return { features, labels: labels.filter(l => l !== null) };
};

// Enhanced Naive Bayes with time-aware features
export class IntradayNaiveBayesClassifier extends NaiveBayesClassifier {
  constructor(numClasses = 2, predictionHorizon = 'EOD') {
    super(numClasses);
    this.predictionHorizon = predictionHorizon;
    this.timeOfDayWeights = {
      'OPENING': 1.2,  // Opening patterns are more predictive
      'MORNING': 1.0,
      'MIDDAY': 0.8,   // Lunch hour less predictive
      'AFTERNOON': 1.1,
      'CLOSING': 1.3   // Closing patterns very predictive
    };
  }
  
  // Enhanced prediction with time-of-day weighting
  predictWithTimeWeighting(discretizedFeatures, timeOfDay = 'MORNING') {
    const basePredictions = this.predictProba(discretizedFeatures);
    const weight = this.timeOfDayWeights[timeOfDay] || 1.0;
    
    return basePredictions.map(probs => {
      // Apply time-of-day weighting to confidence
      const maxProb = Math.max(...probs);
      const adjustedConfidence = Math.min(maxProb * weight, 0.95);
      
      // Redistribute probabilities while maintaining the predicted class
      const predictedClass = probs.indexOf(maxProb);
      const newProbs = [...probs];
      newProbs[predictedClass] = adjustedConfidence;
      newProbs[1 - predictedClass] = 1 - adjustedConfidence;
      
      return newProbs;
    });
  }
}

// NEW: Real-time prediction engine
export class RealTimePredictionEngine {
  constructor() {
    this.models = {}; // Store different models for different horizons
    this.lastUpdate = null;
    this.predictions = {};
  }
  
  // Train models for different prediction horizons
  trainModels(stockData, horizons = ['1H', '2H', 'EOD']) {
    const results = {};
    
    horizons.forEach(horizon => {
      try {
        console.log(`Training model for ${horizon} predictions...`);
        
        // Prepare features for this horizon
        const { features, labels } = prepareIntradayFeatures(stockData);
        
        // Discretize features
        const { discretizedFeatures, metadata } = discretizeFeatures(features, 5);
        
        // Split data (use more recent data for testing since it's more relevant)
        const totalSamples = discretizedFeatures.length;
        const testSize = Math.floor(totalSamples * 0.2);
        const trainSize = totalSamples - testSize;
        
        const trainFeatures = discretizedFeatures.slice(0, trainSize);
        const trainLabels = labels.slice(0, trainSize);
        const testFeatures = discretizedFeatures.slice(trainSize);
        const testLabels = labels.slice(trainSize);
        
        // Train model
        const model = new IntradayNaiveBayesClassifier(2, horizon);
        model.train(trainFeatures, trainLabels, metadata);
        
        // Evaluate
        const metrics = model.evaluate(testFeatures, testLabels);
        
        this.models[horizon] = { model, metadata, metrics };
        results[horizon] = { metrics };
        
      } catch (error) {
        console.error(`Error training ${horizon} model:`, error);
        results[horizon] = { error: error.message };
      }
    });
    
    return results;
  }
  
  // Make real-time predictions
  updatePredictions(stockData, currentPrice, currentVolume) {
    const marketInfo = getMarketTimeInfo();
    
    if (!marketInfo.isMarketHours) {
      return {
        error: 'Market is closed',
        nextUpdate: marketInfo.marketOpen
      };
    }
    
    const predictions = {};
    
    Object.keys(this.models).forEach(horizon => {
      try {
        const { model, metadata } = this.models[horizon];
        
        // Prepare current features
        const latestFeatures = prepareIntradayLatestFeatures(
          stockData, currentPrice, currentVolume, horizon
        );
        
        // Discretize and predict
        const discretizedFeatures = [discretizePredictionSample(latestFeatures, metadata)];
        const probabilities = model.predictWithTimeWeighting(
          discretizedFeatures, 
          marketInfo.timeOfDay
        )[0];
        
        predictions[horizon] = {
          prediction: probabilities[1] > probabilities[0] ? 'UP' : 'DOWN',
          upProbability: probabilities[1],
          downProbability: probabilities[0],
          confidence: Math.max(...probabilities),
          timeOfDay: marketInfo.timeOfDay,
          minutesFromOpen: marketInfo.minutesFromOpen
        };
        
      } catch (error) {
        predictions[horizon] = { error: error.message };
      }
    });
    
    this.predictions = predictions;
    this.lastUpdate = new Date();
    
    return {
      predictions,
      lastUpdate: this.lastUpdate,
      marketInfo,
      nextUpdateRecommended: new Date(Date.now() + 15 * 60 * 1000) // 15 minutes
    };
  }
  
  // Get trading signals based on predictions
  getTradingSignals() {
    const signals = [];
    
    // Analyze predictions across different horizons
    const horizons = Object.keys(this.predictions);
    const upCount = horizons.filter(h => 
      this.predictions[h].prediction === 'UP' && this.predictions[h].confidence > 0.6
    ).length;
    
    const downCount = horizons.filter(h => 
      this.predictions[h].prediction === 'DOWN' && this.predictions[h].confidence > 0.6
    ).length;
    
    // Generate signals based on consensus
    if (upCount > downCount && upCount >= 2) {
      signals.push({
        type: 'BUY',
        strength: upCount / horizons.length,
        reason: `${upCount}/${horizons.length} models predict upward movement`,
        timeframe: 'INTRADAY'
      });
    } else if (downCount > upCount && downCount >= 2) {
      signals.push({
        type: 'SELL',
        strength: downCount / horizons.length,
        reason: `${downCount}/${horizons.length} models predict downward movement`,
        timeframe: 'INTRADAY'
      });
    }
    
    return signals;
  }
}

// Helper function to prepare latest features for different horizons
const prepareIntradayLatestFeatures = (stockData, currentPrice, currentVolume, horizon) => {
  const lookbackDays = horizon === '1H' ? 2 : horizon === '2H' ? 3 : 5;
  
  if (!stockData || stockData.length < lookbackDays) {
    throw new Error(`Insufficient data for ${horizon} prediction.`);
  }
  
  const latestData = stockData.slice(stockData.length - lookbackDays);
  const features = [];
  const marketInfo = getMarketTimeInfo();
  
  // Historical features
  for (let j = 1; j < lookbackDays; j++) {
    const priceChange = (latestData[j].close - latestData[j-1].close) / latestData[j-1].close;
    features.push(priceChange);
    
    const volatility = (latestData[j].high - latestData[j].low) / latestData[j].open;
    features.push(volatility);
    
    const volumeChange = (latestData[j].volume - latestData[j-1].volume) / latestData[j-1].volume;
    features.push(volumeChange);
    
    const openCloseChange = (latestData[j].close - latestData[j].open) / latestData[j].open;
    features.push(openCloseChange);
    
    const gap = (latestData[j].open - latestData[j-1].close) / latestData[j-1].close;
    features.push(gap);
  }
  
  // Current intraday features
  if (currentPrice && currentVolume) {
    const lastDayData = latestData[latestData.length - 1];
    
    const currentVsOpen = (currentPrice - lastDayData.open) / lastDayData.open;
    features.push(currentVsOpen);
    
    const currentVsYesterday = (currentPrice - latestData[latestData.length - 2].close) / latestData[latestData.length - 2].close;
    features.push(currentVsYesterday);
    
    const timeOfDayNumeric = marketInfo.timeOfDay === 'OPENING' ? 0 :
                            marketInfo.timeOfDay === 'MORNING' ? 1 :
                            marketInfo.timeOfDay === 'MIDDAY' ? 2 :
                            marketInfo.timeOfDay === 'AFTERNOON' ? 3 : 4;
    features.push(timeOfDayNumeric / 4);
    
    const minutesElapsed = marketInfo.minutesFromOpen;
    const expectedVolumeRatio = minutesElapsed / 390;
    const actualVolumeRatio = currentVolume / lastDayData.volume;
    const volumePace = actualVolumeRatio / expectedVolumeRatio;
    features.push(Math.min(volumePace, 5));
  }
  
  return features;
};

// Enhanced training function for intraday models
export const trainIntradayModels = (stockData, horizons = ['1H', '2H', 'EOD']) => {
  const engine = new RealTimePredictionEngine();
  const results = engine.trainModels(stockData, horizons);
  
  return {
    engine,
    results,
    isReady: Object.keys(engine.models).length > 0
  };
};