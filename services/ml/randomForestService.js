// Random Forest Implementation for Stock Market Prediction
// src/services/randomForestService.js
// Updated to use comprehensive technical indicators

import { extractTechnicalFeatures } from './technicalIndicators';

// Decision Tree Node
class TreeNode {
  constructor() {
    this.feature = null;
    this.threshold = null;
    this.left = null;
    this.right = null;
    this.value = null; // For leaf nodes
    this.samples = 0;
    this.impurity = 0;
  }
  
  isLeaf() {
    return this.value !== null;
  }
}

// Decision Tree Implementation
class DecisionTree {
  constructor(maxDepth = 10, minSamplesSplit = 2, minSamplesLeaf = 1) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.root = null;
    this.featureImportances = null;
  }
  
  // Calculate Gini impurity
  calculateGini(labels) {
    if (labels.length === 0) return 0;
    
    const counts = {};
    labels.forEach(label => {
      counts[label] = (counts[label] || 0) + 1;
    });
    
    let gini = 1;
    const total = labels.length;
    
    Object.values(counts).forEach(count => {
      const probability = count / total;
      gini -= probability * probability;
    });
    
    return gini;
  }
  
  // Find best split for a feature
  findBestSplit(features, labels, featureIndex) {
    const values = features.map(sample => sample[featureIndex]);
    const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
    
    let bestThreshold = null;
    let bestGain = -1;
    
    for (let i = 0; i < uniqueValues.length - 1; i++) {
      const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
      
      const leftIndices = [];
      const rightIndices = [];
      
      for (let j = 0; j < features.length; j++) {
        if (features[j][featureIndex] <= threshold) {
          leftIndices.push(j);
        } else {
          rightIndices.push(j);
        }
      }
      
      if (leftIndices.length === 0 || rightIndices.length === 0) continue;
      
      const leftLabels = leftIndices.map(idx => labels[idx]);
      const rightLabels = rightIndices.map(idx => labels[idx]);
      
      const parentGini = this.calculateGini(labels);
      const leftGini = this.calculateGini(leftLabels);
      const rightGini = this.calculateGini(rightLabels);
      
      const weightedGini = (leftLabels.length / labels.length) * leftGini + 
                          (rightLabels.length / labels.length) * rightGini;
      
      const gain = parentGini - weightedGini;
      
      if (gain > bestGain) {
        bestGain = gain;
        bestThreshold = threshold;
      }
    }
    
    return { threshold: bestThreshold, gain: bestGain };
  }
  
  // Find best split across all features
  findBestFeatureSplit(features, labels, availableFeatures) {
    let bestFeature = null;
    let bestThreshold = null;
    let bestGain = -1;
    
    availableFeatures.forEach(featureIndex => {
      const split = this.findBestSplit(features, labels, featureIndex);
      
      if (split.gain > bestGain) {
        bestGain = split.gain;
        bestFeature = featureIndex;
        bestThreshold = split.threshold;
      }
    });
    
    return { feature: bestFeature, threshold: bestThreshold, gain: bestGain };
  }
  
  // Build tree recursively
  buildTree(features, labels, depth = 0, availableFeatures = null) {
    const node = new TreeNode();
    node.samples = features.length;
    node.impurity = this.calculateGini(labels);
    
    // Initialize available features on first call
    if (availableFeatures === null) {
      availableFeatures = Array.from({length: features[0].length}, (_, i) => i);
    }
    
    // Check stopping criteria
    if (depth >= this.maxDepth || 
        features.length < this.minSamplesSplit ||
        node.impurity === 0 ||
        availableFeatures.length === 0) {
      
      // Create leaf node
      const counts = {};
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      node.value = Object.keys(counts).reduce((a, b) => 
        counts[a] > counts[b] ? a : b
      );
      
      return node;
    }
    
    // Find best split
    const bestSplit = this.findBestFeatureSplit(features, labels, availableFeatures);
    
    if (bestSplit.feature === null || bestSplit.gain <= 0) {
      // No good split found, create leaf
      const counts = {};
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      node.value = Object.keys(counts).reduce((a, b) => 
        counts[a] > counts[b] ? a : b
      );
      
      return node;
    }
    
    node.feature = bestSplit.feature;
    node.threshold = bestSplit.threshold;
    
    // Split data
    const leftFeatures = [];
    const leftLabels = [];
    const rightFeatures = [];
    const rightLabels = [];
    
    for (let i = 0; i < features.length; i++) {
      if (features[i][node.feature] <= node.threshold) {
        leftFeatures.push(features[i]);
        leftLabels.push(labels[i]);
      } else {
        rightFeatures.push(features[i]);
        rightLabels.push(labels[i]);
      }
    }
    
    // Check minimum samples per leaf
    if (leftFeatures.length < this.minSamplesLeaf || rightFeatures.length < this.minSamplesLeaf) {
      const counts = {};
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      node.value = Object.keys(counts).reduce((a, b) => 
        counts[a] > counts[b] ? a : b
      );
      
      return node;
    }
    
    // Recursively build subtrees
    node.left = this.buildTree(leftFeatures, leftLabels, depth + 1, availableFeatures);
    node.right = this.buildTree(rightFeatures, rightLabels, depth + 1, availableFeatures);
    
    return node;
  }
  
  // Train the tree
  train(features, labels) {
    this.root = this.buildTree(features, labels);
    this.calculateFeatureImportances(features);
    return this;
  }
  
  // Calculate feature importances
  calculateFeatureImportances(features) {
    const numFeatures = features[0].length;
    this.featureImportances = new Array(numFeatures).fill(0);
    
    const calculateNodeImportance = (node, totalSamples) => {
      if (node.isLeaf()) return;
      
      const importance = (node.samples / totalSamples) * node.impurity;
      this.featureImportances[node.feature] += importance;
      
      if (node.left) calculateNodeImportance(node.left, totalSamples);
      if (node.right) calculateNodeImportance(node.right, totalSamples);
    };
    
    if (this.root) {
      calculateNodeImportance(this.root, this.root.samples);
      
      // Normalize importances
      const sum = this.featureImportances.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        this.featureImportances = this.featureImportances.map(imp => imp / sum);
      }
    }
  }
  
  // Predict single sample
  predictSample(sample) {
    let node = this.root;
    
    while (!node.isLeaf()) {
      if (sample[node.feature] <= node.threshold) {
        node = node.left;
      } else {
        node = node.right;
      }
    }
    
    return parseInt(node.value);
  }
  
  // Predict multiple samples
  predict(features) {
    return features.map(sample => this.predictSample(sample));
  }
}

// Random Forest Implementation
export class RandomForestClassifier {
  constructor(nTrees = 100, maxDepth = 10, minSamplesSplit = 2, minSamplesLeaf = 1, maxFeatures = 'sqrt') {
    this.nTrees = nTrees;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.maxFeatures = maxFeatures;
    this.trees = [];
    this.featureImportances = null;
    this.trained = false;
  }
  
  // Bootstrap sampling
  bootstrapSample(features, labels) {
    const n = features.length;
    const indices = [];
    
    for (let i = 0; i < n; i++) {
      indices.push(Math.floor(Math.random() * n));
    }
    
    const sampledFeatures = indices.map(idx => features[idx]);
    const sampledLabels = indices.map(idx => labels[idx]);
    
    return { features: sampledFeatures, labels: sampledLabels };
  }
  
  // Select random features for each tree
  getRandomFeatures(numFeatures) {
    let maxFeaturesCount;
    
    if (this.maxFeatures === 'sqrt') {
      maxFeaturesCount = Math.floor(Math.sqrt(numFeatures));
    } else if (this.maxFeatures === 'log2') {
      maxFeaturesCount = Math.floor(Math.log2(numFeatures));
    } else if (typeof this.maxFeatures === 'number') {
      maxFeaturesCount = Math.min(this.maxFeatures, numFeatures);
    } else {
      maxFeaturesCount = numFeatures;
    }
    
    const allFeatures = Array.from({length: numFeatures}, (_, i) => i);
    const selectedFeatures = [];
    
    for (let i = 0; i < maxFeaturesCount; i++) {
      const randomIndex = Math.floor(Math.random() * allFeatures.length);
      selectedFeatures.push(allFeatures.splice(randomIndex, 1)[0]);
    }
    
    return selectedFeatures;
  }
  
  // Train the forest
  train(features, labels) {
    console.log(`Training Random Forest with ${this.nTrees} trees...`);
    
    this.trees = [];
    const numFeatures = features[0].length;
    const allFeatureImportances = [];
    
    for (let i = 0; i < this.nTrees; i++) {
      if (i % 20 === 0) {
        console.log(`Training tree ${i + 1}/${this.nTrees}`);
      }
      
      // Bootstrap sample
      const { features: sampledFeatures, labels: sampledLabels } = this.bootstrapSample(features, labels);
      
      // Random feature selection
      const selectedFeatures = this.getRandomFeatures(numFeatures);
      
      // Create and train tree
      const tree = new DecisionTree(this.maxDepth, this.minSamplesSplit, this.minSamplesLeaf);
      tree.selectedFeatures = selectedFeatures;
      
      // Filter features for this tree
      const filteredFeatures = sampledFeatures.map(sample => 
        selectedFeatures.map(featureIdx => sample[featureIdx])
      );
      
      tree.train(filteredFeatures, sampledLabels);
      this.trees.push(tree);
      
      // Collect feature importances
      if (tree.featureImportances) {
        const fullImportances = new Array(numFeatures).fill(0);
        selectedFeatures.forEach((featureIdx, treeFeatureIdx) => {
          fullImportances[featureIdx] = tree.featureImportances[treeFeatureIdx];
        });
        allFeatureImportances.push(fullImportances);
      }
    }
    
    // Calculate average feature importances
    this.featureImportances = new Array(numFeatures).fill(0);
    allFeatureImportances.forEach(importances => {
      importances.forEach((importance, idx) => {
        this.featureImportances[idx] += importance;
      });
    });
    
    this.featureImportances = this.featureImportances.map(imp => imp / this.nTrees);
    
    this.trained = true;
    console.log('Random Forest training completed');
    return this;
  }
  
  // Predict probabilities
  predictProba(features) {
    if (!this.trained) {
      throw new Error('Model not trained yet!');
    }
    
    const predictions = features.map(sample => {
      const votes = { 0: 0, 1: 0 };
      
      this.trees.forEach(tree => {
        // Filter sample for this tree's features
        const filteredSample = tree.selectedFeatures.map(featureIdx => sample[featureIdx]);
        const prediction = tree.predictSample(filteredSample);
        votes[prediction]++;
      });
      
      const total = this.nTrees;
      return [votes[0] / total, votes[1] / total];
    });
    
    return predictions;
  }
  
  // Predict classes
  predict(features) {
    const probabilities = this.predictProba(features);
    return probabilities.map(probs => probs[1] > probs[0] ? 1 : 0);
  }
  
  // Evaluate model performance
  evaluate(features, trueLabels) {
    const predictions = this.predict(features);
    
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === trueLabels[i]) {
        correct++;
      }
    }
    
    const accuracy = correct / predictions.length;
    
    // Calculate precision, recall and F1 for binary classification
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
    
    return {
      accuracy,
      precision,
      recall,
      f1,
      confusionMatrix: [
        [tn, fp], // [TN, FP]
        [fn, tp]  // [FN, TP]
      ]
    };
  }
}

// Feature preparation for Random Forest using comprehensive technical indicators
export const prepareRandomForestFeatures = (stockData, lookbackDays = 5) => {
  if (!stockData || stockData.length < Math.max(lookbackDays + 1, 200)) {
    throw new Error(`Insufficient data. Need at least 200 days of data for technical indicators.`);
  }

  // Prepare data in the format expected by extractTechnicalFeatures
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  // Extract technical features
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays + 1) {
    throw new Error(`Insufficient technical feature data. Need at least ${lookbackDays + 1} processed features.`);
  }

  const features = [];
  const labels = [];

  for (let i = lookbackDays; i < technicalFeatures.length; i++) {
    const currentFeatures = [];
    
    // Use lookback period of technical features
    for (let j = i - lookbackDays; j < i; j++) {
      const feature = technicalFeatures[j];
      
      // Technical indicator features
      currentFeatures.push(feature.sma20Ratio);
      currentFeatures.push(feature.sma50Ratio);
      currentFeatures.push(feature.sma200Ratio);
      currentFeatures.push(feature.macd);
      currentFeatures.push(feature.macdSignal);
      currentFeatures.push(feature.macdHistogram);
      currentFeatures.push(feature.rsi / 100); // Normalize RSI to 0-1
      currentFeatures.push(feature.volatility);
      currentFeatures.push(feature.volumeRatio);
      currentFeatures.push(feature.pricePosition);
      currentFeatures.push(feature.bottomSignal);
      currentFeatures.push(feature.peakSignal);
      
      // Additional price-based features
      if (j > 0) {
        const prevFeature = technicalFeatures[j - 1];
        
        // Price momentum
        const priceChange = (feature.close - prevFeature.close) / prevFeature.close;
        currentFeatures.push(priceChange);
        
        // RSI momentum
        const rsiChange = feature.rsi - prevFeature.rsi;
        currentFeatures.push(rsiChange / 100); // Normalize
        
        // MACD momentum
        const macdChange = feature.macd - prevFeature.macd;
        currentFeatures.push(macdChange);
        
        // Volume ratio change
        const volRatioChange = feature.volumeRatio - prevFeature.volumeRatio;
        currentFeatures.push(volRatioChange);
      } else {
        // For the first item in lookback, use zero momentum
        currentFeatures.push(0, 0, 0, 0);
      }
    }

    // Label: 1 if next day's close price is higher than today's, 0 otherwise
    const currentClose = technicalFeatures[i].close;
    const prevClose = technicalFeatures[i - 1].close;
    const label = currentClose > prevClose ? 1 : 0;
    
    features.push(currentFeatures);
    labels.push(label);
  }

  return { features, labels };
};

// Prepare latest features for prediction using technical indicators
export const prepareLatestRandomForestFeatures = (stockData, lookbackDays = 5) => {
  if (!stockData || stockData.length < Math.max(lookbackDays, 200)) {
    throw new Error(`Insufficient data for prediction. Need at least 200 days for technical indicators.`);
  }
  
  // Prepare data in the format expected by extractTechnicalFeatures
  const formattedData = {
    close: stockData.map(d => d.close),
    high: stockData.map(d => d.high),
    low: stockData.map(d => d.low),
    volume: stockData.map(d => d.volume),
    dates: stockData.map(d => d.date)
  };
  
  // Extract technical features
  const technicalFeatures = extractTechnicalFeatures(formattedData);
  
  if (technicalFeatures.length < lookbackDays) {
    throw new Error(`Insufficient technical feature data for prediction. Need at least ${lookbackDays} processed features.`);
  }
  
  // Use the latest lookback period
  const latestFeatures = technicalFeatures.slice(technicalFeatures.length - lookbackDays);
  const features = [];
  
  // Calculate features from the latest data
  for (let j = 0; j < lookbackDays; j++) {
    const feature = latestFeatures[j];
    
    // Technical indicator features
    features.push(feature.sma20Ratio);
    features.push(feature.sma50Ratio);
    features.push(feature.sma200Ratio);
    features.push(feature.macd);
    features.push(feature.macdSignal);
    features.push(feature.macdHistogram);
    features.push(feature.rsi / 100); // Normalize RSI to 0-1
    features.push(feature.volatility);
    features.push(feature.volumeRatio);
    features.push(feature.pricePosition);
    features.push(feature.bottomSignal);
    features.push(feature.peakSignal);
    
    // Additional price-based features
    if (j > 0) {
      const prevFeature = latestFeatures[j - 1];
      
      // Price momentum
      const priceChange = (feature.close - prevFeature.close) / prevFeature.close;
      features.push(priceChange);
      
      // RSI momentum
      const rsiChange = feature.rsi - prevFeature.rsi;
      features.push(rsiChange / 100); // Normalize
      
      // MACD momentum
      const macdChange = feature.macd - prevFeature.macd;
      features.push(macdChange);
      
      // Volume ratio change
      const volRatioChange = feature.volumeRatio - prevFeature.volumeRatio;
      features.push(volRatioChange);
    } else {
      // For the first item in lookback, use zero momentum
      features.push(0, 0, 0, 0);
    }
  }
  
  return features;
};

// Train and evaluate Random Forest model using technical indicators
export const trainRandomForest = (stockData, lookbackDays = 5, nTrees = 50, testSplit = 0.2) => {
  // Prepare features using technical indicators
  const { features, labels } = prepareRandomForestFeatures(stockData, lookbackDays);
  
  console.log(`Prepared ${features.length} samples with ${features[0].length} technical features each`);
  
  // Split into training and testing sets
  const totalSamples = features.length;
  const testSize = Math.floor(totalSamples * testSplit);
  const trainSize = totalSamples - testSize;
  
  const trainFeatures = features.slice(0, trainSize);
  const trainLabels = labels.slice(0, trainSize);
  const testFeatures = features.slice(trainSize);
  const testLabels = labels.slice(trainSize);
  
  console.log(`Training set: ${trainSize} samples, Test set: ${testSize} samples`);
  
  // Train the model
  const model = new RandomForestClassifier(nTrees, 10, 2, 1, 'sqrt');
  model.train(trainFeatures, trainLabels);
  
  // Evaluate
  const metrics = model.evaluate(testFeatures, testLabels);
  
  // Calculate feature importance analysis
  const featureNames = [];
  for (let day = 0; day < lookbackDays; day++) {
    featureNames.push(`Day${day}_SMA20Ratio`);
    featureNames.push(`Day${day}_SMA50Ratio`);
    featureNames.push(`Day${day}_SMA200Ratio`);
    featureNames.push(`Day${day}_MACD`);
    featureNames.push(`Day${day}_MACDSignal`);
    featureNames.push(`Day${day}_MACDHistogram`);
    featureNames.push(`Day${day}_RSI`);
    featureNames.push(`Day${day}_Volatility`);
    featureNames.push(`Day${day}_VolumeRatio`);
    featureNames.push(`Day${day}_PricePosition`);
    featureNames.push(`Day${day}_BottomSignal`);
    featureNames.push(`Day${day}_PeakSignal`);
    
    if (day > 0) {
      featureNames.push(`Day${day}_PriceMomentum`);
      featureNames.push(`Day${day}_RSIMomentum`);
      featureNames.push(`Day${day}_MACDMomentum`);
      featureNames.push(`Day${day}_VolRatioMomentum`);
    } else {
      featureNames.push(`Day${day}_PriceMomentum_Zero`);
      featureNames.push(`Day${day}_RSIMomentum_Zero`);
      featureNames.push(`Day${day}_MACDMomentum_Zero`);
      featureNames.push(`Day${day}_VolRatioMomentum_Zero`);
    }
  }
  
  const featureImportanceDetails = model.featureImportances.map((importance, index) => ({
    feature: featureNames[index] || `Feature_${index}`,
    importance: importance,
    rank: index + 1
  })).sort((a, b) => b.importance - a.importance);
  
  // Return model, metrics, and metadata for later predictions
  return {
    model,
    metrics,
    featureImportances: model.featureImportances,
    featureImportanceDetails,
    numTrees: nTrees,
    numFeatures: features[0].length,
    lookbackDays,
    trainingSamples: trainSize,
    testSamples: testSize,
    technicalIndicatorsUsed: [
      'SMA (20, 50, 200)', 'MACD & Signal', 'RSI', 
      'Volatility', 'Volume Ratio', 'Price Position',
      'Bottom/Peak Signals', 'Momentum Indicators'
    ]
  };
};

export const predictWithRandomForest = (model, stockData, lookbackDays = 5, horizonDays = 1) => {
  if (!model || typeof model.predictProba !== 'function') {
    throw new Error('Random Forest model is not available for prediction');
  }

  const latestFeatures = prepareLatestRandomForestFeatures(stockData, lookbackDays);
  const probabilities = model.predictProba([latestFeatures])[0] || [0.5, 0.5];
  const downProbability = Number(probabilities[0] ?? 0.5);
  const upProbability = Number(probabilities[1] ?? 0.5);

  const currentPrice = Number(stockData?.[stockData.length - 1]?.close ?? 0);
  if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
    throw new Error('Unable to resolve current price for Random Forest prediction');
  }

  const recentReturns = (stockData || [])
    .slice(-30)
    .map((row, index, arr) => {
      if (index === 0) return null;
      const prev = Number(arr[index - 1]?.close);
      const curr = Number(row?.close);
      if (!Number.isFinite(prev) || !Number.isFinite(curr) || prev <= 0) return null;
      return (curr - prev) / prev;
    })
    .filter((value) => Number.isFinite(value));

  const averageAbsoluteMove = recentReturns.length
    ? recentReturns.reduce((sum, value) => sum + Math.abs(value), 0) / recentReturns.length
    : 0.01;

  const score = upProbability - downProbability;
  const scaledHorizon = Math.max(1, Number(horizonDays) || 1);
  const expectedReturn = score * averageAbsoluteMove * Math.sqrt(scaledHorizon);
  const predictedPrice = currentPrice * (1 + expectedReturn);

  return {
    currentPrice,
    predictedPrice,
    predictedReturn: expectedReturn,
    percentChange: expectedReturn * 100,
    priceChange: predictedPrice - currentPrice,
    direction: expectedReturn >= 0 ? 'UP' : 'DOWN',
    confidence: Math.max(upProbability, downProbability),
    upProbability,
    downProbability,
    horizonDays: scaledHorizon,
    heuristic: 'Probability score scaled by recent realized volatility',
  };
};