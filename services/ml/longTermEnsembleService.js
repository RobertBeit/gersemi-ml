// Fixed Long-Term Ensemble Service - Properly handles RF Classification + LSTM Regression
// src/services/longTermEnsembleService.js

// UPDATED IMPORTS
import { 
  trainLongTermRandomForest, 
  prepareLatestLongTermFeatures,
  LongTermRandomForestClassifier 
} from './longTermRandomForestService';
import { 
  trainLongTermLSTM, 
  predictLongTermWithConfidence, 
} from './longTermLSTMService';

import { extractTechnicalFeatures } from './technicalIndicators';

// Enhanced category to return conversion with confidence weighting
const convertCategoryToReturn = (category, confidence = 0.6) => {
  const baseReturns = {
    0: -0.025, // Big Down: -2.5%
    1: -0.008, // Small Down: -0.8%
    2: 0.000,  // Flat: 0%
    3: 0.008,  // Small Up: +0.8%
    4: 0.025   // Big Up: +2.5%
  };
  
  const baseReturn = baseReturns[category] || 0;
  
  // Scale return by confidence (high confidence = closer to base, low confidence = closer to 0)
  const scaledReturn = baseReturn * Math.max(0.3, confidence);
  
  return scaledReturn;
};

// Enhanced confidence conversion for classification results
const convertClassificationConfidence = (classificationResult) => {
  if (!classificationResult) return 0.5;
  
  // If we have probabilities, calculate confidence from the distribution
  if (classificationResult.probabilities && Array.isArray(classificationResult.probabilities)) {
    const probs = classificationResult.probabilities;
    const maxProb = Math.max(...probs);
    
    // Shannon entropy-based confidence
    const entropy = -probs.reduce((sum, p) => {
      return sum + (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
    const maxEntropy = Math.log2(probs.length);
    const normalizedEntropy = entropy / maxEntropy;
    
    // Combine max probability and entropy for confidence
    const entropyConfidence = 1 - normalizedEntropy;
    const confidence = (maxProb * 0.7) + (entropyConfidence * 0.3);
    
    return Math.max(0.1, Math.min(0.95, confidence));
  }
  
  // Fallback confidence measures
  if (classificationResult.confidence !== undefined) {
    return Math.max(0.1, Math.min(0.95, classificationResult.confidence));
  }
  
  if (classificationResult.agreementScore !== undefined) {
    return Math.max(0.1, Math.min(0.95, classificationResult.agreementScore));
  }
  
  // Default classification confidence
  return 0.6;
};

// Enhanced Long-Term Ensemble Predictor Class
export class LongTermEnsemblePredictor {
  constructor(ensembleConfig = {}) {
    this.config = {
      // Model weights
      randomForestWeight: ensembleConfig.randomForestWeight || 0.4,
      lstmWeight: ensembleConfig.lstmWeight || 0.6,
      
      // Prediction parameters
      lookbackDays: ensembleConfig.lookbackDays || 30,
      targetDaysAhead: ensembleConfig.targetDaysAhead || 5,
      
      // Confidence thresholds
      minConfidenceThreshold: ensembleConfig.minConfidenceThreshold || 0.6,
      consensusThreshold: ensembleConfig.consensusThreshold || 0.7,
      
      // Model-specific parameters
      modelParams: {
        randomForest: {
          lookbackDays: ensembleConfig.lookbackDays || 30,
          targetDaysAhead: ensembleConfig.targetDaysAhead || 22, // RF typically uses longer horizons
          nTrees: ensembleConfig.rfTrees || 100,
          testSplit: 0.2,
          maxSelectedFeatures: 50
        },
        lstm: {
          sequenceLength: ensembleConfig.lstmSequenceLength || 40,
          targetDaysAhead: ensembleConfig.targetDaysAhead || 5,
          epochs: ensembleConfig.lstmEpochs || 50,
          batchSize: 32
        }
      }
    };
    
    // Model instances and results
    this.models = {
      randomForest: null,
      lstm: null
    };
    
    // Training results with proper metrics
    this.trainingResults = {
      randomForest: null,
      lstm: null
    };
    
    this.trained = false;
    this.lastTrainingData = null;
    this.featureSelector = null; // For RF feature selection
  }
  
  // Enhanced ensemble training with proper metrics
  async trainLongTermEnsemble(stockData, progressCallback = null) {
    console.log(`🎯 Starting Enhanced Long-Term Ensemble Training - RF(Classification) + LSTM(Regression)...`);
    
    const totalSteps = 6;
    let currentStep = 0;
    
    const updateProgress = (message) => {
      currentStep++;
      const progress = Math.floor((currentStep / totalSteps) * 100);
      console.log(`[${progress}%] ${message}`);
      if (progressCallback) progressCallback(progress, message);
    };
    
    try {
      this.lastTrainingData = stockData;
      const requiredDataPoints = Math.max(
        this.config.lookbackDays + this.config.targetDaysAhead + 252, 
        400
      );
      
      if (!stockData || stockData.length < requiredDataPoints) {
        throw new Error(`Insufficient data. Need at least ${requiredDataPoints} days of stock data.`);
      }
      
      // Train Random Forest Classifier
      updateProgress('Training Random Forest Classifier...');
      try {
        console.log('🌲 Training RF with params:', this.config.modelParams.randomForest);
        
        const rfResult = trainLongTermRandomForest(
          stockData,
          this.config.modelParams.randomForest.lookbackDays,
          this.config.modelParams.randomForest.targetDaysAhead,
          this.config.modelParams.randomForest.nTrees,
          this.config.modelParams.randomForest.testSplit,
          this.config.modelParams.randomForest.maxSelectedFeatures
        );
        
        this.trainingResults.randomForest = {
          ...rfResult,
          // Use classification accuracy (not R²)
          accuracy: rfResult.metrics.accuracy,
          f1Score: rfResult.metrics.f1,
          precision: rfResult.metrics.precision,
          recall: rfResult.metrics.recall,
          modelType: 'classifier'
        };
        
        this.models.randomForest = rfResult.model;
        this.featureSelector = rfResult.model.featureSelector; // Save feature selector
        
        console.log('✅ RF Classifier trained - Accuracy:', 
          (rfResult.metrics.accuracy * 100).toFixed(1) + '%', 
          'F1:', (rfResult.metrics.f1 * 100).toFixed(1) + '%');
          
      } catch (error) {
        console.error('❌ Random Forest training failed:', error);
        this.trainingResults.randomForest = { 
          error: error.message, 
          accuracy: 0, 
          modelType: 'classifier' 
        };
      }
      
      // Train LSTM Regressor
      updateProgress('Training LSTM Regressor...');
      try {
        console.log('🧠 Training LSTM with params:', this.config.modelParams.lstm);
        
        const lstmResult = await trainLongTermLSTM(
          stockData,
          this.config.modelParams.lstm.sequenceLength,
          this.config.modelParams.lstm.targetDaysAhead,
          this.config.modelParams.lstm.epochs,
          this.config.modelParams.lstm.batchSize
        );
        
        this.trainingResults.lstm = {
          ...lstmResult,
          // Convert MAE to accuracy-like metric
          accuracy: Math.max(0, 1 - lstmResult.finalMetrics.valMAE),
          mae: lstmResult.finalMetrics.valMAE,
          mse: lstmResult.finalMetrics.trainLoss,
          modelType: 'regressor'
        };
        
        this.models.lstm = lstmResult.model;
        
        console.log('✅ LSTM Regressor trained - Val MAE:', 
          lstmResult.finalMetrics.valMAE.toFixed(4),
          'Accuracy-like:', (this.trainingResults.lstm.accuracy * 100).toFixed(1) + '%');
          
      } catch (error) {
        console.error('❌ LSTM training failed:', error);
        this.trainingResults.lstm = { 
          error: error.message, 
          accuracy: 0, 
          modelType: 'regressor' 
        };
      }
      
      // Calculate optimal weights
      updateProgress('Calculating optimal ensemble weights...');
      this.calculateDynamicWeights();
      
      // Validate ensemble
      updateProgress('Validating ensemble performance...');
      const ensembleMetrics = await this.validateLongTermEnsemble(stockData);
      
      updateProgress('Ensemble training completed!');
      
      this.trained = true;
      
      return {
        success: true,
        individualResults: this.trainingResults,
        ensembleMetrics: ensembleMetrics,
        optimalWeights: {
          randomForest: this.config.randomForestWeight,
          lstm: this.config.lstmWeight
        },
        totalModels: this.getTrainedModelCount(),
        predictionHorizon: this.config.targetDaysAhead
      };
      
    } catch (error) {
      console.error('❌ Ensemble training failed:', error);
      throw new Error(`Ensemble training failed: ${error.message}`);
    }
  }
  
  // Enhanced dynamic weight calculation
  calculateDynamicWeights() {
    const rfAccuracy = this.trainingResults.randomForest?.accuracy || 0;
    const lstmAccuracy = this.trainingResults.lstm?.accuracy || 0;
    
    console.log('📊 Individual Model Performance:');
    console.log(`  Random Forest (Classifier): ${(rfAccuracy * 100).toFixed(1)}% accuracy`);
    console.log(`  LSTM (Regressor): ${(lstmAccuracy * 100).toFixed(1)}% accuracy-like metric`);
    
    // Enhanced weighting with model type consideration
    const minWeight = 0.1; // Minimum weight to prevent zero influence
    const maxWeight = 0.8; // Maximum weight to prevent dominance
    
    // Normalize accuracies for comparison
    const totalAccuracy = rfAccuracy + lstmAccuracy;
    
    if (totalAccuracy > 0) {
      let rfWeight = Math.max(minWeight, rfAccuracy / totalAccuracy);
      let lstmWeight = Math.max(minWeight, lstmAccuracy / totalAccuracy);
      
      // Apply max weight constraints
      rfWeight = Math.min(maxWeight, rfWeight);
      lstmWeight = Math.min(maxWeight, lstmWeight);
      
      // Renormalize to sum to 1
      const totalWeight = rfWeight + lstmWeight;
      this.config.randomForestWeight = rfWeight / totalWeight;
      this.config.lstmWeight = lstmWeight / totalWeight;
    } else {
      // Fallback to default weights if no valid accuracies
      this.config.randomForestWeight = 0.4;
      this.config.lstmWeight = 0.6;
    }
    
    console.log('🎯 Calculated Optimal Weights:');
    console.log(`  Random Forest: ${(this.config.randomForestWeight * 100).toFixed(1)}%`);
    console.log(`  LSTM: ${(this.config.lstmWeight * 100).toFixed(1)}%`);
  }
  
  // Enhanced ensemble prediction with proper model handling
  async predictLongTermEnsemble(stockData) {
    if (!this.trained) {
      throw new Error('Ensemble not trained yet!');
    }
    
    console.log(`🔮 Making ${this.config.targetDaysAhead}-day ensemble prediction...`);
    
    const predictions = { randomForest: null, lstm: null };
    const confidences = { randomForest: 0, lstm: 0 };
    const priceTargets = { randomForest: null, lstm: null };
    const currentPrice = stockData[stockData.length - 1].close;
    
    // Random Forest Classification Prediction
    try {
      if (this.models.randomForest) {
        console.log('🌲 Making RF classification prediction...');
        
        // Prepare features for RF
        const formattedData = {
          close: stockData.map(d => d.close),
          high: stockData.map(d => d.high),
          low: stockData.map(d => d.low),
          volume: stockData.map(d => d.volume),
          dates: stockData.map(d => d.date)
        };
        
        const technicalFeatures = extractTechnicalFeatures(formattedData);
        
        // Use the same feature preparation as training
        const features = prepareLatestLongTermFeatures(
          stockData, 
          this.config.modelParams.randomForest.lookbackDays,
          this.featureSelector // Pass feature selector if available
        );
        
        console.log('🔍 RF prepared features length:', features.length);
        
        // Get classification prediction with confidence
        let classificationResult;
        
        if (this.models.randomForest.predictWithConfidence) {
          const rawResults = this.models.randomForest.predictWithConfidence([features]);
          classificationResult = rawResults[0];
          console.log('🔍 RF classification result:', classificationResult);
        } else if (this.models.randomForest.predict) {
          const rawPrediction = this.models.randomForest.predict([features]);
          classificationResult = { 
            prediction: rawPrediction[0],
            confidence: 0.6 // Default confidence
          };
        } else {
          throw new Error('RF model has no prediction method');
        }
        
        // Convert classification to return and price
        const category = Math.round(classificationResult.prediction);
        const confidence = convertClassificationConfidence(classificationResult);
        const percentReturn = convertCategoryToReturn(category, confidence);
        const predictedPrice = currentPrice * (1 + percentReturn);
        
        predictions.randomForest = percentReturn;
        priceTargets.randomForest = predictedPrice;
        confidences.randomForest = confidence;
        
        console.log(`🌲 RF Result: Category ${category} → ${(percentReturn * 100).toFixed(2)}% → $${predictedPrice.toFixed(2)}`);
      }
    } catch (error) {
      console.warn('⚠️ RF prediction failed:', error);
    }
    
    // LSTM Regression Prediction
    try {
      if (this.models.lstm && this.trainingResults.lstm?.normParams) {
        console.log('🧠 Making LSTM regression prediction...');
        
        const formattedData = {
          close: stockData.map(d => d.close),
          high: stockData.map(d => d.high),
          low: stockData.map(d => d.low),
          volume: stockData.map(d => d.volume),
          dates: stockData.map(d => d.date)
        };
        
        const technicalFeatures = extractTechnicalFeatures(formattedData);
        
        const result = await predictLongTermWithConfidence(
          this.models.lstm,
          technicalFeatures,
          this.trainingResults.lstm.normParams,
          this.trainingResults.lstm.sequenceLength,
          this.config.targetDaysAhead,
          20 // MC samples
        );
        
        if (result.length > 0) {
          const lstmResult = result[0];
          predictions.lstm = lstmResult.predictedReturn;
          priceTargets.lstm = lstmResult.predictedPrice;
          confidences.lstm = lstmResult.confidence;
          
          console.log(`🧠 LSTM Result: ${(lstmResult.predictedReturn * 100).toFixed(2)}% → $${lstmResult.predictedPrice.toFixed(2)}`);
        }
      }
    } catch (error) {
      console.warn('⚠️ LSTM prediction failed:', error);
    }
    
    console.log('📊 Individual Predictions:');
    console.log('  Returns:', predictions);
    console.log('  Prices:', priceTargets);
    console.log('  Confidences:', confidences);
    
    // Calculate weighted ensemble prediction
    const ensemblePrediction = this.calculateWeightedLongTermPrediction(
      predictions, 
      priceTargets, 
      confidences, 
      currentPrice
    );
    
    return ensemblePrediction;
  }
  
  // Enhanced weighted prediction calculation
  calculateWeightedLongTermPrediction(predictions, priceTargets, confidences, currentPrice) {
    let weightedReturnSum = 0;
    let weightedPriceSum = 0;
    let totalWeight = 0;
    let validPredictions = 0;
    
    const modelNames = ['randomForest', 'lstm'];
    const modelWeights = [this.config.randomForestWeight, this.config.lstmWeight];
    
    // Confidence-adjusted weighted averaging
    modelNames.forEach((modelName, index) => {
      if (predictions[modelName] !== null && priceTargets[modelName] !== null) {
        const confidence = confidences[modelName] || 0.5;
        const baseWeight = modelWeights[index];
        
        // Dynamic weight adjustment based on confidence
        const confidenceMultiplier = 0.5 + (confidence * 0.5); // 0.5 to 1.0 range
        const adjustedWeight = baseWeight * confidenceMultiplier;
        
        weightedReturnSum += predictions[modelName] * adjustedWeight;
        weightedPriceSum += priceTargets[modelName] * adjustedWeight;
        totalWeight += adjustedWeight;
        validPredictions++;
        
        console.log(`📊 ${modelName}: weight=${baseWeight.toFixed(2)}, confidence=${confidence.toFixed(2)}, adjusted=${adjustedWeight.toFixed(2)}`);
      }
    });
    
    // Calculate ensemble predictions
    const ensembleReturn = totalWeight > 0 ? weightedReturnSum / totalWeight : 0;
    const ensemblePrice = totalWeight > 0 ? weightedPriceSum / totalWeight : currentPrice;
    
    // Calculate ensemble confidence (weighted average)
    const ensembleConfidence = validPredictions > 0 ? 
      Object.values(confidences).filter(c => c > 0).reduce((sum, c) => sum + c, 0) / validPredictions : 0;
    
    // Enhanced uncertainty estimation
    const validPrices = Object.values(priceTargets).filter(p => p !== null);
    let priceStdDev = 0;
    
    if (validPrices.length > 1) {
      const meanPrice = validPrices.reduce((sum, p) => sum + p, 0) / validPrices.length;
      priceStdDev = Math.sqrt(
        validPrices.reduce((sum, p) => sum + Math.pow(p - meanPrice, 2), 0) / validPrices.length
      );
    } else {
      // Single model uncertainty based on confidence
      priceStdDev = Math.abs(ensemblePrice - currentPrice) * (1 - ensembleConfidence) * 0.5;
    }
    
    // Consensus analysis
    const returnThreshold = 0.01; // 1%
    const upVotes = Object.values(predictions).filter(p => p !== null && p > returnThreshold).length;
    const downVotes = Object.values(predictions).filter(p => p !== null && p < -returnThreshold).length;
    const flatVotes = validPredictions - upVotes - downVotes;
    
    const consensus = validPredictions > 0 ? Math.max(upVotes, downVotes, flatVotes) / validPredictions : 0;
    
    // Agreement level
    let agreementLevel = 'Low';
    if (consensus >= 1.0) agreementLevel = 'Perfect';
    else if (consensus >= 0.5) agreementLevel = 'Medium';
    
    // Direction determination
    let direction = 'FLAT';
    if (upVotes > Math.max(downVotes, flatVotes)) direction = 'UP';
    else if (downVotes > Math.max(upVotes, flatVotes)) direction = 'DOWN';
    
    // Target date calculation
    const targetDate = new Date();
    targetDate.setDate(targetDate.getDate() + this.config.targetDaysAhead);
    
    console.log(`🎯 Ensemble Result: ${direction} ${(ensembleReturn * 100).toFixed(2)}% → $${ensemblePrice.toFixed(2)}`);
    
    return {
      // Core prediction
      predictedPrice: ensemblePrice,
      currentPrice: currentPrice,
      expectedReturn: ensembleReturn * 100, // Convert to percentage
      priceChange: ensemblePrice - currentPrice,
      direction: direction,
      
      // Confidence and consensus
      confidence: ensembleConfidence,
      consensus: consensus,
      agreementLevel: agreementLevel,
      
      // Uncertainty estimation
      priceRange: [
        Math.max(0, ensemblePrice - 1.96 * priceStdDev),
        ensemblePrice + 1.96 * priceStdDev
      ],
      standardDeviation: priceStdDev,
      
      // Meta information
      validModels: validPredictions,
      totalModels: 2,
      targetDate: targetDate.toLocaleDateString(),
      daysAhead: this.config.targetDaysAhead,
      timestamp: new Date(),
      
      // Individual model results
      individualPredictions: {
        randomForest: {
          return: predictions.randomForest ? (predictions.randomForest * 100).toFixed(2) + '%' : 'N/A',
          price: priceTargets.randomForest ? '$' + priceTargets.randomForest.toFixed(2) : 'N/A',
          confidence: confidences.randomForest ? (confidences.randomForest * 100).toFixed(1) + '%' : 'N/A',
          type: 'Classification → Return'
        },
        lstm: {
          return: predictions.lstm ? (predictions.lstm * 100).toFixed(2) + '%' : 'N/A',
          price: priceTargets.lstm ? '$' + priceTargets.lstm.toFixed(2) : 'N/A',
          confidence: confidences.lstm ? (confidences.lstm * 100).toFixed(1) + '%' : 'N/A',
          type: 'Regression'
        }
      },
      
      // Voting breakdown
      votingBreakdown: {
        upVotes: upVotes,
        downVotes: downVotes,
        flatVotes: flatVotes,
        abstains: 2 - validPredictions
      },
      
      // Model weights
      modelWeights: {
        randomForest: this.config.randomForestWeight,
        lstm: this.config.lstmWeight
      }
    };
  }
  
  // Update prediction timeframe
  updatePredictionHorizon(newTargetDaysAhead) {
    this.config.targetDaysAhead = newTargetDaysAhead;
    this.config.modelParams.randomForest.targetDaysAhead = Math.max(newTargetDaysAhead, 22); // RF needs longer horizon
    this.config.modelParams.lstm.targetDaysAhead = newTargetDaysAhead;
    
    this.trained = false;
    console.log(`📅 Updated prediction horizon to ${newTargetDaysAhead} days ahead`);
  }
  
  // Enhanced ensemble validation
  async validateLongTermEnsemble(stockData) {
    const trainedModels = this.getTrainedModelCount();
    const avgAccuracy = this.getAverageAccuracy();
    
    return {
      trainedModels: trainedModels,
      totalModels: 2,
      averageAccuracy: avgAccuracy,
      ensembleReady: trainedModels >= 1,
      predictionHorizon: this.config.targetDaysAhead,
      recommendedConfidenceThreshold: this.getRecommendedThreshold(),
      modelTypes: {
        randomForest: 'Classification',
        lstm: 'Regression'
      }
    };
  }
  
  // Helper methods
  getTrainedModelCount() {
    let count = 0;
    if (this.models.randomForest) count++;
    if (this.models.lstm) count++;
    return count;
  }
  
  getAverageAccuracy() {
    const accuracies = [];
    
    if (this.trainingResults.randomForest?.accuracy && !this.trainingResults.randomForest?.error) {
      accuracies.push(this.trainingResults.randomForest.accuracy);
    }
    if (this.trainingResults.lstm?.accuracy && !this.trainingResults.lstm?.error) {
      accuracies.push(this.trainingResults.lstm.accuracy);
    }
    
    return accuracies.length > 0 ? 
      accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length : 0;
  }
  
  getRecommendedThreshold() {
    const avgAccuracy = this.getAverageAccuracy();
    if (avgAccuracy > 0.7) return 0.6;
    if (avgAccuracy > 0.6) return 0.7;
    return 0.8;
  }
  
  // Enhanced ensemble status
  getEnsembleStatus() {
    return {
      trained: this.trained,
      trainedModels: this.getTrainedModelCount(),
      totalModels: 2,
      averageAccuracy: this.getAverageAccuracy(),
      predictionHorizon: this.config.targetDaysAhead,
      weights: {
        randomForest: this.config.randomForestWeight,
        lstm: this.config.lstmWeight
      },
      modelTypes: {
        randomForest: 'Classification (5 categories)',
        lstm: 'Regression (log-returns)'
      },
      lastTrainingDate: this.lastTrainingData ? new Date() : null
    };
  }
}

// Factory and convenience functions
export const createLongTermEnsemblePredictor = (config = {}) => {
  return new LongTermEnsemblePredictor(config);
};

export const trainLongTermEnsembleModel = async (stockData, config = {}, progressCallback = null) => {
  const ensemble = createLongTermEnsemblePredictor(config);
  const result = await ensemble.trainLongTermEnsemble(stockData, progressCallback);
  
  return {
    ensemble: ensemble,
    trainingResult: result
  };
};

export const predictWithLongTermEnsemble = async (ensemble, stockData) => {
  if (!ensemble || !ensemble.trained) {
    throw new Error('Ensemble model not trained');
  }
  
  return await ensemble.predictLongTermEnsemble(stockData);
};