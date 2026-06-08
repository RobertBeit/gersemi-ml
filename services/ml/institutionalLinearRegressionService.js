// Institutional-Grade Linear Regression Service (FIXED)
// src/services/institutionalLinearRegressionService.js
// Implements CAPM, Fama-French, and Multi-Factor models like professional financial institutions

import * as tf from '@tensorflow/tfjs';
import { fetchStockData } from './api'; // Use our new multi-provider API service
import { logger } from './fileLogger'; // File logger for debugging

// Market Factor Data Manager
class MarketFactorDataManager {
  constructor() {
    this.factorData = null;
    this.lastUpdate = null;
    this.cache = new Map();
  }

  // Normalize date format to YYYY-MM-DD
  normalizeDate(date) {
    if (typeof date === 'string') {
      // Handle different date formats
      const parts = date.split('-');
      if (parts.length === 3) {
        const year = parts[0];
        const month = parts[1].padStart(2, '0');
        const day = parts[2].padStart(2, '0');
        return `${year}-${month}-${day}`;
      }
    }
    if (date instanceof Date) {
      return date.toISOString().split('T')[0];
    }
    return date;
  }

  // Fetch all required market factor data
  async fetchMarketFactors(startDate, endDate) {
    logger.info('🏦 Fetching institutional market factor data...');
    logger.info('📅 Date range: ' + startDate + ' to ' + endDate);
    logger.info('📝 Using ETF proxies for market factors via multi-provider API');
    
    // Test if fetchStockData is working
    logger.info('🧪 Testing API with SPY first...');
    try {
      logger.info('🧪 Calling fetchStockData("SPY", "' + startDate + '", "' + endDate + '")...');
      const testData = await fetchStockData('SPY', startDate, endDate);
      logger.info('🧪 SPY test result: ' + (testData ? testData.length + ' data points' : 'NULL'));
      if (testData && testData.length > 0) {
        logger.info('🧪 Sample SPY data: ' + JSON.stringify(testData[0]));
      } else {
        logger.warn('⚠️ SPY returned empty array: testData = ' + JSON.stringify(testData));
      }
    } catch (testError) {
      logger.error('🧪 SPY test FAILED: ' + testError.message);
      logger.error('Stack: ' + testError.stack);
      // Continue anyway to test other factors
    }
    
    const factorSymbols = {
      market: 'SPY',        // SPDR S&P 500 ETF (Market Factor)
      risk_free: 'TLT',     // 20+ Year Treasury Bond ETF (Risk-free proxy)
      volatility: 'UVXY',   // VIX ETF (Volatility Factor proxy)
      small_cap: 'IWM',     // Russell 2000 (Small Cap Factor)
      tech: 'XLK',          // Technology Sector
      finance: 'XLF',       // Financial Sector
      healthcare: 'XLV',    // Healthcare Sector
      energy: 'XLE',        // Energy Sector
      consumer: 'XLY',      // Consumer Discretionary
      staples: 'XLP',       // Consumer Staples
      utilities: 'XLU',     // Utilities
      real_estate: 'XLRE',  // Real Estate
      materials: 'XLB',     // Materials
      industrials: 'XLI'    // Industrials
    };

    const factorData = {};
    
    // Fetch data for each factor SEQUENTIALLY to avoid rate limits
    logger.info('📊 Fetching factors one at a time to respect API rate limits...');
    
    for (const [factorName, symbol] of Object.entries(factorSymbols)) {
      try {
        logger.info('📊 Fetching ' + factorName + ' data (' + symbol + ')...');
        
        // Use our new multi-provider API service
        const historicalData = await fetchStockData(symbol, startDate, endDate);
        
        if (historicalData && historicalData.length > 0) {
          // Data is already sorted by our API service
          const returns = this.calculateReturns(historicalData);
          
          factorData[factorName] = {
            prices: historicalData,
            returns: returns,
            symbol: symbol
          };
          
          logger.info('✅ ' + factorName + ': ' + returns.length + ' return data points');
        } else {
          logger.error('❌ No data returned for ' + factorName + ' (' + symbol + ') - API returned empty array');
          factorData[factorName] = null;
        }
        
        // Wait 2 seconds between requests to give the data backend recovery time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
      } catch (error) {
        logger.error('❌ Failed to fetch ' + factorName + ' (' + symbol + '): ' + error.message);
        logger.error('Stack: ' + error.stack);
        factorData[factorName] = null;
      }
    }
    
    this.factorData = factorData;
    this.lastUpdate = new Date();
    
    logger.info('🏦 Market factor data fetch completed');
    return factorData;
  }

  // Calculate daily returns from price data
  calculateReturns(priceData) {
    const returns = [];
    
    for (let i = 1; i < priceData.length; i++) {
      // Our new API service provides 'close' as the adjusted close
      const currentPrice = priceData[i].close;
      const previousPrice = priceData[i - 1].close;
      
      if (currentPrice && previousPrice && previousPrice > 0) {
        const dailyReturn = (currentPrice - previousPrice) / previousPrice;
        returns.push({
          date: this.normalizeDate(priceData[i].date),
          return: dailyReturn,
          price: currentPrice
        });
      }
    }
    
    return returns;
  }

  // Get factor data by date (aligned) - FIXED VERSION
  getAlignedFactorData(stockReturns) {
    if (!this.factorData) {
      throw new Error('Market factor data not loaded');
    }

    const alignedData = [];
    
    // Debug: Check data availability
    logger.info('📊 DEBUG: Checking factor data availability...');
    Object.entries(this.factorData).forEach(([name, data]) => {
      if (data?.returns) {
        logger.info(`${name}: ${data.returns.length} returns, sample dates: ${data.returns.slice(0, 3).map(r => r.date).join(', ')}`);
      } else {
        logger.warn(`${name}: NO DATA`);
      }
    });

    // Debug: Check stock data
    logger.info(`📊 DEBUG: Stock returns sample dates: ${stockReturns.slice(0, 3).map(r => this.normalizeDate(r.date)).join(', ')}`);

    // Create date-aligned dataset
    let skippedRows = 0;
    for (const stockReturn of stockReturns) {
      const normalizedDate = this.normalizeDate(stockReturn.date);
      const factorRow = { 
        date: normalizedDate, 
        stock_return: stockReturn.return 
      };

      // Market factor (using SPY instead of ^GSPC)
      const marketReturn = this.getFactorReturnByDate('market', normalizedDate);
      
      // Debug first few iterations
      if (stockReturns.indexOf(stockReturn) < 3) {
        logger.info(`🔍 Alignment check - Date: ${normalizedDate}, Market return: ${marketReturn}`);
      }
      
      // Use Treasury bond ETF (TLT) as risk-free proxy, but treat it differently
      let riskFreeRate = this.getFactorReturnByDate('risk_free', normalizedDate);
      if (riskFreeRate === null) {
        // Fallback: use a fixed daily risk-free rate (e.g., 4% annual = ~0.0001 daily)
        riskFreeRate = 0.04 / 252; // 4% annual converted to daily
      } else {
        // TLT returns are bond returns, not rates, so use a smaller portion
        // and apply inverse relationship (when rates go up, bond prices go down)
        riskFreeRate = Math.abs(riskFreeRate) * 0.1; // Scale down bond volatility
        if (riskFreeRate > 0.001) riskFreeRate = 0.001; // Cap at reasonable daily rate
      }
      
      if (marketReturn !== null) {
        factorRow.market_factor = marketReturn - riskFreeRate;
        factorRow.stock_excess_return = stockReturn.return - riskFreeRate;
        factorRow.risk_free_rate = riskFreeRate;
        factorRow.market_return = marketReturn;

        // Size factor (Small cap - Large cap)
        const smallCapReturn = this.getFactorReturnByDate('small_cap', normalizedDate);
        if (smallCapReturn !== null) {
          factorRow.size_factor = smallCapReturn - marketReturn; // SMB (Small Minus Big)
        }

        // Volatility factor (using UVXY as VIX proxy)
        const volatilityData = this.factorData.volatility?.returns?.find(r => r.date === normalizedDate);
        if (volatilityData) {
          factorRow.volatility_factor = volatilityData.return;
          // For VIX level, we'll estimate from UVXY price (scaled approximation)
          factorRow.vix_level = Math.max(10, Math.min(50, volatilityData.price * 2)); // Rough VIX estimate
        } else {
          // Fallback: use market volatility as proxy
          if (marketReturn !== null) {
            factorRow.volatility_factor = Math.abs(marketReturn) * 2; // Market vol proxy
            factorRow.vix_level = 20; // Default VIX level
          }
        }

        // Sector factors
        const sectors = ['tech', 'finance', 'healthcare', 'energy', 'consumer', 'staples', 'utilities', 'real_estate', 'materials', 'industrials'];
        let sectorCount = 0;
        for (const sector of sectors) {
          const sectorReturn = this.getFactorReturnByDate(sector, normalizedDate);
          if (sectorReturn !== null) {
            factorRow[`${sector}_factor`] = sectorReturn - marketReturn; // Sector excess return
            sectorCount++;
          }
        }

        // Only include rows with basic market data (relaxed requirements)
        if (factorRow.market_factor !== undefined && factorRow.stock_excess_return !== undefined) {
          alignedData.push(factorRow);
        }
      }
    }

    console.log(`📊 Aligned ${alignedData.length} data points with market factors`);
    logger.info(`📊 Aligned ${alignedData.length} data points with market factors`);
    logger.info(`⚠️ Skipped ${stockReturns.length - alignedData.length} rows (${Math.round(((stockReturns.length - alignedData.length) / stockReturns.length) * 100)}%)`);
    
    // Debug: Show sample of aligned data
    if (alignedData.length > 0) {
      console.log('📊 Sample aligned data:', alignedData.slice(0, 2));
      logger.info('📊 Sample aligned data: ' + JSON.stringify(alignedData.slice(0, 2)));
    } else {
      console.error('❌ No data aligned - debugging...');
      logger.error('❌ No data aligned - debugging...');
      // Show first few dates from each dataset for debugging
      const stockDates = stockReturns.slice(0, 5).map(r => this.normalizeDate(r.date));
      const marketDates = this.factorData.market?.returns?.slice(0, 5).map(r => r.date) || [];
      console.log('Stock dates sample:', stockDates);
      console.log('Market dates sample:', marketDates);
      logger.error('Stock dates sample: ' + JSON.stringify(stockDates));
      logger.error('Market dates sample: ' + JSON.stringify(marketDates));
    }
    
    return alignedData;
  }

  // Helper function to get factor return by date - IMPROVED
  getFactorReturnByDate(factorName, normalizedDate) {
    const factorData = this.factorData[factorName];
    if (!factorData?.returns) return null;

    const returnData = factorData.returns.find(r => r.date === normalizedDate);
    return returnData ? returnData.return : null;
  }
}

// Institutional Linear Regression Models - ENHANCED
class InstitutionalLinearRegression {
  constructor(modelType = 'multi_factor') {
    this.modelType = modelType; // 'capm', 'fama_french_3', 'multi_factor'
    this.model = null;
    this.trained = false;
    this.factors = null;
    this.coefficients = null;
    this.diagnostics = null;
    // Removed normalizationParams - not using normalization
  }

  // Prepare features based on model type - ENHANCED
  prepareFeatures(alignedData, modelType = this.modelType) {
    const features = [];
    const targets = [];
    const dates = [];
    const featureNames = [];

    for (const row of alignedData) {
      const featureRow = [];
      
      // All models include market factor
      if (row.market_factor !== undefined && !isNaN(row.market_factor)) {
        featureRow.push(row.market_factor);
        if (featureNames.length === 0) featureNames.push('Market_Factor');
      } else {
        continue; // Skip this row if no market factor
      }

      if (modelType === 'capm') {
        // CAPM: Only market factor
        // Features already added above
      } 
      else if (modelType === 'fama_french_3') {
        // Fama-French 3-Factor: Market + Size + Value
        if (row.size_factor !== undefined && !isNaN(row.size_factor)) {
          featureRow.push(row.size_factor);
          if (featureNames.length <= 1) featureNames.push('Size_Factor');
        } else {
          featureRow.push(0); // Use 0 if size factor not available
          if (featureNames.length <= 1) featureNames.push('Size_Factor');
        }
        
        // Value factor approximation using sector rotation
        const valueProxy = (row.finance_factor || 0) - (row.tech_factor || 0);
        featureRow.push(valueProxy);
        if (featureNames.length <= 2) featureNames.push('Value_Factor_Proxy');
      }
      else if (modelType === 'multi_factor') {
        // Multi-Factor Model: All available factors
        
        // Size factor
        featureRow.push(row.size_factor || 0);
        if (featureNames.length <= 1) featureNames.push('Size_Factor');

        // Volatility factor
        featureRow.push(row.volatility_factor || 0);
        if (featureNames.length <= 2) featureNames.push('Volatility_Factor');

        // VIX level (risk sentiment)
        if (row.vix_level !== undefined && !isNaN(row.vix_level)) {
          featureRow.push(row.vix_level / 100); // Normalize VIX
        } else {
          featureRow.push(0.2); // Default VIX level
        }
        if (featureNames.length <= 3) featureNames.push('VIX_Level');

        // Key sector factors (only include the most important ones)
        const keyFactors = ['tech_factor', 'finance_factor', 'healthcare_factor'];
        for (let i = 0; i < keyFactors.length; i++) {
          const factor = keyFactors[i];
          featureRow.push(row[factor] || 0);
          if (featureNames.length === 4 + i) { // Ensure unique feature names
            featureNames.push(factor.replace('_factor', '_Factor'));
          }
        }

        // Momentum factor (simple approximation) - only add once
        const rowIndex = alignedData.indexOf(row);
        if (rowIndex >= 20) {
          const recentReturns = alignedData.slice(rowIndex - 20, rowIndex);
          const momentum = recentReturns.reduce((sum, r) => sum + (r.stock_return || 0), 0) / 20;
          featureRow.push(momentum);
        } else {
          featureRow.push(0);
        }
        if (featureNames.length === 7) featureNames.push('Momentum_Factor'); // Only add once
      }

      // Only add if we have valid stock return and features
      if (featureRow.length === featureNames.length && 
          row.stock_return !== undefined && 
          !isNaN(row.stock_return)) {
        features.push(featureRow);
        targets.push(row.stock_return); // Use regular stock returns instead of excess returns
        dates.push(row.date);
      }
    }

    console.log(`📊 Prepared ${features.length} samples with ${featureNames.length} factors for ${modelType} model`);
    console.log(`📋 Factors: ${featureNames.join(', ')}`);

    return { features, targets, dates, featureNames };
  }

  // Train the institutional model - ENHANCED
  async train(alignedData, modelType = this.modelType) {
    console.log(`🏦 Training ${modelType.toUpperCase()} model...`);
    logger.info(`🏦 Training ${modelType.toUpperCase()} model...`);

    const { features, targets, dates, featureNames } = this.prepareFeatures(alignedData, modelType);
    
    if (features.length === 0) {
      logger.error('❌ No valid data for training institutional model');
      throw new Error('No valid data for training institutional model');
    }

    console.log(`📊 Training with ${features.length} samples`);
    logger.info(`📊 Training with ${features.length} samples using ${featureNames.length} factors`);
    this.factors = featureNames;
    
    // Check feature ranges before training
    const featureMeans = [];
    const featureStds = [];
    for (let j = 0; j < features[0].length; j++) {
      const column = features.map(row => row[j]);
      const mean = column.reduce((sum, val) => sum + val, 0) / column.length;
      const std = Math.sqrt(column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / column.length);
      featureMeans.push(mean);
      featureStds.push(std);
      console.log(`📊 Feature ${j} (${featureNames[j]}): mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
    }
    
    // Create TensorFlow model with SIMPLE LINEAR architecture (NO NORMALIZATION)
    this.model = tf.sequential({
      layers: [
        tf.layers.dense({
          units: 1,
          inputShape: [features[0].length],
          activation: 'linear', // Pure linear regression
          kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), // Lighter regularization
          kernelInitializer: 'zeros', // Start with zero weights
          biasInitializer: 'zeros',
          name: 'linear_regression'
        })
      ]
    });

    // Compile with appropriate optimizer for financial data
    this.model.compile({
      optimizer: tf.train.adam(0.001), // Much smaller learning rate
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    // Prepare tensors (NO NORMALIZATION)
    const xTensor = tf.tensor2d(features);
    const yTensor = tf.tensor2d(targets, [targets.length, 1]);

    // Train model with very conservative settings
    const history = await this.model.fit(xTensor, yTensor, {
      epochs: 30, // Even fewer epochs
      batchSize: Math.min(128, Math.floor(features.length / 2)), // Larger batch size
      validationSplit: 0.1, // Smaller validation split
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 5 === 0) {
            console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(6)}, val_loss=${logs.val_loss.toFixed(6)}, mae=${logs.mae.toFixed(6)}`);
          }
        }
      }
    });

    // Get model coefficients (from the single linear layer)
    const weights = this.model.layers[0].getWeights()[0].dataSync();
    const bias = this.model.layers[0].getWeights()[1].dataSync()[0];

    this.coefficients = {
      weights: Array.from(weights),
      bias: bias,
      factors: featureNames
    };

    console.log(`📊 Final model weights: ${Array.from(weights).map(w => w.toFixed(4)).join(', ')}`);
    console.log(`📊 Final model bias: ${bias.toFixed(4)}`);

    // Mark as trained BEFORE calculating diagnostics (which calls predict)
    this.trained = true;

    // Calculate diagnostics (NO NORMALIZATION)
    this.diagnostics = await this.calculateDiagnostics(features, targets);

    // Clean up tensors
    xTensor.dispose();
    yTensor.dispose();
    
    console.log(`✅ ${modelType.toUpperCase()} model trained successfully`);
    console.log(`📊 Model R²: ${this.diagnostics.r2.toFixed(3)}`);
    console.log(`📊 Adjusted R²: ${this.diagnostics.adjustedR2.toFixed(3)}`);
    
    logger.info(`✅ ${modelType.toUpperCase()} model trained successfully`);
    logger.info(`📊 Model R²: ${this.diagnostics.r2.toFixed(3)} (Quality: ${this.diagnostics.significance.r2Category})`);
    logger.info(`📊 Adjusted R²: ${this.diagnostics.adjustedR2.toFixed(3)}`);
    logger.info(`📊 RMSE: ${this.diagnostics.rmse.toFixed(6)}, MAE: ${this.diagnostics.mae.toFixed(6)}`);
    logger.info(`📊 Model is ready for predictions!`);
    
    return this;
  }

  // Normalize features for stable training
  async normalizeFeatures(features) {
    const featureTensor = tf.tensor2d(features);
    
    // Calculate normalization parameters
    const mean = featureTensor.mean(0);
    const std = tf.sqrt(featureTensor.sub(mean).square().mean(0));
    const epsilon = tf.scalar(1e-8);
    
    // Normalize
    const normalizedTensor = featureTensor.sub(mean).div(std.add(epsilon));
    
    // Store parameters
    this.normalizationParams = {
      mean: await mean.data(),
      std: await std.data()
    };
    
    const normalizedData = await normalizedTensor.data();
    
    // Clean up tensors
    featureTensor.dispose();
    mean.dispose();
    std.dispose();
    epsilon.dispose();
    normalizedTensor.dispose();
    
    // Convert back to 2D array
    const normalizedFeatures = [];
    const numFeatures = features[0].length;
    for (let i = 0; i < features.length; i++) {
      const row = [];
      for (let j = 0; j < numFeatures; j++) {
        row.push(normalizedData[i * numFeatures + j]);
      }
      normalizedFeatures.push(row);
    }
    
    return normalizedFeatures;
  }

  // Calculate comprehensive diagnostics
  async calculateDiagnostics(features, targets) {
    console.log(`📊 Calculating diagnostics for ${targets.length} samples`);
    console.log(`📊 Target range: ${Math.min(...targets).toFixed(4)} to ${Math.max(...targets).toFixed(4)}`);
    console.log(`📊 Target mean: ${(targets.reduce((sum, val) => sum + val, 0) / targets.length).toFixed(4)}`);
    
    // Make predictions
    const predictions = await this.predict(features);
    console.log(`📊 Prediction range: ${Math.min(...predictions).toFixed(4)} to ${Math.max(...predictions).toFixed(4)}`);
    
    // Calculate R²
    const meanTarget = targets.reduce((sum, val) => sum + val, 0) / targets.length;
    const totalSumSquares = targets.reduce((sum, val) => sum + Math.pow(val - meanTarget, 2), 0);
    const residualSumSquares = targets.reduce((sum, actual, i) => sum + Math.pow(actual - predictions[i], 2), 0);
    
    console.log(`📊 Total Sum of Squares: ${totalSumSquares.toFixed(6)}`);
    console.log(`📊 Residual Sum of Squares: ${residualSumSquares.toFixed(6)}`);
    
    const r2 = Math.max(-1000, 1 - (residualSumSquares / totalSumSquares)); // Cap extreme negative values
    const adjustedR2 = Math.max(-1000, 1 - ((1 - r2) * (targets.length - 1)) / (targets.length - features[0].length - 1));
    
    // Calculate other metrics
    const mae = targets.reduce((sum, actual, i) => sum + Math.abs(actual - predictions[i]), 0) / targets.length;
    const rmse = Math.sqrt(residualSumSquares / targets.length);
    
    // F-statistic
    const msr = (totalSumSquares - residualSumSquares) / features[0].length;
    const mse = residualSumSquares / (targets.length - features[0].length - 1);
    const fStatistic = mse > 0 ? msr / mse : 0;
    
    // Durbin-Watson test for autocorrelation
    let durbinWatson = 0;
    const residuals = targets.map((actual, i) => actual - predictions[i]);
    
    for (let i = 1; i < residuals.length; i++) {
      durbinWatson += Math.pow(residuals[i] - residuals[i-1], 2);
    }
    durbinWatson = residualSumSquares > 0 ? durbinWatson / residualSumSquares : 2;

    return {
      r2: r2,
      adjustedR2: adjustedR2,
      mae: mae,
      rmse: rmse,
      fStatistic: fStatistic,
      durbinWatson: durbinWatson,
      sampleSize: targets.length,
      numFactors: features[0].length,
      significance: this.assessSignificance(r2, fStatistic, targets.length, features[0].length)
    };
  }

  // Assess statistical significance
  assessSignificance(r2, fStatistic, n, p) {
    // Simple significance assessment
    const criticalF = 2.5; // Approximate F-critical for typical scenarios
    const isSignificant = fStatistic > criticalF;
    
    let interpretation = '';
    if (r2 > 0.7) interpretation = 'Strong explanatory power';
    else if (r2 > 0.4) interpretation = 'Moderate explanatory power';
    else if (r2 > 0.2) interpretation = 'Weak but meaningful explanatory power';
    else interpretation = 'Poor explanatory power';

    return {
      isSignificant: isSignificant,
      fStatistic: fStatistic,
      interpretation: interpretation,
      r2Category: r2 > 0.4 ? 'Good' : r2 > 0.2 ? 'Acceptable' : 'Poor'
    };
  }

  // Make predictions (NO NORMALIZATION)
  async predict(features) {
    if (!this.trained) {
      throw new Error('Model not trained');
    }

    // Predict directly on raw features (no normalization)
    const featureTensor = tf.tensor2d(features);
    const predictions = this.model.predict(featureTensor);
    const predictionValues = await predictions.data();
    
    // Clean up
    featureTensor.dispose();
    predictions.dispose();
    
    // Apply reasonable bounds for daily stock returns (-50% to +50%)
    const boundedPredictions = Array.from(predictionValues).map(pred => {
      return Math.max(-0.5, Math.min(0.5, pred));
    });
    
    return boundedPredictions;
  }

  // Remove normalization methods (not needed anymore)
  async normalizeFeatures(features) {
    // Not used - return features as-is
    return features;
  }

  // Apply normalization to new features (not needed anymore)
  applyNormalization(features) {
    // Not used - return features as-is  
    return features;
  }

  // Get factor loadings (betas)
  getFactorLoadings() {
    if (!this.coefficients) return null;

    return this.coefficients.factors.map((factor, idx) => ({
      factor: factor,
      beta: this.coefficients.weights[idx],
      interpretation: this.interpretBeta(factor, this.coefficients.weights[idx])
    }));
  }

  // Interpret beta coefficients
  interpretBeta(factor, beta) {
    const absBeta = Math.abs(beta);
    let strength = absBeta > 0.5 ? 'High' : absBeta > 0.2 ? 'Moderate' : 'Low';
    let direction = beta > 0 ? 'Positive' : 'Negative';
    
    let interpretation = `${strength} ${direction.toLowerCase()} sensitivity to ${factor.replace('_', ' ')}`;
    
    if (factor === 'Market_Factor') {
      if (beta > 1) interpretation += ' (High beta - more volatile than market)';
      else if (beta < 1) interpretation += ' (Low beta - less volatile than market)';
    }
    
    return interpretation;
  }
}

// Main service functions - ENHANCED
export const createInstitutionalLinearRegression = async (stockData, modelType = 'multi_factor', startDate, endDate) => {
  console.log('🏦 Creating institutional linear regression model...');
  logger.info('🏦 Creating institutional linear regression model...');
  
  // Initialize market factor manager
  const factorManager = new MarketFactorDataManager();
  
  // Fetch market factor data
  const marketFactors = await factorManager.fetchMarketFactors(startDate, endDate);
  
  // Calculate stock returns with normalized dates
  const stockReturns = [];
  for (let i = 1; i < stockData.length; i++) {
    const currentPrice = stockData[i].close;
    const previousPrice = stockData[i - 1].close;
    
    if (currentPrice && previousPrice && previousPrice > 0) {
      stockReturns.push({
        date: stockData[i].date, // Will be normalized in getAlignedFactorData
        return: (currentPrice - previousPrice) / previousPrice
      });
    }
  }
  
  console.log(`📊 Calculated ${stockReturns.length} stock return data points`);
  logger.info(`📊 Calculated ${stockReturns.length} stock return data points`);
  
  // Align stock data with market factors
  const alignedData = factorManager.getAlignedFactorData(stockReturns);
  
  logger.info(`⚠️ CRITICAL: alignedData length = ${alignedData.length}`);
  logger.info(`⚠️ CRITICAL: modelType = ${modelType}`);
  
  if (alignedData.length === 0) {
    logger.error('❌ CRITICAL ERROR: No aligned data available - aborting model creation');
    throw new Error('No aligned data available for institutional model - check date formats and data availability');
  }
  
  console.log(`✅ Successfully aligned ${alignedData.length} data points`);
  logger.info(`✅ Successfully aligned ${alignedData.length} data points`);
  
  // Create and train model
  const model = new InstitutionalLinearRegression(modelType);
  await model.train(alignedData, modelType);
  
  logger.info('🏦 Model training completed');
  
  return {
    model: model,
    alignedData: alignedData,
    factorManager: factorManager,
    diagnostics: model.diagnostics,
    factorLoadings: model.getFactorLoadings(),
    modelType: modelType
  };
};

// Compare different institutional models
export const compareInstitutionalModels = async (stockData, startDate, endDate) => {
  console.log('🏦 Comparing institutional linear regression models...');
  
  const modelTypes = ['capm', 'fama_french_3', 'multi_factor'];
  const results = {};
  
  // Initialize factor manager once
  const factorManager = new MarketFactorDataManager();
  const marketFactors = await factorManager.fetchMarketFactors(startDate, endDate);
  
  // Calculate stock returns once
  const stockReturns = [];
  for (let i = 1; i < stockData.length; i++) {
    const currentPrice = stockData[i].close;
    const previousPrice = stockData[i - 1].close;
    
    if (currentPrice && previousPrice && previousPrice > 0) {
      stockReturns.push({
        date: stockData[i].date,
        return: (currentPrice - previousPrice) / previousPrice
      });
    }
  }
  
  const alignedData = factorManager.getAlignedFactorData(stockReturns);
  
  if (alignedData.length === 0) {
    throw new Error('No aligned data available for institutional model comparison');
  }
  
  // Train each model type
  for (const modelType of modelTypes) {
    try {
      console.log(`🔄 Training ${modelType.toUpperCase()} model...`);
      
      const model = new InstitutionalLinearRegression(modelType);
      await model.train(alignedData, modelType);
      
      results[modelType] = {
        model: model,
        diagnostics: model.diagnostics,
        factorLoadings: model.getFactorLoadings(),
        success: true
      };
      
      console.log(`✅ ${modelType.toUpperCase()}: R² = ${model.diagnostics.r2.toFixed(3)}, Adj R² = ${model.diagnostics.adjustedR2.toFixed(3)}`);
      
    } catch (error) {
      console.error(`❌ ${modelType.toUpperCase()} failed:`, error.message);
      results[modelType] = {
        error: error.message,
        success: false
      };
    }
  }
  
  // Find best model (handle negative R² values)
  const validModels = Object.entries(results).filter(([_, result]) => result.success);
  const bestModel = validModels.reduce((best, [type, result]) => {
    const score = result.diagnostics.adjustedR2;
    // For negative R², choose the least negative (closest to 0)
    return score > (best.score || -Infinity) ? { type, result, score } : best;
  }, {});
  
  console.log(`🏆 Best model: ${bestModel.type ? bestModel.type.toUpperCase() : 'NONE'} (Adj R² = ${bestModel.score ? bestModel.score.toFixed(3) : 'N/A'})`);
  
  return {
    results: results,
    bestModel: bestModel,
    alignedData: alignedData,
    factorManager: factorManager,
    comparison: {
      totalModels: modelTypes.length,
      successfulModels: validModels.length,
      bestModelType: bestModel.type || 'none',
      bestAdjustedR2: bestModel.score || null
    }
  };
};

// Enhanced prediction function for institutional models
export const predictWithInstitutionalModel = async (model, factorManager, currentStockData, lookbackDays = 5) => {
  console.log('🔮 Making institutional model prediction...');
  
  // Get recent stock returns
  const recentReturns = [];
  for (let i = Math.max(1, currentStockData.length - lookbackDays); i < currentStockData.length; i++) {
    const currentPrice = currentStockData[i].close;
    const previousPrice = currentStockData[i - 1].close;
    
    if (currentPrice && previousPrice && previousPrice > 0) {
      recentReturns.push({
        date: currentStockData[i].date,
        return: (currentPrice - previousPrice) / previousPrice
      });
    }
  }
  
  if (recentReturns.length === 0) {
    throw new Error('Insufficient recent data for prediction');
  }
  
  // Get latest aligned factor data
  const alignedData = factorManager.getAlignedFactorData(recentReturns);
  
  if (alignedData.length === 0) {
    throw new Error('No recent factor data available');
  }
  
  // Use latest factor data for prediction
  const latestFactors = alignedData[alignedData.length - 1];
  const { features } = model.prepareFeatures([latestFactors], model.modelType);
  
  if (features.length === 0) {
    throw new Error('Cannot prepare features for prediction');
  }
  
  // Make prediction
  const predictions = await model.predict(features);
  const predictedExcessReturn = predictions[0];
  
  // Convert excess return to actual return
  const riskFreeRate = latestFactors.risk_free_rate || 0;
  const predictedReturn = predictedExcessReturn + riskFreeRate;
  
  const currentPrice = currentStockData[currentStockData.length - 1].close;
  const predictedPrice = currentPrice * (1 + predictedReturn);
  
  return {
    currentPrice: currentPrice,
    predictedReturn: predictedReturn,
    predictedExcessReturn: predictedExcessReturn,
    predictedPrice: predictedPrice,
    priceChange: predictedPrice - currentPrice,
    percentChange: predictedReturn * 100,
    riskFreeRate: riskFreeRate,
    confidence: Math.min(0.9, Math.max(0.1, model.diagnostics.adjustedR2)),
    factorLoadings: model.getFactorLoadings(),
    modelType: model.modelType,
    r2: model.diagnostics.r2,
    adjustedR2: model.diagnostics.adjustedR2
  };
};