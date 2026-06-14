const { fetchStockDataWithIntraday } = require("../services/stockDataClient");
const { addJob } = require("../services/mlJobQueueService");

/**
 * POST /api/predict
 * Body: { symbol, algorithm, startDate, endDate, options?, metadata? }
 *
 * Fetches historical data from stock-app-backend-data, then enqueues
 * an ML job with the data already populated. The frontend never needs
 * to pass raw price arrays.
 */
const enqueuePredict = async (request, response) => {
  const {
    symbol,
    algorithm,
    startDate,
    endDate,
    options = {},
    metadata = {},
  } = request.body || {};

  if (!symbol || !algorithm || !startDate || !endDate) {
    return response.status(400).json({
      error: "symbol, algorithm, startDate, and endDate are required",
    });
  }

  const datePattern = /^\d{4}-\d{2}-\d{2}$/;
  if (!datePattern.test(startDate) || !datePattern.test(endDate)) {
    return response.status(400).json({
      error: "startDate and endDate must be in YYYY-MM-DD format",
    });
  }

  if (new Date(startDate) > new Date(endDate)) {
    return response.status(400).json({
      error: "startDate must be before or equal to endDate",
    });
  }

  // Map shorthand algorithm name → executable ML service methods.
  const ALGORITHM_MAP = {
    linearRegression: {
      service: "linearRegressionService",
      method: "trainLinearRegression",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.targetDaysAhead,
        reqOptions.lookbackDays,
        reqOptions.testSplit,
      ],
    },
    institutionalLinearRegression: {
      service: "institutionalLinearRegressionService",
      method: "createInstitutionalLinearRegression",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.modelType || "multi_factor",
        reqOptions.startDate || startDate,
        reqOptions.endDate || endDate,
      ],
    },
    randomForest: {
      service: "randomForestService",
      method: "trainRandomForest",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.lookbackDays,
        reqOptions.nTrees,
        reqOptions.testSplit,
      ],
    },
    longTermRandomForest: {
      service: "longTermRandomForestService",
      method: "trainLongTermRandomForest",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.lookbackDays,
        reqOptions.targetDaysAhead,
        reqOptions.nTrees,
        reqOptions.testSplit,
        reqOptions.maxSelectedFeatures,
      ],
    },
    longTermNaiveBayes: {
      service: "longTermNaiveBayesService",
      method: "trainLongTermNaiveBayes",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.lookbackDays,
        reqOptions.targetDaysAhead,
        reqOptions.bins,
        reqOptions.testSplit,
      ],
    },
    ensemble: {
      service: "ensembleService",
      method: "trainEnsembleModel",
      buildArgs: (data, reqOptions = {}) => [data, reqOptions],
    },
    longTermEnsemble: {
      service: "longTermEnsembleService",
      method: "trainLongTermEnsembleModel",
      buildArgs: (data, reqOptions = {}) => [data, reqOptions],
    },
    longTermLSTM: {
      service: "longTermLSTMService",
      method: "trainLongTermLSTM",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.sequenceLength,
        reqOptions.targetDaysAhead,
        reqOptions.epochs,
        reqOptions.batchSize,
      ],
    },
    xgboost: {
      service: "xgBoostStockService",
      method: "trainXGBoostModel",
      buildArgs: (data, reqOptions = {}) => [
        data,
        reqOptions.marketFactors || null,
        reqOptions,
      ],
    },
    bottomPeakLSTM: {
      service: "bottomPeakDetectorService",
      method: "runBottomPeakDetector",
      buildArgs: (data, reqOptions = {}) => [data, reqOptions],
    },
  };

  const target = ALGORITHM_MAP[algorithm];
  if (!target) {
    return response.status(400).json({
      error: `Unknown algorithm "${algorithm}". Valid values: ${Object.keys(ALGORITHM_MAP).join(", ")}`,
    });
  }

  let stockData;
  let intradayMergeInfo = null;
  try {
    const stockFetch = await fetchStockDataWithIntraday(symbol, startDate, endDate, {
      includeIntraday: options.includeIntraday !== false,
    });
    stockData = stockFetch.data;
    intradayMergeInfo = {
      didMerge: Boolean(stockFetch.didMerge),
      reason: stockFetch.reason || null,
      intradaySnapshot: stockFetch.intradaySnapshot || null,
    };
  } catch (err) {
    return response.status(502).json({
      error: `Failed to fetch stock data: ${err.message}`,
    });
  }

  if (!stockData || stockData.length === 0) {
    return response.status(422).json({
      error: `No historical data found for ${symbol} between ${startDate} and ${endDate}`,
    });
  }

  const methodArgs = (target.buildArgs ? target.buildArgs(stockData, options) : [stockData]).filter(
    (arg) => arg !== undefined
  );

  console.log(
    `[predictController] Queueing ${algorithm} -> ${target.service}.${target.method} with ${methodArgs.length} arg(s)`
  );

  if (intradayMergeInfo) {
    console.log(
      `[predictController] Intraday merge for ${symbol.toUpperCase()}: ${intradayMergeInfo.didMerge ? "applied" : "not-applied"} (${intradayMergeInfo.reason || "n/a"})`
    );
  }

  const job = await addJob({
    service: target.service,
    method: target.method,
    args: methodArgs,
    metadata: {
      ...metadata,
      symbol: symbol.toUpperCase(),
      algorithm,
      startDate,
      endDate,
      intradayMerge: intradayMergeInfo,
    },
  });

  return response.status(202).json({
    ...job,
    stockData,
    intradayMerge: intradayMergeInfo,
  });
};

/**
 * GET /api/predict/stock-data?symbol=...&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&includeIntraday=true|false
 *
 * Frontend calls ML backend only. ML backend owns calls to data backend.
 */
const getStockDataForAnalysis = async (request, response) => {
  const {
    symbol,
    startDate,
    endDate,
    includeIntraday = "true",
  } = request.query || {};

  if (!symbol || !startDate || !endDate) {
    return response.status(400).json({
      error: "symbol, startDate, and endDate are required",
    });
  }

  const datePattern = /^\d{4}-\d{2}-\d{2}$/;
  if (!datePattern.test(startDate) || !datePattern.test(endDate)) {
    return response.status(400).json({
      error: "startDate and endDate must be in YYYY-MM-DD format",
    });
  }

  if (new Date(startDate) > new Date(endDate)) {
    return response.status(400).json({
      error: "startDate must be before or equal to endDate",
    });
  }

  try {
    const stockFetch = await fetchStockDataWithIntraday(symbol, startDate, endDate, {
      includeIntraday: String(includeIntraday).toLowerCase() !== "false",
    });

    return response.status(200).json({
      symbol: String(symbol).toUpperCase(),
      startDate,
      endDate,
      data: Array.isArray(stockFetch?.data) ? stockFetch.data : [],
      intradayMerge: {
        didMerge: Boolean(stockFetch?.didMerge),
        reason: stockFetch?.reason || null,
        intradaySnapshot: stockFetch?.intradaySnapshot || null,
      },
    });
  } catch (error) {
    return response.status(502).json({
      error: `Failed to fetch stock data: ${error.message}`,
    });
  }
};

module.exports = { enqueuePredict, getStockDataForAnalysis };
