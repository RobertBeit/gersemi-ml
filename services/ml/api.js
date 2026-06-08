import stockDataClientModule from '../stockDataClient.js';

const { fetchStockData: fetchHistoricalStockData } = stockDataClientModule;

export const applyFullSplitAdjustment = (rawData) => {
  if (!rawData || rawData.length === 0) return [];

  return rawData.map((item) => {
    const rawOpen = item.raw_open ?? item.open ?? 0;
    const rawHigh = item.raw_high ?? item.high ?? 0;
    const rawLow = item.raw_low ?? item.low ?? 0;
    const rawClose = item.raw_close ?? item.close ?? 0;
    const adjustedClose = item.adjusted_close ?? item.close ?? rawClose;

    const adjustmentFactor = rawClose > 0 ? adjustedClose / rawClose : 1;

    return {
      date: item.date,
      open: rawOpen * adjustmentFactor,
      high: rawHigh * adjustmentFactor,
      low: rawLow * adjustmentFactor,
      close: adjustedClose,
      volume: item.volume ?? 0,
      raw_open: rawOpen,
      raw_high: rawHigh,
      raw_low: rawLow,
      raw_close: rawClose,
      adjusted_close: adjustedClose,
      split_coefficient: item.split_coefficient ?? 1,
      dividend_amount: item.dividend_amount ?? 0,
      adjustment_factor: adjustmentFactor,
      alpha_vantage_adjusted_close: adjustedClose,
    };
  });
};

export const fetchStockData = async (symbol, startDate, endDate) => {
  const rawData = await fetchHistoricalStockData(symbol, startDate, endDate);
  return applyFullSplitAdjustment(rawData);
};
