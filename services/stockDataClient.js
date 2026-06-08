const axios = require("axios");
const https = require("https");

const DATA_BACKEND_URL =
  process.env.STOCK_DATA_BACKEND_URL || "https://localhost:3003";

const trimTrailingSlash = (value) => value.replace(/\/+$/, "");

const buildCandidateBaseUrls = (configuredUrl) => {
  const normalized = trimTrailingSlash(configuredUrl);
  const candidates = [normalized];

  try {
    const parsed = new URL(normalized);
    const isLocalHost = ["localhost", "127.0.0.1"].includes(parsed.hostname);
    if (isLocalHost) {
      const alternateProtocol = parsed.protocol === "https:" ? "http:" : "https:";
      const fallback = `${alternateProtocol}//${parsed.host}`;
      if (!candidates.includes(fallback)) {
        candidates.push(fallback);
      }
    }
  } catch (_error) {
    // Keep the configured value only if URL parsing fails.
  }

  return candidates;
};

const buildAxiosConfig = (baseUrl, symbol, startDate, endDate) => {
  const config = {
    params: { symbol, startDate, endDate },
    timeout: 30000,
  };

  // Local HTTPS servers use self-signed certs during development.
  if (baseUrl.startsWith("https://localhost") || baseUrl.startsWith("https://127.0.0.1")) {
    config.httpsAgent = new https.Agent({ rejectUnauthorized: false });
  }

  return config;
};

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const isTransientNetworkError = (error) => {
  const code = String(error?.code || "").toUpperCase();
  const message = String(error?.message || "").toLowerCase();

  if (["ECONNRESET", "ETIMEDOUT", "EPIPE", "ECONNABORTED", "EAI_AGAIN"].includes(code)) {
    return true;
  }

  return (
    message.includes("socket hang up") ||
    message.includes("timeout") ||
    message.includes("network error")
  );
};

const formatDateInNewYork = (value) => {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(date);
};

const extractPricePoint = (entry) => {
  if (!entry || typeof entry !== "object") {
    return null;
  }

  const price = Number(entry.price);
  if (!Number.isFinite(price) || price <= 0) {
    return null;
  }

  const timestamp = entry.timestamp || entry.fetchedAt || entry.quoteTimestamp;
  const timeValue = timestamp ? new Date(timestamp).getTime() : NaN;

  return {
    price,
    timestamp: Number.isFinite(timeValue) ? timeValue : null,
  };
};

const buildIntradayOhlcSnapshot = (symbol, botStatus) => {
  const session = botStatus && typeof botStatus === "object" ? botStatus : null;
  if (!session) {
    return null;
  }

  const todayNy = formatDateInNewYork(new Date());
  const history = Array.isArray(session?.portfolio?.equityHistory)
    ? session.portfolio.equityHistory
    : [];

  const intradayPoints = history
    .map((entry) => {
      const point = extractPricePoint(entry);
      if (!point || point.timestamp === null) return null;
      return {
        ...point,
        dateNy: formatDateInNewYork(point.timestamp),
      };
    })
    .filter((point) => point && point.dateNy === todayNy)
    .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

  const latestQuotePoint = extractPricePoint(session.lastQuote);
  if (latestQuotePoint) {
    const latestDateNy = latestQuotePoint.timestamp !== null
      ? formatDateInNewYork(latestQuotePoint.timestamp)
      : todayNy;
    if (latestDateNy === todayNy) {
      intradayPoints.push({
        ...latestQuotePoint,
        dateNy: todayNy,
      });
    }
  }

  if (!intradayPoints.length) {
    return null;
  }

  intradayPoints.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

  const open = intradayPoints[0].price;
  const close = intradayPoints[intradayPoints.length - 1].price;
  const high = intradayPoints.reduce((max, p) => Math.max(max, p.price), Number.NEGATIVE_INFINITY);
  const low = intradayPoints.reduce((min, p) => Math.min(min, p.price), Number.POSITIVE_INFINITY);

  const pickRepresentativePrices = (points) => {
    if (!points.length) return [];
    const indexes = [0, Math.floor((points.length - 1) / 3), Math.floor((2 * (points.length - 1)) / 3), points.length - 1];
    const unique = [...new Set(indexes)].map((idx) => points[idx]?.price).filter((v) => Number.isFinite(v));
    return unique;
  };

  return {
    date: todayNy,
    open,
    high,
    low,
    close,
    representativePrices: pickRepresentativePrices(intradayPoints),
    pointCount: intradayPoints.length,
    source: "bot-delayed-intraday",
    symbol,
  };
};

const fetchQuoteFromDataBackend = async (symbol) => {
  const baseUrls = buildCandidateBaseUrls(DATA_BACKEND_URL);
  let lastError = null;

  for (const baseUrl of baseUrls) {
    const requestUrl = `${baseUrl}/api/stocks/quote/${encodeURIComponent(symbol)}`;

    for (let attempt = 1; attempt <= 3; attempt += 1) {
      try {
        const config = { timeout: 15000 };
        if (baseUrl.startsWith("https://localhost") || baseUrl.startsWith("https://127.0.0.1")) {
          config.httpsAgent = new https.Agent({ rejectUnauthorized: false });
        }
        const response = await axios.get(requestUrl, config);
        return response.data;
      } catch (error) {
        lastError = error;
        if (!isTransientNetworkError(error) || attempt === 3) {
          break;
        }
        await delay(1000 * attempt);
      }
    }
  }

  throw new Error(lastError?.message || "Failed to fetch quote from stock-app-backend-data");
};

const buildIntradaySnapshotFromQuote = (symbol, quote) => {
  if (!quote || typeof quote !== "object") {
    return null;
  }

  const price = Number(quote.price);
  const open = Number(quote.open);
  const high = Number(quote.high);
  const low = Number(quote.low);

  if (![price, open, high, low].every((value) => Number.isFinite(value) && value > 0)) {
    return null;
  }

  const todayNy = formatDateInNewYork(new Date());
  return {
    date: todayNy,
    open,
    high,
    low,
    close: price,
    representativePrices: [open, low, high, price],
    pointCount: 4,
    source: `data-backend-quote:${quote?.provider || "unknown"}`,
    symbol,
  };
};

const mergeIntradaySnapshot = (historicalRows, snapshot) => {
  if (!snapshot) {
    return { data: historicalRows, intradaySnapshot: null, didMerge: false };
  }

  const rows = Array.isArray(historicalRows) ? [...historicalRows] : [];
  const lastVolume = rows.length ? Number(rows[rows.length - 1].volume) : null;
  const mergedRow = {
    date: snapshot.date,
    open: snapshot.open,
    high: snapshot.high,
    low: snapshot.low,
    close: snapshot.close,
    volume: Number.isFinite(lastVolume) ? lastVolume : 0,
  };

  const existingIndex = rows.findIndex((row) => row.date === snapshot.date);
  if (existingIndex >= 0) {
    rows[existingIndex] = {
      ...rows[existingIndex],
      ...mergedRow,
    };
  } else {
    rows.push(mergedRow);
    rows.sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }

  return { data: rows, intradaySnapshot: snapshot, didMerge: true };
};

/**
 * Fetch historical stock data from stock-app-backend-data.
 * Returns an array of { date, open, high, low, close, volume } objects.
 *
 * @param {string} symbol  - e.g. "AAPL"
 * @param {string} startDate - "YYYY-MM-DD"
 * @param {string} endDate   - "YYYY-MM-DD"
 */
const fetchStockData = async (symbol, startDate, endDate) => {
  const baseUrls = buildCandidateBaseUrls(DATA_BACKEND_URL);
  let lastError = null;

  for (const baseUrl of baseUrls) {
    const requestUrl = `${baseUrl}/api/stocks`;

    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const requestConfig = buildAxiosConfig(baseUrl, symbol, startDate, endDate);

      try {
        const response = await axios.get(requestUrl, requestConfig);
        const data = response.data;

        // Normalise: backend may return { data: [...] } or a bare array
        if (Array.isArray(data)) {
          return data;
        }
        if (data && Array.isArray(data.data)) {
          return data.data;
        }

        throw new Error("Unexpected response shape from stock-app-backend-data");
      } catch (error) {
        lastError = error;
        console.error(
          `[stockDataClient] Request failed via ${requestUrl} (attempt ${attempt}/3): ${error.code || "UNKNOWN"} ${error.message}`
        );

        if (!isTransientNetworkError(error) || attempt === 3) {
          break;
        }

        await delay(1500 * attempt);
      }
    }
  }

  throw new Error(lastError?.message || "Failed to fetch stock data from stock-app-backend-data");
};

const fetchStockDataWithIntraday = async (symbol, startDate, endDate, options = {}) => {
  const includeIntraday = options.includeIntraday !== false;
  const historical = await fetchStockData(symbol, startDate, endDate);

  if (!includeIntraday) {
    return {
      data: historical,
      intradaySnapshot: null,
      didMerge: false,
      reason: "disabled",
    };
  }

  const todayNy = formatDateInNewYork(new Date());
  if (!todayNy || endDate < todayNy) {
    return {
      data: historical,
      intradaySnapshot: null,
      didMerge: false,
      reason: "date-range-before-today",
    };
  }

  try {
    const quote = await fetchQuoteFromDataBackend(symbol);
    const quoteSnapshot = buildIntradaySnapshotFromQuote(symbol, quote);
    const quoteMerged = mergeIntradaySnapshot(historical, quoteSnapshot);
    return {
      ...quoteMerged,
      reason: quoteMerged.didMerge ? quoteSnapshot?.source || "merged-from-data-backend-quote" : "quote-unavailable",
    };
  } catch (error) {
    console.warn(`[stockDataClient] Direct quote merge skipped for ${symbol}: ${error.message}`);
    return {
      data: historical,
      intradaySnapshot: null,
      didMerge: false,
      reason: "quote-unavailable",
    };
  }
};

module.exports = { fetchStockData, fetchStockDataWithIntraday };
