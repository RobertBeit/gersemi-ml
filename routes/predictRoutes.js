const express = require("express");
const { enqueuePredict, getStockDataForAnalysis } = require("../controllers/predictController");

const router = express.Router();

router.get("/stock-data", getStockDataForAnalysis);
router.post("/", enqueuePredict);

module.exports = router;
