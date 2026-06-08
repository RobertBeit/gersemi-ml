const express = require("express");
const cors = require("cors");
const mlJobRoutes = require("./routes/mlJobRoutes");
const predictRoutes = require("./routes/predictRoutes");

const app = express();

app.use(cors());
app.use(express.json({ limit: "20mb" }));

app.get("/", (_request, response) => {
  response.status(200).json({
    status: "ok",
    message: "stock-app-backend-ml is running",
  });
});

app.use("/api/ml-jobs", mlJobRoutes);
app.use("/api/predict", predictRoutes);

app.get("/health", (_request, response) => {
  response.status(200).json({ status: "ok" });
});

module.exports = app;
