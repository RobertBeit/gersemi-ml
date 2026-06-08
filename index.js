require("dotenv").config();
const app = require("./app");
const port = process.env.PORT || 3004;
const { buildBanner } = require("./services/buildInfo");

if (process.env.NODE_ENV === "production") {
  app.listen(port, () => {
    console.log(`${buildBanner} 🚀 startup`);
    console.log(`stock-app-backend-ml listening on http://0.0.0.0:${port}`);
  });
} else {
  const httpsLocalhost = require("https-localhost")();
  httpsLocalhost
    .getCerts()
    .then((certs) => {
      const https = require("https");
      https.createServer(certs, app).listen(port, () => {
        console.log(`${buildBanner} 🚀 startup`);
        console.log(`stock-app-backend-ml listening on https://localhost:${port}`);
      });
    })
    .catch((error) => {
      console.error("Failed to get SSL certificates:", error);
      app.listen(port, () => {
        console.log(`${buildBanner} 🚀 startup`);
        console.log(`stock-app-backend-ml listening on http://localhost:${port} (HTTP fallback)`);
      });
    });
}
